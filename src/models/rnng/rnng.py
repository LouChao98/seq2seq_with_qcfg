from sys import breakpointhook

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from src.models.src_parser.gold import tree2span
from src.utils.fn import annotate_snt_with_brackets, convert_annotated_str_to_nltk_str

from .action_dict import InOrderActionDict, TopDownActionDict
from .fixed_stack_in_order_models import FixedStackInOrderRNNG
from .fixed_stack_models import AttentionComposition as AttentionComposition2
from .fixed_stack_models import FixedStackRNNG
from .in_order_models import InOrderRNNG
from .models import AttentionComposition as AttentionComposition1
from .models import TopDownRNNG
from .modules import HackedEmbedding, HackedLinear
from .utils import (
    get_in_order_actions,
    get_in_order_max_stack_size,
    get_top_down_actions,
    get_top_down_max_stack_size,
)


class GeneralRNNG(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        strategy,
        fixed_stack,
        vocab,
        w_dim,
        h_dim,
        num_layers,
        composition,
        not_swap_in_order_stack,
        dropout,
        decode_use_particle_filter=False,
        particle_size=10000,
        original_reweight=False,
        beam_size=200,
        word_beam_size=20,
        shift_size=5,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.fixed_stack = fixed_stack

        self.action_dict_type = TopDownActionDict if self.strategy == "top_down" else InOrderActionDict
        self.action_dict = self.action_dict_type(["DUMMY"])

        model_args = {
            "action_dict": self.action_dict,
            "vocab": vocab,
            "padding_idx": 1,
            "w_dim": w_dim,
            "h_dim": h_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "attention_composition": composition == "attention",
        }

        if strategy == "top_down":
            if fixed_stack:
                model = FixedStackRNNG(**model_args)
            else:
                model = TopDownRNNG(**model_args)
            action_builder = get_top_down_actions
            stack_size_solver = get_top_down_max_stack_size
        elif strategy == "in_order":
            if fixed_stack:
                model = FixedStackInOrderRNNG(**model_args)
            else:
                model_args["do_swap_in_rnn"] = not not_swap_in_order_stack
                model = InOrderRNNG(**model_args)
            action_builder = get_in_order_actions
            stack_size_solver = get_in_order_max_stack_size

        self.model = model
        self.action_builder = action_builder
        self.stack_size_solver = stack_size_solver

        self.mlp_action_mlp_weight = nn.Linear(input_dim, w_dim)
        self.mlp_nt_emb = nn.Linear(input_dim, w_dim)
        self.mlp_composition_nt_emb = None
        self.mlp_composition_nt_emb2 = None

        self.decode_use_particle_filter = decode_use_particle_filter
        self.particle_size = particle_size
        self.original_reweight = original_reweight
        self.beam_size = beam_size
        self.word_beam_size = word_beam_size
        self.shift_size = shift_size

        # need + 2

        self.model.action_mlp = HackedLinear()
        self.special_action_embedding = nn.Embedding(self.action_dict.nt_begin_id(), input_dim)
        # breakpoint()
        # no need + 2
        if isinstance(self.model, (TopDownRNNG, InOrderRNNG)):

            self.model.nt_emb = HackedEmbedding()
            if isinstance(self.model.composition, AttentionComposition2):
                self.model.composition.nt_emb = HackedEmbedding()
                self.model.composition.nt_emb2 = HackedEmbedding()

                self.mlp_composition_nt_emb = nn.Linear(input_dim, w_dim)
                self.mlp_composition_nt_emb2 = nn.Linear(input_dim, w_dim * 2)
        else:
            self.model.rnng.nt_emb = HackedEmbedding()
            if isinstance(self.model.rnng.composition, AttentionComposition2):
                self.model.rnng.composition.nt_emb = HackedEmbedding()
                self.model.rnng.composition.nt_emb2 = HackedEmbedding()

                self.mlp_composition_nt_emb = nn.Linear(input_dim, w_dim)
                self.mlp_composition_nt_emb2 = nn.Linear(input_dim, w_dim * 2)

    def make_batch(self, spans, lengths, device):
        actions_ids = []
        stack_size = []
        for spans_item, l in zip(spans, lengths):
            actions_ids_item, stack_size_item = self.make_line(spans_item, l)
            actions_ids.append(actions_ids_item.to(device))
            stack_size.append(stack_size_item)
        return pad_sequence(actions_ids, batch_first=True), stack_size

    def make_line(self, spans, l):
        string = annotate_snt_with_brackets(["X"] * l, spans, "(", ")")
        mapping = {(span[0], span[1]): str(span[-1]) for span in spans}
        tree = convert_annotated_str_to_nltk_str(string, label_mapping=mapping)
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        actions = self.action_builder(" ".join(str(tree).split()))
        action_ids = [self.action_dict.a2i[a] for a in actions]

        if self.fixed_stack:
            stack_size = self.stack_size_solver(actions)
        else:
            stack_size = -1
        return torch.tensor(action_ids), stack_size

    def setup_nt(self, nt_features, nt_num_nodes):
        assert nt_features.shape[1] == nt_num_nodes
        nts = [str(i) for i in range(nt_num_nodes)]
        action_dict = self.action_dict_type(nts)

        self.action_dict = action_dict
        self.model.action_dict = action_dict
        self.model.num_actions = action_dict.num_actions()

        emb = torch.cat([self.special_action_embedding.weight.expand(len(nt_features), -1, -1), nt_features], dim=1)
        emb = self.mlp_action_mlp_weight(emb)
        self.model.action_mlp.setup_weight(emb)

        if isinstance(self.model, (TopDownRNNG, InOrderRNNG)):
            self.model.nt_emb.setup_weight(self.mlp_nt_emb(nt_features))
            if isinstance(self.model.composition, AttentionComposition1):
                self.model.composition.num_labels = action_dict.num_nts()
                self.model.composition.nt_emb.setup_weight(self.mlp_composition_nt_emb(nt_features))
                self.model.composition.nt_emb2.setup_weight(self.mlp_composition_nt_emb2(nt_features))
        else:
            self.model.rnng.action_dict = action_dict
            self.model.rnng.nt_emb.setup_weight(self.mlp_nt_emb(nt_features))
            if isinstance(self.model.rnng.composition, AttentionComposition2):
                self.model.rnng.composition.num_labels = action_dict.num_nts()
                self.model.rnng.composition.nt_emb.setup_weight(self.mlp_composition_nt_emb(nt_features))
                self.model.rnng.composition.nt_emb2.setup_weight(self.mlp_composition_nt_emb2(nt_features))

    def forward(self, token_ids, action_ids, max_stack_size, subword_end_mask=None):
        loss, a_loss, w_loss, stack = self.model(
            token_ids,
            action_ids,
            stack_size_bound=max_stack_size,
            subword_end_mask=subword_end_mask if subword_end_mask is not None else torch.ones_like(token_ids),
        )
        return loss, a_loss, w_loss, stack

    def decode(self, tgt, tgt_ids):
        if self.decode_use_particle_filter:
            parses, surprisals = self.decode_particle_filter(tgt_ids)
        else:
            parses, surprisals = self.decode_naive(tgt_ids)

        best_actions = [p[0][0] for p in parses]
        trees = []
        for i in range(len(best_actions)):
            tree = self.action_dict.build_tree_str(best_actions[i], tgt_ids[i], ["X"] * len(tgt_ids[i]), None)
            spans = tree2span(tree)
            tree = annotate_snt_with_brackets(tgt[i], spans)
            trees.append(tree)
        return trees

    def decode_particle_filter(self, tokens, max_stack_size=-1):
        return self.model.variable_beam_search(
            tokens,
            torch.ones_like(tokens),
            self.particle_size,
            self.original_reweight,
            stack_size_bound=max_stack_size,
        )

    def decode_naive(self, tokens, max_stack_size=-1):
        return self.model.word_sync_beam_search(
            tokens,
            torch.ones_like(tokens),
            self.beam_size,
            self.word_beam_size,
            self.shift_size,
            return_beam_history=False,
            stack_size_bound=max_stack_size,
        )

    def generate(
        self,
        batch_size,
        max_length,
        beam_size,
        word_beam_size=0,
        shift_size=0,
        stack_size_bound=100,
        device="cpu",
    ):
        return self.model.generate(
            batch_size,
            max_length,
            beam_size,
            word_beam_size,
            shift_size,
            stack_size_bound,
            device,
        )
