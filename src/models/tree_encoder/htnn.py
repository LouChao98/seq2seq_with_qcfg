from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence

from src.utils.fn import spans2tree

# Based on https://github.com/GreyChou98/HTNN/blob/cbbf51551f1bb6ce7aa9d8e49496e8b8f4c1b6d2/srl_biaffine/treelstm_new.py


class TreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, dropout, first_layer=False, comb_method="attn"):
        super(TreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.dropout = nn.Dropout(dropout)
        self.first_layer = first_layer
        self.comb_method = comb_method
        self.iou_w = nn.Linear(self.mem_dim, 3 * self.mem_dim, bias=False)
        self.iou_l = nn.Linear(self.mem_dim, 3 * self.mem_dim, bias=False)
        self.iou_r = nn.Linear(self.mem_dim, 3 * self.mem_dim, bias=False)
        self.iou_p = nn.Linear(self.mem_dim, 3 * self.mem_dim, bias=False)
        self.w_fl = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.w_fr = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.w_fp = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.u_fl_l = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.u_fr_l = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.u_fp_l = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.u_fl_r = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.u_fr_r = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.u_fp_r = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.u_fl_p = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.u_fr_p = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.u_fp_p = nn.Linear(self.mem_dim, self.mem_dim, bias=False)

        # self.bi = nn.Parameter(torch.FloatTensor(self.mem_dim))
        # self.bf = nn.Parameter(torch.FloatTensor(self.mem_dim))
        # self.bo = nn.Parameter(torch.FloatTensor(self.mem_dim))
        # self.bu = nn.Parameter(torch.FloatTensor(self.mem_dim))

        if self.first_layer:
            self.project_c = nn.Linear(self.in_dim, self.mem_dim)
            self.layernorm1 = nn.LayerNorm(self.mem_dim, eps=1e-6)
            self.project_h = nn.Linear(self.in_dim, self.mem_dim)
            self.layernorm2 = nn.LayerNorm(self.mem_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(self.mem_dim, eps=1e-6)
        self.layernorm4 = nn.LayerNorm(self.mem_dim, eps=1e-6)
        self.layernorm5 = nn.LayerNorm(self.mem_dim, eps=1e-6)

        self.layernorm_i = nn.LayerNorm(self.mem_dim, eps=1e-6)
        self.layernorm_o = nn.LayerNorm(self.mem_dim, eps=1e-6)
        self.layernorm_u = nn.LayerNorm(self.mem_dim, eps=1e-6)

        self.attn_combine = nn.Linear(2 * self.mem_dim, self.mem_dim)
        self.v = nn.Linear(self.mem_dim, 1, bias=False)

    def forward(self, span_repr_h, child_rel, span_repr_c=None):
        """
        span_repr: [B, num_spans, D]
        child_rel: [B, num_rels * 3]
        return : [B, num_spans, D]
        """

        if self.first_layer:
            # span_rel_repr_h = self.get_child_rel_repr(child_rel, span_repr_h)  # [(B * L_3), 3, feature_dim]
            span_repr_c = self.layernorm1(self.project_c(span_repr_h))
            span_repr_h = self.layernorm2(self.project_h(span_repr_h))
            # feature_c = self.get_child_rel_repr(child_rel)

        feature_c = self.get_child_rel_repr(child_rel, span_repr_c)
        feature_h = self.get_child_rel_repr(child_rel, span_repr_h)

        batch_p = (feature_h[:, 0, :], feature_c[:, 0, :])
        batch_l = (feature_h[:, 1, :], feature_c[:, 1, :])
        batch_r = (feature_h[:, 2, :], feature_c[:, 2, :])

        new_ph, new_pc = self.lr2p(batch_p, batch_l, batch_r)
        new_rh, new_rc = self.lp2r(batch_p, batch_l, batch_r)
        new_lh, new_lc = self.rp2l(batch_p, batch_l, batch_r)

        span_rel_repr_h = torch.stack([new_ph, new_lh, new_rh], dim=1)
        span_rel_repr_c = torch.stack([new_pc, new_lc, new_rc], dim=1)

        rearrange_repr_h = self.rearrage(span_rel_repr_h, child_rel, span_repr_h.size(1))
        rearrange_repr_c = self.rearrage(span_rel_repr_c, child_rel, span_repr_h.size(1))
        if self.comb_method == "attn":
            span_repr_h = self.combine_feature_attn(rearrange_repr_h, span_repr_h)
            span_repr_c = self.combine_feature_attn(rearrange_repr_c, span_repr_c)
        elif self.comb_method == "attn2wise":
            span_repr_h = self.combine_feature_attn2wise(rearrange_repr_h, span_repr_h)
            span_repr_c = self.combine_feature_attn2wise(rearrange_repr_c, span_repr_c)
        else:
            span_repr_h = self.combine_feature_avg(rearrange_repr_h, span_repr_h)
            span_repr_c = self.combine_feature_avg(rearrange_repr_c, span_repr_c)

        return span_repr_h, span_repr_c

    def lr2p(self, batch_p, batch_l, batch_r):
        """
        batch_p, batch_l, batch_r: [(B * 2L_3), feature_dim]
        return: batch_p
        """
        batch_ph, batch_pc = batch_p
        batch_lh, batch_lc = batch_l
        batch_rh, batch_rc = batch_r

        iou = self.iou_w(batch_ph) + self.iou_l(batch_lh) + self.iou_r(batch_rh)

        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = self.dropout(F.sigmoid(i))
        o = self.dropout(F.sigmoid(o))
        u = self.dropout(F.tanh(u))

        fl = self.dropout(F.sigmoid(self.w_fl(batch_ph) + self.u_fl_l(batch_lh) + self.u_fl_r(batch_rh)))
        fr = self.dropout(F.sigmoid(self.w_fl(batch_ph) + self.u_fr_l(batch_lh) + self.u_fr_r(batch_rh)))

        # check = int((fl != fl).sum())
        # if (check > 0):
        #     logging.info("tree layer fl contains Nan")
        # else:
        #     logging.info("tree layer fl does not contain Nan, it might be other problem")
        #
        # check = int((fr != fr).sum())
        # if (check > 0):
        #     logging.info("tree layer fr contains Nan")
        # else:
        #     logging.info("tree layer fr does not contain Nan, it might be other problem")

        new_c = self.layernorm3(torch.mul(i, u) + torch.mul(fl, batch_lc) + torch.mul(fr, batch_rc))
        new_h = torch.mul(o, F.leaky_relu_(new_c))

        return new_h, new_c

    def lp2r(self, batch_p, batch_l, batch_r):
        """
        batch_p, batch_l, batch_r: [(B * 2L_3), feature_dim]
        return: batch_r
        """
        batch_ph, batch_pc = batch_p
        batch_lh, batch_lc = batch_l
        batch_rh, batch_rc = batch_r

        iou = self.iou_w(batch_rh) + self.iou_l(batch_lh) + self.iou_p(batch_ph)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        # i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)
        i = self.dropout(F.sigmoid(i))
        o = self.dropout(F.sigmoid(o))
        u = self.dropout(F.tanh(u))

        fl = self.dropout(F.sigmoid(self.w_fl(batch_rh) + self.u_fl_l(batch_lh) + self.u_fl_p(batch_ph)))
        fp = self.dropout(F.sigmoid(self.w_fp(batch_rh) + self.u_fp_l(batch_lh) + self.u_fp_p(batch_ph)))

        new_c = self.layernorm4(torch.mul(i, u) + torch.mul(fl, batch_lc) + torch.mul(fp, batch_pc))
        new_h = torch.mul(o, F.leaky_relu_(new_c))

        return new_h, new_c

    def rp2l(self, batch_p, batch_l, batch_r):
        """
        batch_p, batch_l, batch_r: [(B * 2L_3), feature_dim]
        return: batch_l
        """
        batch_ph, batch_pc = batch_p
        batch_lh, batch_lc = batch_l
        batch_rh, batch_rc = batch_r

        iou = self.iou_w(batch_lh) + self.iou_r(batch_rh) + self.iou_p(batch_ph)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        # i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)
        i = self.dropout(F.sigmoid(i))
        o = self.dropout(F.sigmoid(o))
        u = self.dropout(F.tanh(u))

        fr = self.dropout(F.sigmoid(self.w_fr(batch_lh) + self.u_fr_r(batch_rh) + self.u_fr_p(batch_ph)))
        fp = self.dropout(F.sigmoid(self.w_fp(batch_lh) + self.u_fp_r(batch_rh) + self.u_fp_p(batch_ph)))

        new_c = self.layernorm5(torch.mul(i, u) + torch.mul(fr, batch_rc) + torch.mul(fp, batch_pc))
        new_h = torch.mul(o, F.leaky_relu_(new_c))

        return new_h, new_c

    def get_child_rel_repr(self, child_rel, span_repr):
        """
        child_rel: [B, num_rels * 3]
        span_repr: [B, num_spans, D]
        returns: [B * num_rels, 3, D]
        """
        feature_dim = span_repr.size()[-1]
        zero_feature = torch.zeros(feature_dim, device=child_rel.device)
        output = []
        for i, ex in enumerate(child_rel):
            for index in ex:
                if index == -1:
                    output.append(zero_feature)
                else:
                    output.append(span_repr[i][index])
        output = torch.stack(output, dim=0)
        output = output.view(-1, 3, feature_dim)
        return output

    def rearrage(self, span_rel_repr, child_rel, max_num_spans):
        """
        :param span_rel_repr: [B * num_rels, 3, D]
        :param child_rel: [B, num_rels * 3]
        :param max_num_spans: int
        :return: [B, max_num_spans, 2, D]
        """
        batch_size, num_rels = child_rel.size()
        num_rels = num_rels // 3
        child_rel = child_rel.view(batch_size, num_rels, 3)
        repr_dim = span_rel_repr.size(-1)
        new_shape = [max_num_spans, 2, repr_dim]
        rel_mask = torch.sum(child_rel, -1) > 0

        span_rel_repr = span_rel_repr.view(batch_size, num_rels, 3, repr_dim)

        out = []
        for batch in range(batch_size):
            new = torch.zeros(new_shape, device=span_rel_repr.device)
            for i, (a, b, c) in enumerate(child_rel[batch]):
                if rel_mask[batch][i]:
                    new[a][0] = span_rel_repr[batch][i][0]
                    new[b][1] = span_rel_repr[batch][i][1]
                    new[c][1] = span_rel_repr[batch][i][2]
            out.append(new)
        out = torch.stack(out, dim=0)
        return out

    def combine_feature_attn(self, rearrage_repr, span_repr):
        """
        :param rearrage_repr: [B, max_num_spans, 2, D]
        :param span_repr: [B, num_spans, D]
        :return: [B, num_spans, D]
        """
        batch_size, num_spans, feature_dim = span_repr.size()
        rearrage_repr = rearrage_repr.view(-1, 2, feature_dim)  # [B * num_spans, 2, D]
        rearrage_repr = torch.cat(
            (span_repr.view(-1, feature_dim).unsqueeze(1), rearrage_repr), dim=1
        )  # [B * num_spans, 3, D]
        span_repr = span_repr.view(-1, feature_dim)
        span_repr = span_repr.unsqueeze(1).repeat(1, 3, 1).contiguous()
        combined = torch.cat((span_repr, rearrage_repr), dim=-1)

        energy = F.tanh(self.attn_combine(combined))  # [B * num_spans, 3, D]
        # energy = energy.transpose(2, 1).contiguous()  # [B * num_spans, D, 3]
        energy = self.v(energy)  # [B * num_spans, 3, 1]
        energy = energy.squeeze(-1)  # [B * num_spans, 3]
        attn_mask = torch.sum(rearrage_repr, dim=-1) == 0
        energy = energy.masked_fill(attn_mask, -1e18)
        scores = F.softmax(energy, dim=-1).unsqueeze(1)  # [B * num_spans, 1, 3]
        output = torch.bmm(scores, rearrage_repr)  # [B * num_spans, 1, D]
        output = output.view(batch_size, num_spans, feature_dim)

        return output

    def combine_feature_attn2wise(self, rearrage_repr, span_repr):
        """
        :param rearrage_repr: [B, max_num_spans, 2, D]
        :param span_repr: [B, num_spans, D]
        :return: [B, num_spans, D]
        """
        batch_size, num_spans, feature_dim = span_repr.size()
        rearrage_repr = rearrage_repr.view(-1, 2, feature_dim)  # [B * num_spans, 2, D]
        # rearrage_repr = torch.cat((span_repr.view(-1, feature_dim).unsqueeze(1),
        #                            rearrage_repr), dim=1)              # [B * num_spans, 3, D]
        span_repr = span_repr.view(-1, feature_dim)
        span_repr = span_repr.unsqueeze(1).repeat(1, 2, 1).contiguous()
        combined = torch.cat((span_repr, rearrage_repr), dim=-1)

        energy = F.tanh(self.attn_combine(combined))  # [B * num_spans, 2, D]
        # energy = energy.transpose(2, 1).contiguous()  # [B * num_spans, D, 2]
        energy = self.v(energy)  # [B * num_spans, 2, 1]
        energy = energy.squeeze(-1)  # [B * num_spans, 2]
        attn_mask = torch.sum(rearrage_repr, dim=-1) == 0
        energy = energy.masked_fill(attn_mask, -1e18)
        scores = F.softmax(energy, dim=-1).unsqueeze(1)  # [B * num_spans, 1, 2]
        output = torch.bmm(scores, rearrage_repr)  # [B * num_spans, 1, D]
        output = output.view(batch_size, num_spans, feature_dim)

        return output

    def combine_feature_avg(self, rearrage_repr, span_repr):
        """
        :param rearrage_repr: [B, max_num_spans, 2, D]
        :param span_repr: [B, num_spans, D]
        :return: [B, num_spans, D]
        """
        batch_size, num_spans, feature_dim = span_repr.size()
        rearrage_repr = rearrage_repr.view(-1, 2, feature_dim)  # [B * num_spans, 2, D]
        rearrage_repr = torch.cat(
            (span_repr.view(-1, feature_dim).unsqueeze(1), rearrage_repr), dim=1
        )  # [B * num_spans, 3, D]
        avg_repr = torch.mean(rearrage_repr, dim=1, keepdim=False)  # [B * num_spans, D]

        output = avg_repr.view(batch_size, num_spans, feature_dim)

        return output


class LayerwiseTreeLSTM_new(nn.Module):
    def __init__(self, dim, num_layers, span_net, dropout, comb_method="attn"):
        super(LayerwiseTreeLSTM_new, self).__init__()

        self.num_layers = num_layers
        self.span_net = instantiate(span_net, input_dim=dim, proj_dim=dim)
        self.feature_dim = dim
        self.dropout = dropout
        self.comb_method = comb_method

        self.tree_layer_first = TreeLSTM(
            self.feature_dim,
            self.feature_dim,
            self.dropout,
            first_layer=True,
            comb_method=self.comb_method,
        )
        self.tree_layers = nn.ModuleList(
            [
                TreeLSTM(
                    self.feature_dim,
                    self.feature_dim,
                    self.dropout,
                    first_layer=False,
                    comb_method=self.comb_method,
                )
                for i in range(self.num_layers - 1)
            ]
        )

        # self.tree_layer1 = TreeLSTM(self.feature_dim, self.feature_dim)
        # self.tree_layer2 = TreeLSTM(self.feature_dim, self.feature_dim)

    def forward(self, x, lens, spans):
        hyperedges = []
        spans_processed = []
        spans_list = []
        for bidx, spans_item in enumerate(spans):
            parents_item = spans2tree(spans_item)

            parent2children = defaultdict(list)
            for i, pj in enumerate(parents_item):
                if pj >= 0:
                    parent2children[pj].append(i)

            hyperedges_inst = []
            for pj, children in parent2children.items():
                assert len(children) == 2
                a, b = children
                if spans_item[a][0] < spans_item[b][0]:
                    hyperedges_inst.append((pj, a, b))
                else:
                    hyperedges_inst.append((pj, b, a))

            hyperedges.append(torch.tensor(hyperedges_inst, device=x.device))
            spans_processed.append(torch.tensor([(l, r - 1) for l, r, *_ in spans_item], device=x.device))
            spans_list.append(spans_item)

        spans = pad_sequence(spans_processed, batch_first=True).flatten(1)
        rel = pad_sequence(hyperedges, batch_first=True).flatten(1)
        return self._forward(spans, x, rel, None), spans_list

    def _forward(self, spans, encoded_input, child_rel, tag_repr):
        """
        spans: [B, num_spans * 2]  use with shape (-1, 2)
        encoded_input: [B, seq_len, D]
        child_rel: [B, num_rels * 3]
        tags: [B, num_spans]
        returns: [B, num_spans, D]
        """
        span_repr = self.calc_span_repr(encoded_input, spans)

        if tag_repr is not None:
            span_repr = torch.cat((span_repr, tag_repr), dim=-1)

        span_repr_h, span_repr_c = self.tree_layer_first(span_repr, child_rel)
        for i in range(self.num_layers - 1):
            span_repr_h, span_repr_c = self.tree_layers[i](span_repr_h, child_rel, span_repr_c)

        #
        # span_rel_repr = self.get_child_rel_repr(child_rel, span_repr)
        # span_rel_repr = self.tree_layer1(span_rel_repr)
        # span_repr = self.combine_feature(span_rel_repr, child_rel, span_repr.size())
        #
        # span_rel_repr = self.get_child_rel_repr(child_rel, span_repr)
        # span_rel_repr = self.tree_layer2(span_rel_repr)
        # span_repr = self.combine_feature(span_rel_repr, child_rel, span_repr.size())

        return span_repr_h

    def calc_span_repr(self, encoded_input, spans):
        """
        encoded_input: [B, seq_len, D]
        spans: [B, num_spans * 2]
        returns: [B, num_spans, D]
        """
        batch_size, num_spans = spans.size()
        num_spans = int(num_spans / 2)
        encoder_size = encoded_input.size(-1)
        encoded_input_expand = encoded_input.unsqueeze(1).expand(-1, num_spans, -1, -1).contiguous()
        spans = spans.view(-1, 2)
        encoded_input_expand = encoded_input_expand.view(spans.size(0), -1, encoder_size)

        span_start, span_end = spans[:, 0], spans[:, 1]
        span_repr = self.span_net(encoded_input_expand, span_start, span_end)
        span_repr = span_repr.view(batch_size, num_spans, -1)

        return span_repr

    def get_output_dim(self):
        return self.feature_dim


# if __name__ == "__main__":
#     from src.models.components.span import AttnSpanRepr

#     spans_inst = [(0, 4), (0, 2), (3, 4), (1, 2)]
#     spans = [spans_inst[:], spans_inst[:]]
#     x = torch.randn(2, 7, 5)

#     span_net = AttnSpanRepr(5, use_proj=True, proj_dim=5)
#     model = LayerwiseTreeLSTM_new(
#         num_layers=1, feature_dim=5, span_net=span_net, dropout=0.2, comb_method="attn"
#     )
#     features, spans = model(x, None, spans)
#     print(spans)
