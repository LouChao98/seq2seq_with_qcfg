import pytorch_lightning as pl
import torch

from src.models.tgt_parser.struct.pcfg import PCFG

pl.seed_everything(1)

B = 2
N = 4
VOCAB = 3
TGT_PT = 2
SRC_PT = 4
TGT_NT = 2
SRC_NT = 2
r = 2
NT = TGT_NT * SRC_NT
PT = TGT_PT * SRC_PT

seq = torch.randint(0, VOCAB, (B, N))
seq_inst = seq[0].tolist()
n = seq.size(1)
x_expand = seq.unsqueeze(2).expand(B, n, PT).unsqueeze(3)

lens = torch.tensor([max(N - i, 2) for i in range(B)])

params = {
    "term": torch.randn(B, PT, VOCAB).log_softmax(-1),
    "root": torch.randn(B, NT).log_softmax(-1),
    "rule": torch.randn(B, NT, (NT + PT) ** 2)
    .log_softmax(-1)
    .view(B, NT, NT + PT, NT + PT),
}
terms = params["term"].unsqueeze(1).expand(B, n, PT, params["term"].size(2))
terms = torch.gather(terms, 3, x_expand).squeeze(3)
params["term"] = terms

pcfg = PCFG()
print(pcfg(params, lens))

params = {k: v[list(range(len(v) - 1, -1, -1))] for k, v in params.items()}
lens = lens[list(range(len(lens) - 1, -1, -1))]
pcfg = PCFG()
print(pcfg(params, lens))
