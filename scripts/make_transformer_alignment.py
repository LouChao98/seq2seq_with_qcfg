import pickle

import torch
from torch_scatter import scatter_mean
from transformers import AutoModel, AutoTokenizer

from src.datamodules.tsv_datamodule import TSVDataModule

# datatmodule = TSVDataModule(
#     train_file="data/MT/train_en-fr.txt",
#     dev_file="data/MT/dev_en-fr.txt",
#     test_file="data/MT/test_en-fr.txt",
#     transformer_tokenizer_name="xlm-roberta-base",
#     batch_size=128,
# )
# model = AutoModel.from_pretrained("xlm-roberta-base")
# output_path = 'data/MT/en-fr_xlm_base-cos.pt'
datatmodule = TSVDataModule(
    train_file="data/StylePTB/ATP/train.tsv",
    dev_file="data/StylePTB/ATP/valid.tsv",
    test_file="data/StylePTB/ATP/test.tsv",
    transformer_tokenizer_name="roberta-base",
    batch_size=128,
)
model = AutoModel.from_pretrained("roberta-base")
output_path = "data/StylePTB/ATP/roberta_base-cos.pt"

datatmodule.setup()
model.cuda()
model.eval()

cos = torch.nn.CosineSimilarity(3)

output = []
with torch.inference_mode():
    for batch in datatmodule.train_dataloader(keep_order=True):
        src = batch["src"]
        src_inps, src_mapping = datatmodule.make_transformer_input(src, blocked_ids={6})
        src_inps = {k: v.cuda() for k, v in src_inps.items()}
        h = model(**src_inps)[0]
        out = torch.zeros(
            batch["src_ids"].shape[0],
            batch["src_ids"].shape[1] + 1,
            h.shape[-1],
            device="cuda",
        )
        scatter_mean(h, src_mapping.cuda(), 1, out=out)
        src_h = out[:, 1:]

        tgt = batch["tgt"]
        tgt_inps, tgt_mapping = datatmodule.make_transformer_input(tgt, blocked_ids={6})
        tgt_inps = {k: v.cuda() for k, v in tgt_inps.items()}
        h = model(**tgt_inps)[0]
        out = torch.zeros(
            batch["tgt_ids"].shape[0],
            batch["tgt_ids"].shape[1] + 1,
            h.shape[-1],
            device="cuda",
        )
        scatter_mean(h, tgt_mapping.cuda(), 1, out=out)
        tgt_h = out[:, 1:]

        similarity = cos(src_h.unsqueeze(2), tgt_h.unsqueeze(1))
        similarity = similarity.cpu().numpy()
        similarity = [
            similarity[i, :sl, :tl].copy() for i, (sl, tl) in enumerate(zip(batch["src_lens"], batch["tgt_lens"]))
        ]
        output.extend(similarity)

with open(output_path, "wb") as f:
    pickle.dump(output, f)
