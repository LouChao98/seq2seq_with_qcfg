# Seq2Seq + QCFG

The code heavily uses two packages:
* `pytorch-lighning` for most of workflows, https://www.pytorchlightning.ai/
* `hydra` for configuration, https://hydra.cc/

and is based on the template:
* https://github.com/ashleve/lightning-hydra-template

Train a model:
```shell
python train.py experiment=debug
```

See `README2.md`, which is the readme file of the above template, for more examples.