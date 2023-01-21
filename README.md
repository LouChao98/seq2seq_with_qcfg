# Seq2Seq + QCFG

The code heavily uses two packages:
* `pytorch-lightning` for most of workflows, https://www.pytorchlightning.ai/
* `hydra` for configuration, https://hydra.cc/

and is based on the template:
* https://github.com/ashleve/lightning-hydra-template

You can create the environment by run
```
bash create_env.sh
```

You can find data in https://github.com/yoonkim/neural-qcfg, which is also the repo I initially forked from.

See `README2.md`, which is the readme file of the above template, for more examples.


## Example

Run experiment on ATP using the reimplemented vanilla Neural QCFG:
```
python train.py experiment=styleptb
```

Run experiment on ATP using the E model:
```
python train.py experiment=styleptb_d1
```

Run experiment on ATP using the P model:
```
python train.py experiment=styleptb_d7
```
