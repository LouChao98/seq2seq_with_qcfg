#!/bin/bash

. /public/software/anaconda3/etc/profile.d/conda.sh
while [[ $CONDA_SHLVL -gt 0 ]]; do
    conda deactivate || break
done
# eval "$(conda shell.bash hook)"
conda activate ner

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
