#!/bin/bash

#SBATCH -J ner
#SBATCH -p critical
#SBATCH -N 1
#SBATCH -t 1-0:0:0
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=louchao@shanghaitech.edu.cn
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --nodelist=ai_gpu07
#,gpu_mem:12288


. /public/software/anaconda3/etc/profile.d/conda.sh
while [[ $CONDA_SHLVL -gt 0 ]]; do
    conda deactivate || break
done
# eval "$(conda shell.bash hook)"
conda activate ner

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
#module purge
#module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun python3 train.py logger=csv experiment=v4
