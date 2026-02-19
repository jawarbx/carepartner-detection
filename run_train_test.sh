#!/usr/bin/env bash
#SBATCH --time=300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G

# Load required modules
module load $CONDA_MODULE
conda activate $CONDA_ENV

NUM_GPUS=$(nvidia-smi -L | wc -l)
PORT=$(shuf -i25000-30000 -n1)

PYTHONPATH=. torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT src/train_ablation.py "$@"
