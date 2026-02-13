#!/usr/bin/env bash
#SBATCH --time=1-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G

# Load required modules 
module load $CONDA_MODULE
conda activate $CONDA_ENV

python src/shap_analysis_parallel.py
