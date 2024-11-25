#!/bin/bash
#SBATCH --job-name=RLAlloc-LLM
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

module load cuda/11.8
srun python -m torch.distributed.launch --nproc_per_node=4 src/train_rl.py

