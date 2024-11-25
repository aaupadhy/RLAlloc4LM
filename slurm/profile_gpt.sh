#!/bin/bash
#SBATCH --job-name=gpt2_profiling          # Job name
#SBATCH --time=02:30:00                    # Time limit
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Total number of tasks
#SBATCH --mem=80000M                       # Memory in MB
#SBATCH --output=../logs/profile_gpt2_%j.out  # Output file
#SBATCH --gpus=1                           # Total GPUs required
#SBATCH --gres=gpu:1                       # GPUs per task
#SBATCH --partition=gpu                    # GPU partition

source ~/.bashrc
conda activate RLAlloc4LM

python ../src/profile_gpt.py --batch_sizes 1 2 4 --seq_lengths 16 32 64
