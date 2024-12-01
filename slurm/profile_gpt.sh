#!/bin/bash
#SBATCH --job-name=ProfileGPT       # Set the job name to "JobExample5"
#SBATCH --time=02:30:00              # Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   # Request 8 tasks (1 task per GPU)
#SBATCH --ntasks-per-node=1         # Request 4 tasks (4 GPUs) per node
#SBATCH --mem=50GB                  # Request 2560MB (2.5GB) per node
#SBATCH --output=Example5Out.%j      # Send stdout/err to "Example5Out.[jobID]"
#SBATCH --gres=gpu:1                # Request 4 GPUs per node
#SBATCH --nodes=1                 # Request 2 nodes (total 8 GPUs)
#SBATCH --partition=gpu              # Request the GPU partition/queue


source ~/.bashrc

conda activate RLAlloc4LM

export PYTHONPATH=$PYTHONPATH:/scratch/user/aaupadhy/college/DRL/RLAlloc4LM/

srun python -u ../scripts/profile_gpt.py