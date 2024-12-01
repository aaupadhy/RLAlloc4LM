#!/bin/bash
#SBATCH --job-name=JobExample5       # Set the job name to "JobExample5"
#SBATCH --time=06:30:00              # Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=8                   # Request 8 tasks (1 task per GPU)
#SBATCH --ntasks-per-node=2          # Request 4 tasks (4 GPUs) per node
#SBATCH --mem=50GB                  # Request 2560MB (2.5GB) per node
#SBATCH --output=Example5Out.%j      # Send stdout/err to "Example5Out.[jobID]"
#SBATCH --gres=gpu:2                 # Request 4 GPUs per node
#SBATCH --nodes=4                    # Request 2 nodes (total 8 GPUs)
#SBATCH --partition=gpu              # Request the GPU partition/queue


source ~/.bashrc
conda activate final

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805

srun python -u check.py \