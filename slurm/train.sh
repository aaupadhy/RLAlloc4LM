#!/bin/bash
#SBATCH --job-name=TrainRL          
#SBATCH --time=24:00:00             
#SBATCH --ntasks=1                   
#SBATCH --ntasks-per-node=1         
#SBATCH --mem=100GB                  
#SBATCH --output=train_out.%j      
#SBATCH --gres=gpu:1                
#SBATCH --nodes=1                 
#SBATCH --partition=gpu              

source ~/.bashrc

conda activate RLAlloc4LM

export PYTHONPATH=$PYTHONPATH:/scratch/user/aaupadhy/college/DRL/RLAlloc4LM/

srun python -u ../scripts/train_rlalloc.py