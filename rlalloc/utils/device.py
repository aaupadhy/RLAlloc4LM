import os
import torch

def get_device():
   if torch.cuda.is_available():
       if 'SLURM_JOB_ID' in os.environ:
           return f"cuda:{int(os.environ.get('SLURM_LOCALID', 0))}"
       return "cuda:0"
   return "cpu"

def to_device(tensor, device=None):
   if device is None:
       device = get_device()
   return tensor.to(device)