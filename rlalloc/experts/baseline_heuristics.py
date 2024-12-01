import numpy as np
from typing import List, Tuple
import json
import torch

class ExpertPolicy:
    def __init__(self, config):
        self.config = config
        self.current_jobs = []
        
    def load_trace(self, trace_file: str):
        with open(trace_file, 'r') as f:
            self.trace_data = json.load(f)
            
    def _calculate_job_score(self, job_info: dict) -> float:
        duration = job_info.get('dur', 0)
        gpu_util = job_info.get('args', {}).get('gpu_usage', 0)
        cpu_util = job_info.get('args', {}).get('cpu_usage', 0)
        return duration * (gpu_util + cpu_util) / 2

    def shortest_job_first(self, state: torch.Tensor) -> np.ndarray:
        if not self.current_jobs:
            return np.array([0.5, 0.5, 0.5])
            
        jobs = sorted(self.current_jobs, key=self._calculate_job_score)
        shortest_job = jobs[0]
        
        gpu_alloc = min(1.0, max(0.0, shortest_job.get('args', {}).get('gpu_usage', 0.5) * 1.2))
        cpu_alloc = min(1.0, max(0.0, shortest_job.get('args', {}).get('cpu_usage', 0.5) * 1.2))
        mem_alloc = min(1.0, max(0.0, (gpu_alloc + cpu_alloc) / 2 * 1.1))
        
        return np.array([gpu_alloc, mem_alloc, cpu_alloc])
        
    def highest_utilization(self, state: torch.Tensor) -> np.ndarray:
        current_util = state[:, -1, 0].numpy()  # Get latest utilization
        
        gpu_alloc = min(1.0, max(0.0, float(current_util[0] * 1.3)))
        mem_alloc = min(1.0, max(0.0, float(current_util[1] * 1.3)))
        cpu_alloc = min(1.0, max(0.0, float(current_util[2] * 1.3)))
        
        return np.array([gpu_alloc, mem_alloc, cpu_alloc])

    def get_demonstration(self, state: torch.Tensor) -> np.ndarray:
        if self.config['expert']['type'] == 'sjf':
            return self.shortest_job_first(state)
        else:
            return self.highest_utilization(state)