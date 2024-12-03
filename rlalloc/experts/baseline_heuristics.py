import numpy as np
import torch
import json
from typing import Dict

class ExpertPolicy:
    def __init__(self, config):
        self.config = config
        self.trace_data = None
        self.current_utilization = {'gpu': 0, 'cpu': 0, 'memory': 0}
        self.resource_patterns = []

    def load_trace(self, trace_file: str):
        with open(trace_file, 'r') as f:
            self.trace_data = json.load(f)
            self.process_trace_events()

    def process_trace_events(self):
        if not self.trace_data:
            self.resource_patterns = []
            return
            
        events = self.trace_data.get('traceEvents', [])
        self.resource_patterns = []
        
        for evt in events:
            if evt.get('cat') in ['kernel', 'cpu', 'memory']:
                pattern = {
                    'duration': evt.get('dur', 0),
                    'gpu_util': evt.get('args', {}).get('gpu_usage', 0),
                    'cpu_util': evt.get('args', {}).get('cpu_usage', 0),
                    'memory': evt.get('args', {}).get('memory', 0),
                    'type': evt.get('cat')
                }
                self.resource_patterns.append(pattern)

    def get_demonstration(self, state):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
            
        remaining_gpu = 1.0 - state[:10].max()
        remaining_cpu = 1.0 - state[10:20].max()
        remaining_mem = 1.0 - state[20:30].max()
        
        action = self.sjf_allocation(state) if np.random.random() < 0.5 else self.fcfs_allocation(state)
        return np.clip(action, 0, 1)

    def sjf_allocation(self, state):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
            
        current_util = state[:10].mean()
        remaining = 1.0 - current_util
        
        patterns = sorted(self.resource_patterns, key=lambda x: x['duration'])
        if patterns:
            pattern = patterns[0]
            proportional_alloc = np.array([
                pattern['gpu_util'] / (pattern['gpu_util'] + pattern['cpu_util'] + pattern['memory']),
                pattern['cpu_util'] / (pattern['gpu_util'] + pattern['cpu_util'] + pattern['memory']),
                pattern['memory'] / (pattern['gpu_util'] + pattern['cpu_util'] + pattern['memory'])
            ])
            return np.clip(proportional_alloc * remaining * 1.5, 0, remaining)
            
        return np.array([remaining * 0.3, remaining * 0.3, remaining * 0.3])

    def fcfs_allocation(self, state):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
            
        current_util = state[:10].mean() 
        remaining = 1.0 - current_util
        
        if self.resource_patterns:
            pattern = self.resource_patterns[0]
            proportional_alloc = np.array([
                pattern['gpu_util'] / (pattern['gpu_util'] + pattern['cpu_util'] + pattern['memory']),
                pattern['cpu_util'] / (pattern['gpu_util'] + pattern['cpu_util'] + pattern['memory']), 
                pattern['memory'] / (pattern['gpu_util'] + pattern['cpu_util'] + pattern['memory'])
            ])
            return np.clip(proportional_alloc * remaining * 1.2, 0, remaining)
            
        return np.array([remaining * 0.4, remaining * 0.4, remaining * 0.4])

    def rr_allocation(self, state):
        current_util = state[20:30].mean()
        return np.array([
            min(1.0, (1 - current_util) * 0.8),
            min(1.0, (1 - current_util) * 0.8),
            min(1.0, (1 - current_util) * 0.8)
        ])

    def _evaluate_action(self, state, action):
        util_reward = np.mean(action)
        efficiency = 1.0 - np.mean([
            abs(self.current_utilization['gpu'] - action[0]),
            abs(self.current_utilization['cpu'] - action[1]),
            abs(self.current_utilization['memory'] - action[2])
        ])
        return util_reward + 2 * efficiency

    def update_utilization(self, info: Dict):
        self.current_utilization = {
            'gpu': info.get('gpu_util', 0),
            'cpu': info.get('cpu_util', 0),
            'memory': info.get('mem_util', 0)
        }
        
    def cuda(self):
        self.device = 'cuda'
        return self

    def to(self, device):
        self.device = device
        return self