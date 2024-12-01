import torch
import numpy as np
import json
from typing import Dict, List, Tuple

class TraceProcessor:
    def __init__(self):
        self.kernel_events = []
        self.cpu_events = []
        self.memory_events = []
        
    def load_trace(self, trace_file: str):
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
            
        for event in trace_data.get('traceEvents', []):
            if event['cat'] == 'kernel':
                self.kernel_events.append(event)
            elif event['cat'] == 'cpu_op':
                self.cpu_events.append(event)
                
        self.device_props = trace_data.get('deviceProperties', [{}])[0]
        
    def get_resource_usage(self, timestamp: float) -> Dict[str, float]:
        gpu_util = 0.0
        cpu_util = 0.0
        mem_util = 0.0
        
        # Calculate GPU utilization
        for event in self.kernel_events:
            if event['ts'] <= timestamp <= event['ts'] + event.get('dur', 0):
                gpu_util = max(gpu_util, 
                             event['args'].get('registers per thread', 0) / 
                             self.device_props.get('regsPerBlock', 1))
                
        # Calculate CPU utilization
        for event in self.cpu_events:
            if event['ts'] <= timestamp <= event['ts'] + event.get('dur', 0):
                cpu_util += 0.1  # Approximate CPU usage per operation
                
        cpu_util = min(1.0, cpu_util)
        
        return {
            'gpu_util': gpu_util,
            'cpu_util': cpu_util,
            'mem_util': mem_util
        }
        
    def get_time_window(self, start_time: float, window_size: int = 20) -> torch.Tensor:
        states = []
        for i in range(window_size):
            timestamp = start_time + i * 1000  # 1ms intervals
            usage = self.get_resource_usage(timestamp)
            states.append([
                usage['gpu_util'],
                usage['gpu_util'],  # GPU memory approximation
                usage['cpu_util'],
                usage['mem_util'],
                i / window_size
            ])
            
        return torch.tensor(states).transpose(0, 1).unsqueeze(-1)