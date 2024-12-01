# rlalloc/utils/profiling.py
import torch
import psutil
import time
import json
from datetime import datetime
import GPUtil
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ResourceMetrics:
    timestamp: str
    gpu_memory_used: float 
    gpu_memory_total: float
    gpu_utilization: float
    cpu_percent: float
    ram_used: float
    ram_total: float
    disk_used: float
    disk_total: float
    batch_throughput: float
    loss: float
    grad_norm: float

class ResourceProfiler:
    def __init__(self, log_dir: str = "data/profiles"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def get_gpu_metrics(self):
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            return {
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'gpu_util': gpu.load * 100
            }
        return {'memory_used': 0, 'memory_total': 0, 'gpu_util': 0}

    def get_system_metrics(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu,
            'ram_used': ram.used,
            'ram_total': ram.total,
            'disk_used': disk.used,
            'disk_total': disk.total
        }

    def profile_step(self, loss: float, model: torch.nn.Module, batch_size: int, 
                    step_start_time: float) -> ResourceMetrics:
        gpu_metrics = self.get_gpu_metrics()
        sys_metrics = self.get_system_metrics()
        step_time = time.time() - step_start_time
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        return ResourceMetrics(
            timestamp=datetime.now().isoformat(),
            gpu_memory_used=gpu_metrics['memory_used'],
            gpu_memory_total=gpu_metrics['memory_total'],
            gpu_utilization=gpu_metrics['gpu_util'],
            cpu_percent=sys_metrics['cpu_percent'],
            ram_used=sys_metrics['ram_used'],
            ram_total=sys_metrics['ram_total'],
            disk_used=sys_metrics['disk_used'],
            disk_total=sys_metrics['disk_total'],
            batch_throughput=batch_size/step_time,
            loss=loss.item(),
            grad_norm=grad_norm.item()
        )

    def save_metrics(self, metrics: Dict[str, Any], filename: str):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)