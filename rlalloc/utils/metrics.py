import numpy as np
from collections import defaultdict
import torch
from typing import Dict, List
import json
import psutil
import GPUtil
import time

class MetricsLogger:
    def __init__(self, config):
        self.config = config
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.episode_start = None
        
    def start_episode(self):
        self.episode_start = time.time()
        self.episode_metrics = defaultdict(list)
    
    def log_step(self, state, action, reward, next_state, info):
        gpu = GPUtil.getGPUs()[0]
        
        # Convert numpy arrays/tensors to float values for metrics
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            state = state.float()
            
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float()
        elif isinstance(next_state, torch.Tensor):
            next_state = next_state.float()
            
        if isinstance(action, np.ndarray):
            action_np = action
        elif isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.array(action)
            
        metrics = {
            'gpu_util': gpu.memoryUtil,
            'gpu_temp': gpu.temperature,
            'cpu_util': psutil.cpu_percent(),
            'memory_util': psutil.virtual_memory().percent,
            'reward': float(reward),
            'action_mean': float(np.mean(action_np)),
            'action_std': float(np.std(action_np)),
            'state_mean': torch.mean(state).item(),
            'state_std': torch.std(state).item()
        }
        
        for k, v in metrics.items():
            self.episode_metrics[k].append(v)
            
        return metrics

    def end_episode(self, episode_num: int, total_reward: float, steps: int):
        episode_time = time.time() - self.episode_start
        
        episode_summary = {
            'episode': episode_num,
            'total_reward': total_reward,
            'steps': steps,
            'duration': episode_time,
            'avg_step_time': episode_time / steps
        }
        
        # Compute statistics over episode
        for k, v in self.episode_metrics.items():
            episode_summary.update({
                f'{k}_mean': float(np.mean(v)),
                f'{k}_std': float(np.std(v)),
                f'{k}_min': float(np.min(v)),
                f'{k}_max': float(np.max(v))
            })
            
        self.metrics['episodes'].append(episode_summary)
        return episode_summary

    def compute_convergence_metrics(self) -> Dict:
        rewards = [ep['total_reward'] for ep in self.metrics['episodes']]
        window = min(100, len(rewards))
        
        if len(rewards) <= window:
            return {
                'final_avg_reward': float(np.mean(rewards)),
                'best_avg_reward': float(np.mean(rewards)),
                'convergence_episode': 0,
                'total_steps': sum(ep['steps'] for ep in self.metrics['episodes']),
                'total_time': time.time() - self.start_time
            }
        
        return {
            'final_avg_reward': float(np.mean(rewards[-window:])),
            'best_avg_reward': float(max([np.mean(rewards[i:i+window]) 
                                  for i in range(len(rewards) - window)])),
            'convergence_episode': self._find_convergence_episode(rewards),
            'total_steps': sum(ep['steps'] for ep in self.metrics['episodes']),
            'total_time': time.time() - self.start_time
        }

    def _find_convergence_episode(self, rewards: List[float], window: int = 100, 
                                threshold: float = 0.95) -> int:
        window = min(window, len(rewards))
        if len(rewards) <= window:
            return 0
            
        max_avg = max([np.mean(rewards[i:i+window]) 
                      for i in range(len(rewards) - window)])
        threshold_value = max_avg * threshold
        
        for i in range(len(rewards) - window):
            if np.mean(rewards[i:i+window]) >= threshold_value:
                return i
        return len(rewards)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'episodes': self.metrics['episodes'],
                'convergence': self.compute_convergence_metrics()
            }, f)