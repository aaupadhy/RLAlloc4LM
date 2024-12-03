import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

def moving_average(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def calculate_stats(data, window):
    mean = moving_average(data, window)
    std = np.array([np.std(data[max(0, i-window):min(i+1, len(data))]) 
                   for i in range(len(mean))])
    return mean, std

def plot_comparison():
   metrics = json.load(open('results/metrics_final.json'))
   baseline = json.load(open('results/baseline_rewards.json'))
   sac_only = json.load(open('results/sac_rewards.json'))

   episodes = metrics['episodes']
   window = 5
   fig, axes = plt.subplots(2, 2, figsize=(15, 12))

   # Performance comparison  
   rewards = [ep['total_reward'] for ep in episodes]
   for data, label in [(rewards, 'SAC+IL'), (baseline, 'Baseline'), (sac_only, 'SAC Only')]:
       mean, std = calculate_stats(np.array(data), window)
       axes[0,0].plot(mean, label=label)
       axes[0,0].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
   axes[0,0].set_title('Performance Comparison')
   axes[0,0].legend()

   # Resource utilization
   for metric, label in [('gpu_util_mean', 'GPU'), ('cpu_util_mean', 'CPU'), ('memory_util_mean', 'MEMORY')]:
       data = np.array([ep[metric] for ep in episodes])
       mean, std = calculate_stats(data, window)
       axes[0,1].plot(mean, label=label)
       axes[0,1].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
   axes[0,1].set_title('Resource Utilization')
   axes[0,1].legend()

   # Training metrics
   steps = [ep['steps'] for ep in episodes]
   mean, std = calculate_stats(np.array(steps), window)
   axes[1,0].plot(mean)
   axes[1,0].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
   axes[1,0].set_title('Steps per Episode')

   losses = [ep.get('critic_loss_mean', 0) for ep in episodes] 
   qvals = [ep.get('avg_q_value', 0) for ep in episodes]
   for data, label in [(losses, 'Critic Loss'), (qvals, 'Q-Value')]:
       mean, std = calculate_stats(np.array(data), window)
       axes[1,1].plot(mean, label=label)
   axes[1,1].set_title('Training Metrics')
   axes[1,1].legend()

   plt.tight_layout()
   plt.savefig('results/performance_analysis.png')

if __name__ == '__main__':
    print("Started Plotting")
    plot_comparison()