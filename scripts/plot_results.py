import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
import numpy as np

def plot_metrics():
    with open('results/training_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    episodes = range(len(metrics['episode_rewards']))
    rewards = metrics['episode_rewards']
    
    ax1.plot(episodes, rewards, label='Episode Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True)
    
    df = pd.DataFrame(metrics['convergence_metrics'])
    window = 100
    df['avg_reward_smooth'] = df['avg_reward'].rolling(window=window).mean()
    
    ax2.plot(df['episode'], df['avg_reward'], alpha=0.3, label=f'Average Reward')
    ax2.plot(df['episode'], df['avg_reward_smooth'], label=f'Smoothed ({window} episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Convergence Analysis')
    ax2.grid(True)
    ax2.legend()
    
    data = pd.DataFrame(metrics['resource_utilization'])
    if not data.empty:
        sns.boxplot(data=data, ax=ax3)
        ax3.set_xlabel('Resource Type')
        ax3.set_ylabel('Utilization %')
        ax3.set_title('Resource Utilization Distribution')
    
    plt.tight_layout()
    plt.savefig('results/training_plots.png')
    plt.close()

if __name__ == '__main__':
    plot_metrics()