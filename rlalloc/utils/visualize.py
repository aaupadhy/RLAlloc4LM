import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import json

class VisualizationGenerator:
    def __init__(self, metrics_path: str):
        with open(metrics_path, 'r') as f:
            self.data = json.load(f)
            
    def plot_learning_curves(self, save_path: str):
        df = pd.DataFrame(self.data['episodes'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Reward curve
        self._plot_reward_curve(df, axes[0, 0])
        
        # Resource utilization
        self._plot_resource_util(df, axes[0, 1])
        
        # Action statistics
        self._plot_action_stats(df, axes[1, 0])
        
        # Performance metrics
        self._plot_performance(df, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def generate_comparison_plots(self, baseline_metrics: str, save_path: str):
        with open(baseline_metrics, 'r') as f:
            baseline = json.load(f)
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Reward comparison
        df_rl = pd.DataFrame(self.data['episodes'])
        df_baseline = pd.DataFrame(baseline['episodes'])
        
        self._plot_reward_comparison(df_rl, df_baseline, axes[0])
        self._plot_resource_comparison(df_rl, df_baseline, axes[1])
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_reward_curve(self, df: pd.DataFrame, ax: plt.Axes):
        window = 100
        df['reward_smooth'] = df['total_reward'].rolling(window=window).mean()
        
        ax.plot(df['episode'], df['total_reward'], alpha=0.3, label='Episode Reward')
        ax.plot(df['episode'], df['reward_smooth'], label=f'Smoothed ({window} ep)')
        ax.axhline(y=self.data['convergence']['final_avg_reward'], 
                  color='r', linestyle='--', label='Final Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Progress')
        ax.grid(True)
        ax.legend()

    def _plot_resource_util(self, df: pd.DataFrame, ax: plt.Axes):
        resources = ['gpu_util_mean', 'cpu_util_mean', 'memory_util_mean']
        df[resources].plot(ax=ax)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Utilization %')
        ax.set_title('Resource Utilization')
        ax.grid(True)
        ax.legend()

    def _plot_action_stats(self, df: pd.DataFrame, ax: plt.Axes):
        ax.plot(df['episode'], df['action_mean'], label='Mean')
        ax.fill_between(df['episode'], 
                       df['action_mean'] - df['action_std'],
                       df['action_mean'] + df['action_std'],
                       alpha=0.3, label='±1 std')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Action Value')
        ax.set_title('Action Statistics')
        ax.grid(True)
        ax.legend()

    def _plot_performance(self, df: pd.DataFrame, ax: plt.Axes):
        metrics = ['avg_step_time', 'steps']
        df[metrics].plot(ax=ax)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics')
        ax.grid(True)
        ax.legend()

    def _plot_reward_comparison(self, df_rl: pd.DataFrame, 
                              df_baseline: pd.DataFrame, ax: plt.Axes):
        window = 100
        df_rl['reward_smooth'] = df_rl['total_reward'].rolling(window=window).mean()
        df_baseline['reward_smooth'] = df_baseline['total_reward'].rolling(window=window).mean()
        
        ax.plot(df_rl['episode'], df_rl['reward_smooth'], label='RLAlloc')
        ax.plot(df_baseline['episode'], df_baseline['reward_smooth'], label='Baseline')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title('Performance Comparison')
        ax.grid(True)
        ax.legend()

    def _plot_resource_comparison(self, df_rl: pd.DataFrame, 
                                df_baseline: pd.DataFrame, ax: plt.Axes):
        resources = ['gpu_util_mean', 'cpu_util_mean', 'memory_util_mean']
        
        data = []
        for resource in resources:
            data.extend([
                {'Algorithm': 'RLAlloc', 'Resource': resource, 
                 'Utilization': df_rl[resource].mean()},
                {'Algorithm': 'Baseline', 'Resource': resource, 
                 'Utilization': df_baseline[resource].mean()}
            ])
            
        df_plot = pd.DataFrame(data)
        sns.barplot(data=df_plot, x='Resource', y='Utilization', 
                   hue='Algorithm', ax=ax)
        ax.set_title('Average Resource Utilization')