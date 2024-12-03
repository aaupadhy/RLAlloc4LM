import torch
import yaml
import json
import os
import random
import numpy as np
from tqdm import tqdm
from env.resource_env import ResourceEnv
from rlalloc.agents.sac import SAC
from rlalloc.experts.baseline_heuristics import ExpertPolicy
from rlalloc.utils.metrics import MetricsLogger

def setup_paths(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def train():
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = yaml.safe_load(open('experiments/configs/config.yaml'))
    setup_paths(['results', 'data/logs'])
    
    env = ResourceEnv(config['environment']).to(device)
    env.action_space.seed(42)
    expert = ExpertPolicy(config).to(device)
    expert.load_trace('data/logs/trace_latest.json')
    
    agent = SAC(env.observation_space.shape, env.action_space.shape[0], config).to(device)
    metrics_logger = MetricsLogger(config)
    
    pbar = tqdm(range(config['training']['total_episodes']))
    episode_rewards = []
    running_reward_mean = 0
    running_reward_std = 1
    
    for episode in pbar:
        state = env.reset()
        metrics_logger.start_episode()
        episode_reward = 0
        episode_q = 0
        updates = 0

        while True:
            eps = max(0.1, min(0.99, 1.0 - episode / 100))
            if np.random.random() < eps:
                with torch.no_grad():
                    action = torch.tensor(expert.get_demonstration(state), device=device)
            else:
                action = torch.tensor(agent.select_action(state), device=device)

            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > agent.batch_size:
                actual_batch = min(agent.batch_size * 2, len(agent.replay_buffer))
                update_info = agent.update(actual_batch)
                if update_info and not np.isnan(update_info['q_value']):
                    episode_q += update_info['q_value']
                    updates += 1

            episode_reward += reward
            state = next_state
            
            if done:
                break

        episode_rewards.append(episode_reward)

        if updates > 0:
                avg_q = episode_q / updates
        else:
            avg_q = 0
        utilization_data = env.get_utilization() 
        metrics_logger.end_episode(episode, 
                                total_reward=episode_reward,
                                avg_q_value=avg_q,
                                gpu_util_mean=utilization_data['gpu_util_mean'],
                                cpu_util_mean=utilization_data['cpu_util_mean'], 
                                memory_util_mean=utilization_data['memory_util_mean'])
        
        pbar.set_description(f"Episode {episode} | Reward: {episode_reward:.2f} | Q-value: {avg_q:.2f} | Expert: {eps:.2f}")

        if episode % config['training']['save_interval'] == 0:
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'metrics': metrics_logger.get_metrics()
            }, f'results/checkpoint_ep{episode}.pt')
    
    final_metrics = metrics_logger.get_metrics()
    with open('results/metrics_final.json', 'w') as f:
        json.dump(final_metrics, f)


if __name__ == '__main__':
    train()