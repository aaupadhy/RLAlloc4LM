import torch
import yaml
from pathlib import Path
import json
import os
import numpy as np
from env.resource_env import ResourceEnv
from rlalloc.agents.sac import SAC
from rlalloc.experts.baseline_heuristics import ExpertPolicy
from rlalloc.utils.preprocessing import TraceProcessor
from rlalloc.utils.metrics import MetricsLogger

def setup_directories():
    dirs = ['results', 'data/logs']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def load_config():
    with open('experiments/configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train():
    # Setup
    setup_directories()
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment
    env = ResourceEnv(config['environment'])
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    
    # Initialize expert and data processor
    trace_processor = TraceProcessor()
    trace_processor.load_trace('data/logs/trace_latest.json')
    expert = ExpertPolicy(config)
    expert.load_trace('data/logs/trace_latest.json')

    # Initialize agent and logger
    agent = SAC(state_dim, action_dim, config)
    metrics_logger = MetricsLogger(config)
    
    total_steps = 0
    for episode in range(config['training']['total_episodes']):
        state = env.reset()
        metrics_logger.start_episode()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Select action using epsilon-greedy strategy with expert
            if np.random.random() < config['expert']['epsilon']:
                action = expert.get_demonstration(state)
            else:
                action = agent.select_action(state)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update if enough samples
            if len(agent.replay_buffer) > config['training']['batch_size']:
                agent.update()
            
            # Logging
            metrics_logger.log_step(state, action, reward, next_state, info)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            steps += 1
            total_steps += 1
        
        # End of episode logging
        metrics_logger.end_episode(episode, episode_reward, steps)
        
        # Progress logging
        if episode % config['training']['log_interval'] == 0:
            print(f"Episode {episode}/{config['training']['total_episodes']}: "
                  f"Reward={episode_reward:.2f}, Steps={steps}, "
                  f"Total Steps={total_steps}")
        
        # Save checkpoints
        if episode % config['training']['save_interval'] == 0:
            save_path = Path('results') / f'model_ep{episode}.pt'
            agent.save(save_path)
            metrics_logger.save(f'results/metrics_ep{episode}.json')
    
    # Save final metrics
    metrics_logger.save('results/metrics_final.json')
    return metrics_logger.metrics

if __name__ == '__main__':
    try:
        metrics = train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final metrics...")
        metrics_logger.save('results/metrics_interrupted.json')
    except Exception as e:
        print(f"Error during training: {e}")
        raise