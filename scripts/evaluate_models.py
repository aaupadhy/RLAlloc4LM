import torch
import yaml
import numpy as np
import json
from env.resource_env import ResourceEnv
from rlalloc.agents.sac import SAC
from tqdm import tqdm
from rlalloc.experts.baseline_heuristics import ExpertPolicy

def train_sac_only(env, agent, episodes=1000):
    for episode in tqdm(range(episodes), desc="Training SAC"):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
            if done:
                break
            state = next_state

def evaluate_models(config_path='experiments/configs/config.yaml'):
    config = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = ResourceEnv(config['environment']).to(device)
    
    # Train and evaluate SAC without IL
    print("Training SAC without IL...")
    sac = SAC(env.observation_space.shape, env.action_space.shape[0], config).to(device)
    train_sac_only(env, sac)
    sac_rewards = []
    
    for episode in tqdm(range(1000)):  # Train episodes
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = sac.select_action(state)
            next_state, reward, done, _ = env.step(action)
            sac.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(sac.replay_buffer) > sac.batch_size:
                sac.update()
                
            state = next_state
            episode_reward += reward
            if done:
                break
                
        if episode % 10 == 0:  # Evaluation every 10 episodes
            eval_reward = evaluate_policy(env, sac)
            sac_rewards.append(eval_reward)
    
    # Evaluate baseline
    print("Evaluating baseline...")
    expert = ExpertPolicy(config)
    baseline_rewards = []
    
    for _ in tqdm(range(100)):
        baseline_rewards.append(evaluate_policy(env, expert))
    
    with open('results/sac_rewards.json', 'w') as f:
        json.dump(sac_rewards, f)
    with open('results/baseline_rewards.json', 'w') as f:
        json.dump(baseline_rewards, f)
        
def evaluate_policy(env, policy, episodes=5):
    rewards = []
    for _ in range(episodes):
        state = env.reset()  # Reset for each episode
        total_reward = 0
        done = False
        
        while not done:
            if hasattr(policy, 'select_action'):
                action = policy.select_action(state, evaluate=True)  # Added evaluate flag
            else:
                action = policy.get_demonstration(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            
        rewards.append(total_reward)
        
    return np.mean(rewards)

if __name__ == '__main__':
    evaluate_models()