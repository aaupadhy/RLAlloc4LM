import yaml
import torch
import numpy as np
from env.resource_env import ResourceEnv
from rlalloc.agents.sac import SAC
from rlalloc.experts.baseline_heuristics import ExpertPolicy
from rlalloc.utils.preprocessing import TraceProcessor

def main():
    # Load config
    with open('experiments/configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    env = ResourceEnv(config['environment'])
    
    trace_processor = TraceProcessor()
    trace_processor.load_trace('data/logs/g065.cluster_137762.1732958297754317367.pt.trace.json')
    
    expert = ExpertPolicy(config)
    expert.load_trace('data/logs/g065.cluster_137762.1732958297754317367.pt.trace.json')
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    agent = SAC(state_dim, action_dim, config)
    
    rewards = agent.train(env, expert, config['training']['total_timesteps'])
    
    agent.save('results/model_final.pt')
    
    np.save('results/rewards.npy', rewards)

if __name__ == "__main__":
    main()