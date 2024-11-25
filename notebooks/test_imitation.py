import torch
import numpy as np
from envs.llm_env import LLMResourceEnv
from src.imitation_learning import PolicyNetwork

# Load environment and trained model
env = LLMResourceEnv(num_resources=4, job_queue_size=10)
model = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
model.load_state_dict(torch.load("models/imitation_model.pth"))
model.eval()

# Test the imitation model
state = env.reset()
done = False
total_reward = 0

print("Testing imitation model...")
while not done:
    # Convert state to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_probs = model(state_tensor).detach().numpy().squeeze()
    action = np.argmax(action_probs)

    # Take action in the environment
    state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

print(f"Total Reward: {total_reward}")
