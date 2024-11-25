import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from environments.env import LLMResourceEnv

class JobDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

def generate_expert_data(env, num_episodes=500):
    states = []
    actions = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Expert uses Shortest Job First strategy
            job_lengths = [job["execution_time"] for job in env.job_queue]
            action = np.argmin(job_lengths) if len(job_lengths) > 0 else env.action_space.n - 1
            states.append(state)
            actions.append(action)

            state, _, done, _ = env.step(action)

    return np.array(states), np.array(actions)

def train_imitation_model(states, actions, input_dim, output_dim, epochs=10, batch_size=32, lr=0.001):
    dataset = JobDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for state_batch, action_batch in dataloader:
            state_batch = torch.FloatTensor(state_batch)
            action_batch = torch.LongTensor(action_batch)

            optimizer.zero_grad()
            predictions = model(state_batch)
            loss = loss_fn(predictions, action_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    return model

if __name__ == "__main__":
    env = LLMResourceEnv(num_resources=4, job_queue_size=10)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("Generating expert data...")
    states, actions = generate_expert_data(env)

    print("Training imitation model...")
    model = train_imitation_model(states, actions, state_dim, action_dim)

    torch.save(model.state_dict(), "models/imitation_model.pth")
    print("Imitation model saved!")
