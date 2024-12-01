import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from ..models.cnn import ResourceCNN
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def push(self, state, action, reward, next_state, done):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
            
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = list(zip(*transitions))
        
        states = torch.stack(batch[0]).to(self.device)
        actions = torch.stack(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.stack(batch[3]).to(self.device)
        dones = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class SAC:
    def __init__(self, state_dim, action_dim, config):
        self.gamma = config['training']['gamma']
        self.tau = config['training']['tau']
        self.alpha = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.critic1 = ResourceCNN(action_dim).to(self.device)
        self.critic2 = ResourceCNN(action_dim).to(self.device)
        self.critic1_target = ResourceCNN(action_dim).to(self.device)
        self.critic2_target = ResourceCNN(action_dim).to(self.device)
        self.actor = ResourceCNN(action_dim).to(self.device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['training']['learning_rate'])
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config['training']['learning_rate'])
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config['training']['learning_rate'])
        
        self.replay_buffer = ReplayBuffer(config['training']['buffer_size'])
        self.batch_size = config['training']['batch_size']
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor(state)
            action = action.cpu().numpy()[0]
        return np.clip(action, -1, 1)

    def update(self):
            if len(self.replay_buffer) < self.batch_size:
                return
                
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            batch_size = state.size(0)

            with torch.no_grad():
                next_action, next_value = self.actor(next_state)
                target_q1, _ = self.critic1_target(next_state)
                target_q2, _ = self.critic2_target(next_state)
                target_q = torch.min(torch.cat([target_q1, target_q2], dim=1), dim=1)[0].unsqueeze(1)
                target_q = reward + (1 - done) * self.gamma * target_q

            current_q1, _ = self.critic1(state)
            current_q2, _ = self.critic2(state)
            
            current_q1 = current_q1.view(batch_size, -1)[:, 0].unsqueeze(1)
            current_q2 = current_q2.view(batch_size, -1)[:, 0].unsqueeze(1)
            
            critic1_loss = F.mse_loss(current_q1, target_q.detach())
            critic2_loss = F.mse_loss(current_q2, target_q.detach())

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            action_pred, log_pi = self.actor(state)
            q1_pred, _ = self.critic1(state)
            q2_pred, _ = self.critic2(state)
            
            q1_pred = q1_pred.view(batch_size, -1)[:, 0]
            q2_pred = q2_pred.view(batch_size, -1)[:, 0]
            q_pred = torch.min(q1_pred, q2_pred).unsqueeze(1)
            log_pi = log_pi.view(batch_size, -1)[:, 0].unsqueeze(1)
            
            actor_loss = (self.alpha * log_pi - q_pred).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)