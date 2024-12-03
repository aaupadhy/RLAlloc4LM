import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from rlalloc.models.networks import Actor, Critic
import random

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def push(self, state, action, reward, next_state, done):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        # Higher priority for expert demonstrations and high rewards
        priority = 5.0 if reward > 50.0 else (3.0 if reward > 20.0 else 1.0)
        self.priorities.append(priority)
        self.buffer.append((state, action, reward, next_state, done, priority))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None, None, None
            
        expert_indices = [i for i, x in enumerate(self.buffer) if x[5] > 1.0]
        if expert_indices:
            n_expert = min(len(expert_indices), batch_size // 4)
            expert_batch = np.random.choice(expert_indices, n_expert)
            regular_indices = [i for i in range(len(self.buffer)) if i not in expert_indices]
            regular_batch = np.random.choice(regular_indices, batch_size - n_expert, p=np.array(self.priorities)[regular_indices]/sum(np.array(self.priorities)[regular_indices]))
            indices = np.concatenate([expert_batch, regular_batch])
        else:
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        batch = [self.buffer[i] for i in indices]
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch]) 
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        return (torch.from_numpy(states).float().to(self.device),
                torch.from_numpy(actions).float().to(self.device),
                torch.from_numpy(rewards).float().to(self.device), 
                torch.from_numpy(next_states).float().to(self.device),
                torch.from_numpy(dones).float().to(self.device))
    
    def update_normalizer(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        self.count += 1
        delta = state - self.running_mean
        self.running_mean += delta / self.count
        self.running_var = (self.running_var * (self.count - 1) + delta * (state - self.running_mean)) / self.count
        
    def normalize_state(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        return (state - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
    
    def __len__(self):
        return len(self.buffer)
    


class SAC:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config['training'].get('gamma', 0.99)
        self.tau = config['training'].get('tau', 0.005)
        self.batch_size = config['training'].get('batch_size', 256)
        
        self.actor = Actor(state_dim[0], action_dim).to(self.device)
        self.critic_1 = Critic(state_dim[0], action_dim).to(self.device)
        self.critic_2 = Critic(state_dim[0], action_dim).to(self.device)
        self.critic_target_1 = Critic(state_dim[0], action_dim).to(self.device)
        self.critic_target_2 = Critic(state_dim[0], action_dim).to(self.device)
        self.total_it = 0
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        lr = config['training'].get('learning_rate', 3e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=lr
        )
        
        self.replay_buffer = ReplayBuffer(
            config['training'].get('buffer_size', 1000000),
            state_dim[0]
        )
        

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_target_1': self.critic_target_1.state_dict(),
            'critic_target_2': self.critic_target_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic_1.load_state_dict(state_dict['critic_1'])
        self.critic_2.load_state_dict(state_dict['critic_2'])
        self.critic_target_1.load_state_dict(state_dict['critic_target_1'])
        self.critic_target_2.load_state_dict(state_dict['critic_target_2'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        
        
    def select_action(self, state, evaluate=False):
        if isinstance(state, torch.Tensor):
            state = state.float()
        else:
            state = torch.FloatTensor(state)
        state = state.to(self.device)
        
        action = self.actor(state.unsqueeze(0))
        if not evaluate:
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, 0, 1)
        return action.detach().cpu().numpy()[0]

    def update(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        # Train critics more aggressively on high-reward experiences
        for _ in range(3):  # Multiple critic updates 
            current_q1 = self.critic_1(state, action)
            current_q2 = self.critic_2(state, action)
            
            with torch.no_grad():
                next_action = self.actor(next_state)
                next_q1 = self.critic_target_1(next_state, next_action)
                next_q2 = self.critic_target_2(next_state, next_action)
                next_q = torch.min(next_q1, next_q2)
                target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * next_q

            # Weight critic loss by reward magnitude
            weights = (reward - reward.min()) / (reward.max() - reward.min() + 1e-6)
            weights = weights.unsqueeze(1)
            critic_loss = (weights * (F.mse_loss(current_q1, target_q, reduction='none') + 
                                    F.mse_loss(current_q2, target_q, reduction='none'))).mean()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Actor update remains same
        actions_pred = self.actor(state)
        actor_loss = -self.critic_1(state, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_target()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), 'q_value': current_q1.mean().item()}

    def cuda(self):
        self.actor = self.actor.cuda()
        self.critic_1 = self.critic_1.cuda()
        self.critic_2 = self.critic_2.cuda()
        self.critic_target_1 = self.critic_target_1.cuda()
        self.critic_target_2 = self.critic_target_2.cuda()
        return self

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic_1 = self.critic_1.to(device)
        self.critic_2 = self.critic_2.to(device)
        self.critic_target_1 = self.critic_target_1.to(device)
        self.critic_target_2 = self.critic_target_2.to(device)
        return self


    def _soft_update_target(self):
        for target, source in [(self.critic_target_1, self.critic_1),
                             (self.critic_target_2, self.critic_2)]:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )