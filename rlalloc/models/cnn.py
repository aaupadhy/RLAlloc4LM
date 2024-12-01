import torch
import torch.nn as nn
import torch.nn.functional as F

class ResourceCNN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        
        # Input shape: [batch, 5, 20, 1]
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 20, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_value = nn.Linear(256, 1)
        self.fc_policy = nn.Linear(256, action_dim)

    def forward(self, x):
        # Input comes as [channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Ensure correct shape [batch, channels, height, width]
        if x.shape[-1] == 1:
            x = x.permute(0, 1, 2, 3)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.fc_value(x)
        policy = torch.tanh(self.fc_policy(x))
        
        log_prob = F.log_softmax(policy, dim=-1)
        return policy, value