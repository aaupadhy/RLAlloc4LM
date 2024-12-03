import torch
import torch.nn as nn
import torch.nn.functional as F

class ResourceCNN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_value = nn.Linear(256, 1)
        self.fc_policy = nn.Linear(256, action_dim)

    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        x = x.view(batch_size, 3, 10, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.fc_value(x)
        policy = torch.sigmoid(self.fc_policy(x))
        
        return policy, value