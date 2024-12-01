# rlalloc/models/transformer.py
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
   def __init__(self, d_model, num_heads):
       super().__init__()
       self.num_heads = num_heads
       self.d_k = d_model // num_heads
       
       self.q = nn.Linear(d_model, d_model)
       self.k = nn.Linear(d_model, d_model)
       self.v = nn.Linear(d_model, d_model)
       self.out = nn.Linear(d_model, d_model)
       
   def forward(self, x):
       batch_size = x.size(0)
       
       q = self.q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
       k = self.k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
       v = self.v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
       
       scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
       attn = torch.softmax(scores, dim=-1)
       
       context = torch.matmul(attn, v)
       context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
       
       return self.out(context)

class TransformerBlock(nn.Module):
   def __init__(self, d_model, num_heads, d_ff):
       super().__init__()
       self.attn = MultiHeadAttention(d_model, num_heads)
       self.ff = nn.Sequential(
           nn.Linear(d_model, d_ff),
           nn.ReLU(),
           nn.Linear(d_ff, d_model)
       )
       self.ln1 = nn.LayerNorm(d_model)
       self.ln2 = nn.LayerNorm(d_model)
       
   def forward(self, x):
       attn_out = self.ln1(x + self.attn(x))
       return self.ln2(attn_out + self.ff(attn_out))

class PolicyTransformer(nn.Module):
   def __init__(self, grid_size=32, num_actions=3):
       super().__init__()
       self.d_model = 256
       
       self.embed = nn.Sequential(
           nn.Conv2d(4, 64, 3, stride=2, padding=1),
           nn.ReLU(),
           nn.Conv2d(64, 128, 3, stride=2, padding=1),
           nn.ReLU(),
           nn.Conv2d(128, self.d_model, 3, stride=2, padding=1)
       )
       
       self.transformer = nn.Sequential(
           TransformerBlock(self.d_model, 8, 1024),
           TransformerBlock(self.d_model, 8, 1024),
           TransformerBlock(self.d_model, 8, 1024)
       )
       
       self.policy_head = nn.Sequential(
           nn.Linear(self.d_model, 512),
           nn.ReLU(),
           nn.Linear(512, num_actions),
           nn.Sigmoid()
       )
       
   def forward(self, x):
       x = self.embed(x)
       x = x.flatten(2).transpose(1, 2)
       x = self.transformer(x)
       x = x.mean(dim=1)
       return self.policy_head(x)

class ValueTransformer(nn.Module):
   def __init__(self, grid_size=32):
       super().__init__()
       self.d_model = 256
       
       self.embed = nn.Sequential(
           nn.Conv2d(4, 64, 3, stride=2, padding=1),
           nn.ReLU(),
           nn.Conv2d(64, 128, 3, stride=2, padding=1),
           nn.ReLU(),
           nn.Conv2d(128, self.d_model, 3, stride=2, padding=1)
       )
       
       self.transformer = nn.Sequential(
           TransformerBlock(self.d_model, 8, 1024),
           TransformerBlock(self.d_model, 8, 1024),
           TransformerBlock(self.d_model, 8, 1024)
       )
       
       self.value_head = nn.Sequential(
           nn.Linear(self.d_model, 512),
           nn.ReLU(),
           nn.Linear(512, 1)
       )
       
   def forward(self, x):
       x = self.embed(x)
       x = x.flatten(2).transpose(1, 2)
       x = self.transformer(x)
       x = x.mean(dim=1)
       return self.value_head(x)