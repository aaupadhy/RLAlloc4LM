import pytest
import torch
import numpy as np
from rlalloc.agents.sac import SAC, ReplayBuffer

@pytest.fixture
def config():
    return {
        'training': {
            'gamma': 0.99,
            'tau': 0.005,
            'learning_rate': 0.0003,
            'batch_size': 4,
            'buffer_size': 100
        }
    }

@pytest.fixture
def sac(config):
    state_dim = (5, 20, 1)
    action_dim = 3
    return SAC(state_dim, action_dim, config)

def test_replay_buffer():
    buffer = ReplayBuffer(5)
    state = torch.randn(5, 20, 1)
    action = np.array([0.5, 0.5, 0.5])
    next_state = torch.randn(5, 20, 1)
    
    buffer.push(state, action, 1.0, next_state, False)
    assert len(buffer) == 1

def test_sac_select_action(sac):
    state = torch.randn(5, 20, 1)  # Changed from np.random to torch.randn
    action = sac.select_action(state)
    assert action.shape == (3,)
    assert np.all(action >= -1) and np.all(action <= 1)

def test_sac_update(sac):
    for _ in range(10):
        state = torch.randn(5, 20, 1)
        action = np.random.randn(3)
        next_state = torch.randn(5, 20, 1)
        sac.replay_buffer.push(state, action, 1.0, next_state, False)
    
    sac.update()