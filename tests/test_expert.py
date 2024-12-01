import pytest
import torch
import numpy as np
from rlalloc.experts.baseline_heuristics import ExpertPolicy

@pytest.fixture
def config():
    return {
        'expert': {
            'type': 'sjf'
        }
    }

@pytest.fixture
def expert(config):
    return ExpertPolicy(config)

def test_expert_sjf(expert):
    state = torch.randn(5, 20, 1)
    action = expert.shortest_job_first(state)
    assert action.shape == (3,)
    assert np.all(action >= 0) and np.all(action <= 1)

def test_expert_utilization(expert):
    state = torch.randn(5, 20, 1)
    action = expert.highest_utilization(state)
    assert action.shape == (3,)
    assert np.all(action >= 0) and np.all(action <= 1)

def test_expert_demonstration(expert):
    state = torch.randn(5, 20, 1)
    action = expert.get_demonstration(state)
    assert action.shape == (3,)
    assert np.all(action >= 0) and np.all(action <= 1)