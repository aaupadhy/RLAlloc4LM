import pytest
import numpy as np
from env.resource_env import ResourceEnv

@pytest.fixture
def env():
    config = {
        'max_gpu_memory': 40 * 1024 * 1024 * 1024,
        'max_cpu_memory': 256 * 1024 * 1024 * 1024,
        'max_cpu_cores': 32,
        'time_horizon': 20
    }
    return ResourceEnv(config)

def test_env_init(env):
    assert env.observation_space.shape == (5, 20, 1)
    assert env.action_space.shape == (3,)
    assert env.current_step == 0

def test_env_reset(env):
    state = env.reset()
    assert state.shape == (5, 20, 1)
    assert env.current_step == 0
    assert len(env.history) == 1

def test_env_step(env):
    env.reset()
    action = np.array([0.5, 0.5, 0.5])
    state, reward, done, _ = env.step(action)
    
    assert state.shape == (5, 20, 1)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert env.current_step == 1