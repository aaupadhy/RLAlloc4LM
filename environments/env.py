import gym
from gym import spaces
import numpy as np
import random

class LLMResourceEnv(gym.Env):
    def __init__(self, num_resources=4, job_queue_size=10):
        super(LLMResourceEnv, self).__init__()
        self.num_resources = num_resources
        self.job_queue_size = job_queue_size

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(job_queue_size + 1, 2 + num_resources), dtype=np.float32
        )


        self.action_space = spaces.Discrete(job_queue_size + 1)
        
        self.reset()

    def reset(self):
        self.resources = np.ones(self.num_resources, dtype=np.float32)

        self.job_queue = [
            {
                "arrival_time": random.uniform(0, 1),
                "execution_time": random.uniform(0.1, 1),
                "resource_demand": np.random.uniform(0.1, 1, self.num_resources)
            }
            for _ in range(self.job_queue_size)
        ]

        self.job_queue = sorted(self.job_queue, key=lambda x: x["arrival_time"])

        self.time = 0 
        return self._get_state()

    def _get_state(self):
        # Define maximum job queue size
        max_jobs = self.job_queue_size

        # Initialize the state matrix
        state = np.zeros((max_jobs, 2 + self.num_resources), dtype=np.float32)

        # Fill in job features
        for i, job in enumerate(self.job_queue[:max_jobs]):
            state[i, 0] = job["arrival_time"]  # Arrival time
            state[i, 1] = job["execution_time"]  # Execution time
            state[i, 2:] = job["resource_demand"]  # Resource demands

        # Add resource availability as a separate row (if needed by the policy)
        resource_row = np.expand_dims(self.resources, axis=0)  # Shape (1, num_resources)
        resource_features = np.pad(resource_row, ((0, 0), (0, 2)), mode='constant')  # Pad to match columns

        # Append the resource row to the state matrix (optional)
        state = np.vstack([state, resource_features])

        return state


    def step(self, action):
        reward = 0
        done = False

        if action < len(self.job_queue):
            job = self.job_queue[action]
            if all(self.resources >= job["resource_demand"]):
                self.resources -= job["resource_demand"]
                reward += 1 / job["execution_time"]
                self.job_queue.pop(action)
            else:
                reward -= 1
        else:
            reward -= 0.1 

        self.resources = np.minimum(self.resources + 0.1, 1.0)

        done = len(self.job_queue) == 0

        return self._get_state(), reward, done, {}

    def render(self, mode="human"):
        print(f"Time: {self.time}, Resources: {self.resources}, Jobs: {len(self.job_queue)}")