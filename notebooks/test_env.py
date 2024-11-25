import gym
from environments.env import LLMResourceEnv


env = LLMResourceEnv(num_resources=4, job_queue_size=5)
state = env.reset()
print("Initial State:", state)

done = False
while not done:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    env.render()
