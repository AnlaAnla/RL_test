import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("Ant-v2", render_mode="human")
observation, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(i, reward)
        # observation, info = env.reset()
env.close()
