import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym

env = gym.make('CartPole-v1')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n


class CartPoleNet(nn.Module):
    def __init__(self, num_states, num_actions):
        super(CartPoleNet, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def choose_action(state):
    with torch.no_grad():
        x = torch.tensor(state).unsqueeze(0)
        action = model(x).argmax().item()
    return action



if __name__ == '__main__':
    model = CartPoleNet(num_states, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    state = env.reset()[0]
    epochs = 500
    for epoch in range(epochs):


        for times in range(500):
            action = choose_action(state)

            next_state, reward, done, info, _ = env.step(action)

            if done:
                reward = -10
            # target =