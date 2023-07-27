import gym
import torch
from torch import nn
from torch import optim
import numpy as np


def get_q_function(obs_size: int, n_act: int):
    return nn.Sequential(
        torch.nn.Linear(obs_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, n_act)
    )


class CartPoleAgent(object):

    def __init__(self, q_function: nn.Module,
                 lr: float = 1e-3,
                 gamma: float = .9,
                 delta: int = 100,
                 e_greedy: float = .1):
        self.q_function = q_function
        self.gamma = gamma
        self.optimizer = optim.AdamW(self.q_function.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.delta = delta
        self.e_greedy = e_greedy
        self.cur_episode = 0

    def act(self, state: torch.Tensor):
        if np.random.uniform() > self.e_greedy:
            with torch.no_grad():
                return self.q_function(state).argmax().item()
        return np.random.randint(0, 2)

    def train(self, n_episodes: int):
        for i in range(n_episodes):
            self.cur_episode += 1
            env = gym.make("CartPole-v0")
            state = torch.FloatTensor(env.reset()[0])
            done = False
            action = self.act(state)
            total_reward = .0
            while not done:
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                next_state = torch.FloatTensor(next_state)
                next_action = self.act(next_state)
                with torch.no_grad():
                    target = reward + (1 - float(done)) * self.gamma * self.q_function(next_state).max()
                loss = self.criterion(self.q_function(state)[action], target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                state = next_state
                action = next_action
            env.close()
            print(f"{i = } {total_reward = }")
            if (i + 1) % self.delta == 0:
                self.test()

    def test(self):
        env = gym.make("CartPole-v0", render_mode="human")
        done = False
        state = torch.FloatTensor(env.reset()[0])
        action = self.act(state)
        while not done:
            state, reward, done, _, _ = env.step(action)
            state = torch.FloatTensor(state)
            next_action = self.act(state)
            action = next_action
        env.close()


if __name__ == '__main__':
    agent = CartPoleAgent(get_q_function(4, 2), delta=1000)
    agent.train(1000000)
