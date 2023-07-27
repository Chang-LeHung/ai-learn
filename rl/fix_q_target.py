import gym
import torch
from torch import nn
from torch import optim
import numpy as np
from replay_buffer import ReplayBuffer  # type: ignore


def get_q_function(obs_size: int, n_act: int):
    return nn.Sequential(
        torch.nn.Linear(obs_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, n_act)
    )


class CartPoleAgent(object):

    def __init__(self, q_function: nn.Module,
                 lr: float = 1e-3,
                 gamma: float = .99,
                 delta: int = 100,
                 e_greedy: float = .1,
                 sync_step: int = 8,
                 replay_buffer_limit: int = 1000,
                 replay_buffer_size: int = 1000,
                 replay_interval: int = 4,
                 batch_size: int = 128
                 ):
        assert batch_size <= replay_buffer_size
        self.sync_step = sync_step
        self.env = gym.make("CartPole-v0")
        self.q_function = q_function
        self.t_function = get_q_function(4, 2)
        self.sync()
        self.gamma = gamma
        self.optimizer = optim.Adam(self.q_function.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.delta = delta
        self.e_greedy = e_greedy
        self.cur_episode = 0
        self.replay_buffer_limit = replay_buffer_limit
        self.replay_interval = replay_interval
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def act(self, state: torch.Tensor):
        if self.cur_episode > 1000 or np.random.uniform() > self.e_greedy:
            with torch.no_grad():
                return self.q_function(state).argmax().item()
        return np.random.randint(0, 2)

    def sync(self):
        for t, q in zip(self.t_function.parameters(), self.q_function.parameters()):
            t.data.copy_(q.data)

    def train_from_replay_buffer(self):
        def one_hot_encode(length, number):
            identity_matrix = torch.eye(length)
            one_hot = identity_matrix[number]
            return one_hot

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        rewards = torch.FloatTensor(rewards)
        states = torch.stack(states)
        actions = [one_hot_encode(2, idx) for idx in actions]
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).reshape(-1)
        predict = (self.q_function(states) * actions).sum(dim=-1)
        with torch.no_grad():
            target = rewards + self.gamma * (1 - dones) * self.t_function(next_states).max(dim=-1)[0]
        self.optimizer.zero_grad()
        loss = self.criterion(predict, target)
        loss.backward()
        self.optimizer.step()

    def train(self, n_episodes: int):
        for i in range(n_episodes):
            self.cur_episode += 1
            state = torch.FloatTensor(self.env.reset()[0])
            done = False
            action = self.act(state)
            total_reward = .0
            step = 1
            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                next_state = torch.FloatTensor(next_state)
                next_action = self.act(next_state)
                with torch.no_grad():
                    target = reward + (1 - float(done)) * self.gamma * self.t_function(next_state).max()
                loss = self.criterion(self.q_function(state)[action], target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                state = next_state
                action = next_action
                step += 1
                self.replay_buffer.push(state, action, reward, next_state, done)
                if step % self.sync_step == 0:
                    self.sync()
                if step % self.replay_interval == 0 and len(self.replay_buffer) >= self.batch_size:
                    self.train_from_replay_buffer()

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
    agent = CartPoleAgent(get_q_function(4, 2), delta=200)
    agent.train(10000)
