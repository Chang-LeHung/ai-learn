import gym
from tqdm import tqdm
import numpy as np


class SarsaAgent(object):

    def __init__(self, n_state, n_action, lr=.005, gamma=.9, e_greedy=.1):
        self.n_state = n_state
        self.n_action = n_action
        self.lr = lr
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.q_table = np.zeros((n_state, n_action))

    def next_action(self, state):
        if np.random.uniform() < self.e_greedy:
            return np.random.choice(self.n_action)
        else:
            # if same val in all actions, then choose randomly
            return np.random.choice(np.flatnonzero(self.q_table[state] == np.max(self.q_table[state])))

    def step(self, state, action, reward, next_state, next_action, done):

        # update q_table
        # alpha means learning rate
        # s' means next state
        # a' means next action
        # Q(s, a) <- Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
        if done:
            self.q_table[state, action] += self.lr * (
                    reward - self.q_table[state, action])
        else:
            self.q_table[state, action] += self.lr * (
                    reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action])


if __name__ == '__main__':
    env = gym.make('CliffWalking-v0', render_mode="ansi")

    episodes = 2000000

    agent = SarsaAgent(env.observation_space.n, env.action_space.n)  # type: ignore
    # for i in range(episodes):
    i = 0
    for i in tqdm(range(episodes), desc="CliffWalking"):
        state = env.reset()[0]
        action = agent.next_action(state)
        while True:
            # next_state is observation
            next_state, reward, done, info, _ = env.step(action)
            next_action = agent.next_action(next_state)
            agent.step(state, action, reward, next_state, next_action, done)

            # update state , action
            state = next_state
            action = next_action

            if done:
                state = env.reset()
                break
        if (i + 1) % 20000 == 0:
            env.close()
            env = gym.make('CliffWalking-v0', render_mode="human")
        else:
            env.close()
            env = gym.make('CliffWalking-v0')
    env.close()
