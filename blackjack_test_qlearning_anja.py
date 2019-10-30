from collections import defaultdict

import gym
import numpy as np
EPSILON = 0.2
ALPHA = 0.6


class BlackJackQAgent():
    def __init__(self, num_episodes=1, min_explore=0.1, discount=1.0, decay=25):

        self.num_episodes = num_episodes
        self.min_explore = min_explore
        self.discount = discount
        self.decay = decay
        self.env = gym.make('Blackjack-v0')
        # Variable to store all the possible actions of the environment
        nA = self.env.action_space.n
        nS = self.env.observation_space
        print('na', nA)
        print('ns', nS)
        #self.Q_table = np.zeros((self.env.action_space.n,));
        self.Q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        #self.Q_table = defaultdict(lambda: self.env.action_space.n)

    def choose_action(self, state, epsilon = EPSILON):
        if np.random.random() < epsilon:
            # randomly sample explore rate percent of the time
            return self.env.action_space.sample()
        else:
            # take optimal action
            return np.argmax(self.Q_table[state])

    def update_q(self, state, action, reward, new_state, alpha = ALPHA):
        self.Q_table[state][action] += alpha * (reward + self.discount * np.max((self.Q_table[new_state])- self.Q_table[state][action]))

    def train(self):
        for e in range(self.num_episodes):
            current_state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = obs
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state

        print('finished training')

    def run(self):
        self.env = gym.make('Blackjack-v0')

        while True:
            current_state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
               # print('state', obs)
                new_state = obs
                current_state = new_state


if __name__ == "__main__":
    agent = BlackJackQAgent()
    agent.train()
    agent.run()
