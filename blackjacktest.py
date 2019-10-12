import gym
import numpy as np
env = gym.make('Blackjack-v0')
env.reset()
print(env.observation_space)
print(env.action_space)
