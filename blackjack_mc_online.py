from collections import defaultdict

import gym
import sys
import numpy as np
from collections import defaultdict

env = gym.make('Blackjack-v0')


def update_Q(env, episode, Q, alpha, gamma):
    '''
    Function to update the Q-Table after running 1 episode
    '''
    # 1 - Extract the information of the run episode
    states, actions, rewards = zip(*episode)

    # 2 - Apply the discount factor
    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

    # 3 - Apply the update function to every Q(s,a) <- Q(s,a) + alpha*[Gt - Q(s,a)]
    for i, s in enumerate(states):
        a = actions[i]
        old_Q = Q[s][a]
        Q[s][a] = old_Q + alpha * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_Q)
    return Q


def get_probs(Qs, eps, nA):
    '''
    Function that obtains the probabilites corresponding to e-greedy policy
    '''
    # 1 - Initial equal radom probability for every possible action
    policy_s = np.ones(nA) * eps / nA

    # 2 - Determine which is the current optimal action for that state
    best_a = np.argmax(Qs)

    # 3 - Update (increase) the probability for the optimal action
    policy_s[best_a] = 1 - eps + (eps / nA)

    return policy_s


def generate_episode_from_Q(env, Q, eps, nA):
    '''
    Function to generate a MC episode given the environment, the last Q-Table,
    the ratio of exploration and the total number of actions
    Returns: and episode as a 3-tuple of (states, actions, rewards)
    '''
    # Initialize an empty env to run the new episode
    episode = []
    state = env.reset()

    # Until terminal state
    while True:

        # Generate an action following the policy
        action = np.random.choice(np.arange(nA),
                                  p=get_probs(Q[state], eps, nA)) if state in Q else env.action_space.sample()

        # Perform the 3-tuple for that state - Every visit approach
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))

        # Advance one state
        state = next_state
        if done:
            break

    return episode


def mc_control(env, num_episodes, alpha, gamma=0.9, eps0=1.0, eps_decay=.99999, eps_min=0.05):
    # Variable to store all the possible actions of the environment
    nA = env.action_space.n

    # Initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))

    # Initialize epsilon
    eps = eps0

    # Loop over episodes
    for i_episode in range(1, num_episodes + 1):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Recalculate epsilon with a scheduler (a simple decay)
        eps = max(eps * eps_decay, eps_min)

        # Run the episode by following the eps-greedy policy
        episode = generate_episode_from_Q(env, Q, eps, nA)

        # Update the Q-Table values
        Q = update_Q(env, episode, Q, alpha, gamma)

    # Unroll our Q-Table picking the best action at each state (row) to define the found optimal policy
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    print('Q', Q);
    print('policy', policy)

    return policy, Q


if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    mc_control(env, num_episodes=3, alpha=0.1)
    # agent = BlackJackQAgent()
    # agent.train()
    # agent.run()
