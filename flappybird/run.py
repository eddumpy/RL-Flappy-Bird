import time
import gym
import gym_ple
import os, sys
import logging
import numpy as np
from Tiling import Tiling

from ple import PLE
from gym.wrappers import Monitor
from ple.games.flappybird import FlappyBird

class Agent():
    def __init__(self, environment, alpha=0.05, epsilon=0, gamma=0.99, lambda_=0.9):
        '''Initializes parameter values'''
        self.env = environment
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambda_ = lambda_

    def random_action(self):
        '''Returns random action from available actions'''
        available_actions = self.env.getActionSet()
        action = np.random.choice(available_actions)
        return action

    def choose_action(self):
        '''Chooses action following e-greedy policy'''
        return self.random_action()

    def learn(self):
        pass

# Overriding method
def process_state(state):
    '''Processes state values into state representation'''
    # Input values
    player_y = state["player_y"]
    player_vel = state["player_vel"]
    pipe_1_x_diff = state["next_pipe_dist_to_player"]
    pipe_1_top_y = state["next_pipe_top_y"]
    pipe_1_bottom_y = state["next_pipe_bottom_y"]
    pipe_2_x_diff = state["next_next_pipe_dist_to_player"]
    pipe_2_top_y = state["next_next_pipe_top_y"]
    pipe_2_bottom_y = state["next_next_pipe_bottom_y"]

    # Flip calculations
    player_vel *= -1
    pipe_gap = pipe_1_bottom_y - pipe_1_top_y
    pipe_1_y_diff = player_y - pipe_1_bottom_y
    pipe_2_y_diff = player_y - pipe_2_bottom_y

    # Process state
    state_representation = np.array([pipe_1_x_diff, player_vel, pipe_1_y_diff])
    return state_representation

def getQ(F, theta):
    '''Returns total theta (Q) value for all on features'''
    Q = 0
    for i in F:
        Q += theta[i]
    return Q

def play(episodes=100):
    '''Plays given episodes using Sarsa-Lambda to learn'''
    # Initialize game and agent
    game = FlappyBird()
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    p.init()
    agent = Agent(p)
    total_reward = []

    # Initialize tiles with fixed values
    t = Tiling(0, 300, -10, 16, -300, 300)

    # Load theta from file
    # theta = np.load('theta.npy')
    theta = np.zeros(t.total_tiles)

    # Run given number of episodes
    for x in range(episodes):
        # Initialize episode parameters
        p.reset_game()
        e = np.zeros(t.total_tiles)

        state = p.getGameState()
        action = agent.choose_action()
        total_episode_reward = 0

        # Episode loop
        while not p.game_over():
            # Get features that are 'on'
            F = t.get_indices(state, action)

            for i in F:
                e[i] = 1 # Replacing traces

            # Take action and observe reward and next state
            reward = p.act(action)
            state = p.getGameState()
            delta = reward - getQ(F, theta)

            actions = p.getActionSet()
            chance = np.random.uniform(0,1)
            # Perform action under epsilon-greedy policy
            if chance < (1 - agent.epsilon):
                Qs = []
                for a in actions:
                    F = t.get_indices(state, a)
                    Qa = getQ(F, theta)
                    Qs.append(Qa)
                # Choose maximum action
                action = actions[np.argmax(Qs)]
                F = t.get_indices(state, action)
                Qa = getQ(F, theta)

            else:
                # Else take random action
                action = np.random.choice(actions)
                F = t.get_indices(state, action)
                Qa = getQ(F, theta)

            # Make step updates
            delta += agent.gamma * Qa
            theta += agent.alpha * delta * e
            e *= agent.gamma * agent.lambda_
            total_episode_reward += reward
            # Cap reward achieved
            if total_episode_reward >= 300:
                print("Capped")
                break
            #print(state)

        total_reward.append(total_episode_reward)
        print("Episode: ", x, " Reward: ", total_episode_reward)

    # Save updated theta to file
    # np.save('theta', theta)
    return total_reward

def random_play(episodes = 100):
    # Initialize game and agent
    game = FlappyBird()
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    p.init()
    agent = Agent(p)
    total_reward = []

    # Run given number of episodes
    for _ in range(episodes):
        # Initialize episode
        p.reset_game()
        total_episode_reward = 0

        # Episode loop
        while not p.game_over():
            action = agent.choose_action()
            reward = p.act(action)
            total_episode_reward += reward

        # Save episode reward and return
        total_reward.append(total_episode_reward)

    return total_reward
