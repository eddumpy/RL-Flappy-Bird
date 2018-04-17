import gym
import gym_ple
import os, sys
import logging
import numpy as np
import Tiling

from ple import PLE
from gym.wrappers import Monitor
from ple.games.flappybird import FlappyBird


# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

def visualization_test():
    if __name__ == '__main__':
        # You can optionally set up the logger. Also fine to set the level
        # to logging.DEBUG or logging.WARN if you want to change the
        # amount of output.
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        env = gym.make('FlappyBird-v0' if len(sys.argv)<2 else sys.argv[1])

        # You provide the directory to write to (can be an existing
        # directory, including one with existing data -- all monitor files
        # will be namespaced). You can also dump to a tempdir if you'd
        # like: tempfile.mkdtemp().
        outdir = '/tmp/random-agent-results'

        #env = Monitor(env, directory=outdir, force=True)
        env.render()

        # This declaration must go *after* the monitor call, since the
        # monitor's seeding creates a new action_space instance with the
        # appropriate pseudorandom number generator.
        env.seed(0)
        agent = RandomAgent(env.action_space)

        episode_count = 100
        reward = 0
        done = False

        for i in range(episode_count):
            ob = env.reset()

            while True:
                action = agent.act(ob, reward, done)
                ob, reward, done, _ = env.step(action)
                env.render()
                #print(env)
                if done:
                    break
                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        # Dump result info to disk
        env.close()

        # Upload to the scoreboard. We could also do this from another
        # process if we wanted.
        logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
        gym.upload(outdir)

class Agent():
    def __init__(self, environment, alpha=0.1, epsilon=0.1, gamma=1):
        self.env = environment
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

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
    state_representation = np.array([player_vel, pipe_1_x_diff, pipe_1_y_diff, pipe_gap])
    return state_representation

def getQ(F, theta):
    Q = 0
    for i in F:
        Q += theta[i]
    return Q

def play(episodes=100):
    # Initialize game and agent
    game = FlappyBird()
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    p.init()
    agent = Agent(p)
    tiling = Tiling(-39, 300, -10, 16, -50, 50)

    num_tilings = 5
    num_distance_tiles = 10
    num_height_tiles = 10
    num_velocity_tiles = 13
    num_actions = 2
    num_tiles = num_tilings * num_distance_tiles * num_height_tiles * num_velocity_tiles * num_actions

    theta = np.zeros(num_tiles)

    # Run given number of episodes
    for _ in range(episodes):
        p.reset_game()
        e = np.zeros(num_tiles)

        # Get initial state and action
        state = process_state(p.getGameState())
        action = agent.choose_action()
        print(p.getGameState())

        while not p.game_over():

            # Get features that are 'on'
            F = t.getFeatures(state, action)

            for i in F:
                e[i] = 1 # replacing traces

            # Take action and observe reward and next state
            reward = p.act(action)
            state_prime = process_state(p.getGameState())

            delta = reward - getQ(F, theta)

            if np.random.uniform(0, 1) < (1 - agent.epsilon):
                Qs = np.zeros(3)
                for a in env.getActionSet():
                    F = genIndices(s, a, tilings)
                    Qa = calcQ(F, theta)
                    Qs[a + 1] = Qa
                maxQ = max(Q)
                if (Qs == maxQ).sum() > 1:
                    best = [i for i in range(len(actions)) if Qs[i] == maxQ]
                    i = random.choice(best)
                else:
                    i = np.argmax(Q)
                action = actions[i]
                Qa = Qs[i]
                return action, Qa

            #print(state,action,reward)


play()