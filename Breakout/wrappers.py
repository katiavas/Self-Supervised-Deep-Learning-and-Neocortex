import collections
import cv2
import numpy as np
import gym
import random
import torch as T


# https://livebook.manning.com/book/deep-reinforcement-learning-in-action/chapter-8/v-7/63
# Step the environment with the given action Repeat action, sum reward, and max over last observations
class Step(gym.Wrapper):
    def __init__(self, env, repeat=4, fire = True):
        super(Step, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.fire = fire

    def step(self, action):
        total_reward = 0.0
        done = False
        # For each frame
        for i in range(self.repeat):
            # The step method takes an integer representing the action to be taken
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                # obs = self.env.reset()
                break
        return obs, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.fire:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)
        return obs


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, new_shape, env):
        super(PreprocessFrame, self).__init__(env)
        # self.shape = shape.transpose(2, 0, 1)
        self.new_shape = (new_shape[2], new_shape[0], new_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.new_shape,
                                                dtype=np.float32)

    def observation(self, obs):
        return PreprocessFrame.process(obs, self.new_shape)

    @staticmethod
    def process(frame, shape):
        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, shape[1:], interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(shape)
        new_obs = new_obs / 255.0
        return new_obs


''' instead pass the last 3 frames of the game (in essence adding a channel dimension) 
so the states will be 3x42x42.'''


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32)
        # Set our stack which will be a deque of maxlen repeat
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_atari(env_name, shape=(42, 42, 1), repeat=4):
    env = gym.make(env_name)
    # env.seed(111)
    env = Step(env, repeat)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    return env
