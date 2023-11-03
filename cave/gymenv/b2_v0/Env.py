import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import math
from os import path


class Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.th = 1.0
        # self.viewer = None

        high = np.array([2, 2], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.th,
                                       high=self.th,
                                       shape=(1, ),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-high,
                                            high=high,
                                            dtype=np.float32)
        self.max_obs = 5.0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        p, v = self.state
        terminated = False
        truncated = False
        u = np.clip(u, -self.th, self.th)[0]

        t = 0.2
        p_new = p + (v - p * p * p) * t
        v_new = v + u * t

        p_new = np.clip(p_new, -self.max_obs, self.max_obs)
        p_new = np.clip(p_new, -self.max_obs, self.max_obs)

        self.state = np.array([p_new, v_new], dtype=np.float32)

        reward = -2

        terminated = bool(0.1 >= p_new >= -0.3 and 0.5 >= v_new >= -0.35)

        # done = bool(abs(p_new) > 1.5 or abs(v_new) > 1.5) or done

        # if bool(abs(p_new) > 1.5 or abs(v_new) > 1.5):
        #     reward = -600

        return self._get_obs(), reward, terminated,truncated, {}

    def reset(self, seed=None, options=None):
        self.seed(seed)
        high = np.array([0.9, 0.9])
        low = np.array([0.7, 0.7])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs(),{}

    def _get_obs(self):
        self.state[0] = np.clip(self.state[0], -2, 2)
        self.state[1] = np.clip(self.state[1], -2, 2)
        return self.state
        # return np.array(transition([np.cos(theta), np.sin(theta), thetadot]))

    # def angle_normalize(self, x):
    #     return (( (x + np.pi) % (2 * np.pi) ) - np.pi)
