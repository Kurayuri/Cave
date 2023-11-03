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
        self.viewer = None

        self.max_obs = 10.0
        high = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        p, v, th, thdot = self.state
        terminated = False
        truncated = False
        # print(p, v, th, thdot)

        u = np.clip(u, -2.0, 2.0)[0]

        t = 0.02
        p_new = p + v * t
        v_new = v + (-p + 0.1 * np.sin(th)) * t
        th_new = th + thdot * t
        thdot_new = thdot + u * t

        terminated = bool(
            abs(p_new) > 5.0 or
            abs(v_new) > 5.0 or
            abs(th_new) > math.pi / 2.0 or
            abs(thdot_new) > 5.0
        )

        p_new = np.clip(p_new, -self.max_obs, self.max_obs)
        v_new = np.clip(v_new, -self.max_obs, self.max_obs)
        th_new = self.angle_normalize(th_new)
        thdot_new = np.clip(thdot_new, -self.max_obs, self.max_obs)


        self.state = np.array([p_new, v_new, th_new, thdot_new], dtype=np.float32)

        reward = 1.0

        # terminated = bool(abs(p_new) > 1.5 or abs(p_new) < -1.5 or abs(th_new) > math.pi / 2.0 or abs(th_new) < -math.pi / 2.0)
        done = bool(
            abs(p_new) > 5.0 or
            abs(v_new) > 5.0 or
            abs(th_new) > math.pi / 2.0 or
            abs(thdot_new) > 5.0
        )

        if terminated:
            reward = 0

        return self._get_obs(), reward, terminated, truncated ,{}

    def reset(self,seed=None, options=None):
        high = np.array([-0.75, -0.43, 0.54, -0.28])
        low = np.array([-0.77, -0.45, 0.51, -0.3])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs(),{}

    def _get_obs(self):
        return self.state
        # return np.array(transition([np.cos(theta), np.sin(theta), thetadot]))

    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
