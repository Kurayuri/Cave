import os
import random
from typing import Any
import numpy as np
import gymnasium as gym
from .gymenv import ENVS
from .gymenv.networkcc_v0 import Trace



class Environment(gym.Wrapper):
    def __init__(self, env_name, env_config, reward_api: str = "", log_dirpath: str = ""):
        # env = gym.make(env_name, render_mode="rgb_array")
        env = ENVS[env_name](env_config=env_config)

        super().__init__(env)

        # TODO
        self.reward_api = reward_api
        self.is_violated_func = None
        self.get_reward_func = None
        self.log_dirpath = log_dirpath

        self.counter_step_per_episode = 0
        self.counter_episode = 0

        if self.reward_api:
            if not callable(self.reward_api):
                try:
                    exec(open(self.reward_api).read())
                    self.is_violated_func = locals()["is_violated"]
                    self.get_reward_func = locals()["get_reward"]
                except BaseException:
                    print("Invalid reward_api!")

        self.logger_all = None
        self.logger_occurred = None
        self.logger_violated = None
        self.logger_episode = None
        if self.log_dirpath:
            os.makedirs(self.log_dirpath, exist_ok=True)
            self.logger_all = open(os.path.join(log_dirpath, "all.log"), "w")
            self.logger_occurred = open(os.path.join(log_dirpath, "occurred.log"), "w")
            self.logger_violated = open(os.path.join(log_dirpath, "violated.log"), "w")
            self.logger_episode = open(os.path.join(log_dirpath, "episode.log"), "w")

    def call_reward_api(self, obs, action, reward):
        if callable(self.reward_api):
            _reward_, occured, violated = self.reward_api(obs, action, reward)
        else:
            occured, violated = self.is_violated_func(obs.reshape(1, -1), action.reshape(1, -1))
            _reward_ = self.get_reward_func(violated, reward, obs.reshape(1, -1), action.reshape(1, -1))

        self.log(obs, action, reward, _reward_, logger=self.logger_all)
        if occured:
            self.log(obs, action, reward, _reward_, logger=self.logger_occurred)
        if violated:
            self.log(obs, action, reward, _reward_, logger=self.logger_violated)

        return _reward_

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.counter_step_per_episode += 1

        # Add train
        # if random.random()<5e-4:
        #     obs[0]=random.uniform(-2.4,-2)
        #     obs[2]=random.uniform(0.15,0.21)

        if self.reward_api:
            reward = self.call_reward_api(obs, action, reward)
        if terminated or truncated:
            self.counter_episode += 1
            self.log(f"Steps: {self.counter_step_per_episode} Total Episodes: {self.counter_episode}", logger=self.logger_episode)
        return obs, reward, terminated, truncated, info

    def log(self, *args, logger):
        if logger:
            logger.write(" ".join(map(str, args)) + "\n")
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        
        return super().reset(seed=seed, options=options)
