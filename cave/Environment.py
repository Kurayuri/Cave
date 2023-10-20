import os
import random
from typing import Any
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from typing import Union

def maker_Environment(env_id, env_kwargs, 
                      reward_api: Union[str,callable] = None, 
                      log_dirpath: str = "", rank: int = None):
    if rank is not None and log_dirpath:
        log_dirpath = os.path.join(log_dirpath, "rank_%02d" % rank)
    return lambda: Environment(env_id, env_kwargs, reward_api, log_dirpath, rank)

class Environment(gym.Wrapper):
    ATTR_REWARD_API = "reward_api"
    ATTR_TRACEBACKS = "tracebacks"

    def __init__(self, env_id, env_kwargs, 
                 reward_api: Union[str,callable] = None,
                 log_dirpath: str = "", rank: int=0):
        env = gym.make(env_id, **env_kwargs)

        super().__init__(env)

        # TODO
        self.reward_api = reward_api
        self.rank = rank
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

        # Traceback
        self.tracebacks = []

        # logger
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

        self.tracebacks.append(_reward_ - reward)

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

    def reset(self, *, seed = None, options = {}):
        self.counter_step_per_episode = 0
        return super().reset(seed=seed, options=options)


class CallBack(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.trace_gamma = 0.5
        self.trace_depth = 5

    def _init_callback(self):
        self.buffer = self.model.rollout_buffer if isinstance(
            self.model, OnPolicyAlgorithm) else self.model.replay_buffer
        self.trace_enabled = self.training_env.get_attr(Environment.ATTR_REWARD_API)[0] is not None
        
    def _on_rollout_end(self):
        if self.trace_enabled:
            tracebacks = np.array(self.training_env.get_attr(Environment.ATTR_TRACEBACKS)).T
            self.training_env.set_attr(Environment.ATTR_TRACEBACKS, [])
            num_rows = tracebacks.shape[0]
            for i in range(num_rows - 2, -1, -1):
                tracebacks[i] += tracebacks[i + 1] * self.trace_gamma
            self.buffer.rewards[:num_rows] += tracebacks

    def _on_step(self):
        return True
