from .import CONSTANT
from .import util
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from typing import Any
from typing import Union, Dict
import gymnasium as gym
import random
import numpy as np
import torch as th
import os


def maker_Environment(env_id, env_kwargs,
                      reward_api: Union[str, Dict[str, callable]] = None,
                      log_dirpath: str = ""):
    def _maker_():
        return Monitor(Environment(env_id, env_kwargs, reward_api, log_dirpath))
        # return gym.make(env_id)
    return _maker_


class Environment(gym.Wrapper):
    ATTR_REWARD_API = "reward_api"
    ATTR_REWARD_TRACE = "reward_trace"

    GET_REWARD_FUNC_ID = "get_reward"
    IS_VIOLATED_FUNC_ID = "is_violated"

    LOG_FILENAME_ALL = "all.log"
    LOG_FILENAME_VIOLATED = "violated.log"
    LOG_FILENAME_OCCURRED = "occurred.log"
    LOG_FILENAME_EPISODE = "episode.log"
    LOG_FILENAMES = (LOG_FILENAME_ALL, LOG_FILENAME_OCCURRED, LOG_FILENAME_VIOLATED, LOG_FILENAME_EPISODE)

    def __init__(self, env_id, env_kwargs,
                 reward_api: Union[str, Dict[str, callable]] = None,
                 log_dirpath: str = ""):
        env = gym.make(env_id, **env_kwargs)

        super().__init__(env)

        self.reward_api = reward_api

        self.is_violated_func = None
        self.get_reward_func = None

        self.log_dirpath = log_dirpath

        self.counter_step_per_episode = 0
        self.counter_episode = 0

        # Reward API
        if self.reward_api:
            try:
                if isinstance(self.reward_api, dict):
                    self.is_violated_func = self.reward_api[self.IS_VIOLATED_FUNC_ID]
                    self.get_reward_func = self.reward_api[self.GET_REWARD_FUNC_ID]
                elif isinstance(self.reward_api, str):
                    try:
                        exec(self.reward_api)
                        self.is_violated_func = locals()[self.IS_VIOLATED_FUNC_ID]
                        self.get_reward_func = locals()[self.GET_REWARD_FUNC_ID]
                    except BaseException:
                        with open(self.reward_api) as f:
                            exec(f.read())

                        self.is_violated_func = locals()[self.IS_VIOLATED_FUNC_ID]
                        self.get_reward_func = locals()[self.GET_REWARD_FUNC_ID]
            except BaseException as e:
                raise BaseException("Invalid reward_api.")

        util.log(f"Reward API: {self.reward_api}", level=CONSTANT.INFO)

        # Reward Trace
        setattr(self, self.ATTR_REWARD_TRACE, [])

        # logger
        self.loggers = {k: None for k in self.LOG_FILENAMES}

        if self.log_dirpath:
            os.makedirs(self.log_dirpath, exist_ok=True)
            self.loggers[self.LOG_FILENAME_ALL] = open(os.path.join(log_dirpath, self.LOG_FILENAME_ALL), "w")
            self.loggers[self.LOG_FILENAME_EPISODE] = open(os.path.join(log_dirpath, self.LOG_FILENAME_EPISODE), "w")
            if self.reward_api:
                self.loggers[self.LOG_FILENAME_OCCURRED] = open(os.path.join(log_dirpath, self.LOG_FILENAME_OCCURRED), "w")
                self.loggers[self.LOG_FILENAME_VIOLATED] = open(os.path.join(log_dirpath, self.LOG_FILENAME_VIOLATED), "w")

    def call_reward_api(self, obs, action, reward):
        _reward_ = reward

        if self.reward_api:
            occured, violated = self.is_violated_func(obs.reshape(1, -1), action.reshape(1, -1))
            _reward_ = self.get_reward_func(obs.reshape(1, -1), action.reshape(1, -1), reward, violated)

            getattr(self, self.ATTR_REWARD_TRACE).append(float(_reward_ - reward))

            if occured:
                self.log(obs, action, reward, _reward_, logger=self.LOG_FILENAME_OCCURRED)
            if violated:
                self.log(obs, action, reward, _reward_, logger=self.LOG_FILENAME_VIOLATED)

        return _reward_

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.counter_step_per_episode += 1

        # Add train
        # if random.random()<5e-4:
        #     obs[0]=random.uniform(-2.4,-2)
        #     obs[2]=random.uniform(0.15,0.21)

        _reward_ = self.call_reward_api(obs, action, reward)

        self.log(obs, action, reward, _reward_, logger=self.LOG_FILENAME_ALL)

        if terminated or truncated:
            self.counter_episode += 1
            self.log(f"Steps: {self.counter_step_per_episode} Total Episodes: {self.counter_episode}", logger=self.LOG_FILENAME_EPISODE)

            [logger.flush() if logger else None for logger in self.loggers.values()]

        return obs, _reward_, terminated, truncated, info

    def log(self, *args, logger):
        if self.loggers[logger]:
            self.loggers[logger].write(" ".join(map(str, args)) + "\n")

    def reset(self, *, seed=None, options={}):
        self.counter_step_per_episode = 0
        return super().reset(seed=seed, options=options)


class CallBack(BaseCallback):
    REWARD_TRACE_GAMME = 0.5
    REWARD_TRACE_DEPTH = 5

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _init_callback(self):
        self.is_OnPolicyAlgorithm = isinstance(self.model, OnPolicyAlgorithm)
        self.buffer = self.model.rollout_buffer if self.is_OnPolicyAlgorithm else self.model.replay_buffer
        self.trace_enabled = self.training_env.get_attr(Environment.ATTR_REWARD_API)[0] is not None

    def _on_rollout_end(self):
        if self.trace_enabled:
            reward_traces = np.array(self.training_env.get_attr(Environment.ATTR_REWARD_TRACE)).T
            num_rows = reward_traces.shape[0]

            for i in range(num_rows - 2, -1, -1):
                reward_traces[i] += reward_traces[i + 1] * self.REWARD_TRACE_GAMME
            self.buffer.rewards[:num_rows] += reward_traces

            # Clear Reward Trace
            self.training_env.set_attr(Environment.ATTR_REWARD_TRACE, [])

            ### In Stable Baseline3, callback is called after return and advantage computaion for OnPolicyAlgorithm
            ### File path: stable_baselines3/common/on_policy_algorithm.py
            ###
            ## rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
            ## callback.on_rollout_end()
            if self.is_OnPolicyAlgorithm:
                with th.no_grad():
                    values = self.model.policy.predict_values(obs_as_tensor(self.locals['new_obs'], self.model.device))
                self.buffer.compute_returns_and_advantage(last_values=values, dones=self.locals['dones'])
            

    def _on_step(self):
        return True
