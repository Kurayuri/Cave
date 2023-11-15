import os
import random
from collections import deque
from typing import Any
from typing import Union, Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor,safe_mean

from .import CONSTANT
from .import util
from .util.lib import safe_divison


def maker_Environment(env_id, env_kwargs, env_options: dict = {},
                      reward_api: Union[str, Dict[str, callable]] = None,
                      log_dirpath: str = ""):
    def _maker_():
        return Monitor(Environment(env_id, env_kwargs, env_options, reward_api, log_dirpath))
    return _maker_


class Environment(gym.Wrapper):
    ATTR_REWARD_API = "reward_api"
    ATTR_REWARD_TRACE = "reward_trace"

    GET_REWARD_FUNC_ID = "get_reward"
    IS_VIOLATED_FUNC_ID = "is_violated"

    CAVE = "cave"
    LOG_ALL = "all"
    LOG_EPISODE = "ep"

    # PREFIX_EP = "ep"
    # PREFIX_SUM = "sum"
    # SUFFIX_OCCURRED = "occ"
    # SUFFIX_VIOLATED = "vio"
    # LOG_EP_OCCURRED = f'{PREFIX_EP}_{SUFFIX_OCCURRED}'
    # LOG_EP_VIOLATED = f'{PREFIX_EP}_{SUFFIX_VIOLATED}'
    # LOG_SUM_OCCURRED = f'{PREFIX_SUM}_{SUFFIX_OCCURRED}'
    # LOG_SUM_VIOLATED = f'{PREFIX_SUM}_{SUFFIX_VIOLATED}'
    LOG_EP_OCCURRED = 'ep_occ'
    LOG_EP_VIOLATED = 'ep_vio'
    LOG_EP_REWARD_DELTA = 'ep_r_delta'
    LOG_SUM_OCCURRED = 'sum_occ'
    LOG_SUM_VIOLATED = 'sum_vio'

    LOGS_EP  = (LOG_ALL, LOG_EPISODE, LOG_EP_OCCURRED, LOG_EP_VIOLATED, LOG_EP_REWARD_DELTA)
    LOGS_SUM_DEPENDENCY = {LOG_SUM_OCCURRED:LOG_EP_OCCURRED, LOG_SUM_VIOLATED:LOG_EP_VIOLATED}
    LOGS_EP_CONTRIBUTION = {v:k for k,v in LOGS_SUM_DEPENDENCY.items()}
    LOGS_SUM = LOGS_SUM_DEPENDENCY.keys()

    LOGS = (*LOGS_EP, *LOGS_SUM)
    LOG_FILENAMES = {LOG_ALL:"all.log", LOG_EPISODE:"episode.log", LOG_EP_OCCURRED:"occurred.log", LOG_EP_VIOLATED:"violated.log"}
    
    LOG_DST_LOG_AND_INFO = 0
    LOG_DST_LOG_ONLY = 1
    LOG_DST_INFO_ONLY = 2

    LOG_MODE_ADD_1 = 1
    LOG_MODE_ADD_ARG = 2

    def __init__(self, env_id, env_kwargs, env_options: dict = {},
                 reward_api: Optional[Union[str, Dict[str, callable]]] = None,
                 log_dirpath: str = ""):
        env = gym.make(env_id, **env_kwargs)

        super().__init__(env)

        self.reward_api = reward_api
        self.env_options = env_options

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
                    except BaseException as e:
                        with open(self.reward_api) as f:
                            exec(f.read())

                        self.is_violated_func = locals()[self.IS_VIOLATED_FUNC_ID]
                        self.get_reward_func = locals()[self.GET_REWARD_FUNC_ID]
            except BaseException as e:
                raise BaseException("Invalid reward_api.")

        util.log(f"Reward API: {self.reward_api}", level=CONSTANT.DEBUG)

        # Reward Trace
        setattr(self, self.ATTR_REWARD_TRACE, [])

        # logger
        self.enabled_infos = [self.LOG_ALL, self.LOG_EPISODE]
        if self.reward_api:
            self.enabled_infos.extend([self.LOG_EP_OCCURRED, self.LOG_EP_VIOLATED, self.LOG_EP_REWARD_DELTA,self.LOG_SUM_OCCURRED,self.LOG_SUM_VIOLATED])
        self.infos={k:0 for k in self.enabled_infos}
        
    
        self.loggers = {}
        self.enabled_loggers=[]
        if self.log_dirpath:
            os.makedirs(self.log_dirpath, exist_ok=True)
            self.enabled_loggers =  [self.LOG_ALL, self.LOG_EPISODE]
            if self.reward_api:
                self.enabled_loggers = [self.LOG_EP_OCCURRED,self.LOG_EP_VIOLATED]

            for k in self.enabled_loggers:
                self.loggers[k]=open(os.path.join(log_dirpath, self.LOG_FILENAMES[k]), "w")

            


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.counter_step_per_episode += 1

        # Add train
        # if random.random()<5e-4:
        #     obs[0]=random.uniform(-2.4,-2)
        #     obs[2]=random.uniform(0.15,0.21)

        _reward_ = self.call_reward_api(obs, action, reward, terminated, truncated)

        self.log(obs, action, reward, _reward_, logger=self.LOG_ALL)

        if terminated or truncated:
            self.counter_episode += 1
            self.log(f"Steps: {self.counter_step_per_episode} Total Episodes: {self.counter_episode}", logger=self.LOG_EPISODE)

            [logger.flush() if logger else None for logger in self.loggers.values()]

            info[self.CAVE]=dict(self.infos)
            # print(info)
        return obs, _reward_, terminated, truncated, info

    def reset(self, *, seed=None, options={}):
        self.counter_step_per_episode = 0
        options.update(self.env_options)
        obs, info = super().reset(seed=seed, options=options)

        self.update_info_sum()
        info.update({k:self.infos[k] for k in self.LOGS_SUM if k in self.infos})
                
        for k in self.infos.keys():
            if k in self.LOGS_EP:
                self.infos[k] = 0

        return obs, info
    

    def call_reward_api(self, obs, action, reward, terminated, truncated):
        _reward_ = reward

        if self.reward_api:
            occured, violated = self.is_violated_func(obs.reshape(1, -1), action.reshape(1, -1))
            _reward_ = self.get_reward_func(obs.reshape(1, -1), action.reshape(1, -1), reward, violated)

            # if truncated
            _reward_delta = float(_reward_ - reward)
            getattr(self, self.ATTR_REWARD_TRACE).append(_reward_delta)
            self.infos

            if occured:
                self.log(obs, action, reward, _reward_, logger=self.LOG_EP_OCCURRED, log_mode=self.LOG_MODE_ADD_1)
                self.log(_reward_delta, logger=self.LOG_EP_REWARD_DELTA, log_mode=self.LOG_MODE_ADD_ARG)
            if violated:
                self.log(obs, action, reward, _reward_, logger=self.LOG_EP_VIOLATED, log_mode=self.LOG_MODE_ADD_1)

        return _reward_


    def log(self, *args, logger, log_dst=0, log_mode = 0):
        # if log_dst == self.LOG_DST_LOG_AND_INFO or log_dst == self.LOG_DST_LOG_ONLY:
        if self.loggers.get(logger) is not None:
            self.loggers[logger].write(" ".join(map(str, args)) + "\n")

        # if log_dst == self.LOG_DST_LOG_AND_INFO or log_dst == self.LOG_DST_INFO_ONLY:
        if self.infos.get(logger) is not None:
            if log_mode == 1:
                self.infos[logger] += 1
                # self.eager_update_info_sum(logger)
            elif log_mode == 2:
                self.infos[logger] += args[0]

    def update_info_sum(self):
        for log_sum in self.LOGS_SUM:
            log_ep = self.LOGS_SUM_DEPENDENCY[log_sum]
            if self.infos.get(log_ep) is not None:
                self.infos[log_sum] += self.infos[log_ep]
    
    def eager_update_info_sum(self, log_ep):
        log_sum = self.LOGS_EP_CONTRIBUTION[log_ep]
        self.infos[log_sum] += 1
    
    


class CallBack(BaseCallback):
    REWARD_TRACE_GAMME = 0.9
    REWARD_TRACE_DEPTH = 20

    TRACE = "Trace"
    STOP_TRAINING_ON_MAX_EPISODES = "StopOnEpisodes"
    STOP_TRAINING_ON_NO_MODEL_IMPROVEMENT = "StopOnNoImprovement"
    MODULES = [
        TRACE, STOP_TRAINING_ON_NO_MODEL_IMPROVEMENT,
        STOP_TRAINING_ON_MAX_EPISODES
    ]

    INFO_KEY_MONITOR = "episode"
    INFO_KEY_CAVE = Environment.CAVE

    INFO_KEYS = (INFO_KEY_CAVE,INFO_KEY_MONITOR)


    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _init_callback(self):
        self.is_OnPolicyAlgorithm = isinstance(self.model, OnPolicyAlgorithm)
        self.buffer = self.model.rollout_buffer if self.is_OnPolicyAlgorithm else self.model.replay_buffer
        self.enabled_trace = self.training_env.get_attr(
            Environment.ATTR_REWARD_API)[0] is not None
        
        self.ep_info_buffer = deque(maxlen=self.model._stats_window_size)
        self.monitor_ep_info_buffer = deque(maxlen=self.model._stats_window_size)

    def _on_rollout_end(self):
        self.module_trace()

    def module_trace(self):
        if not self.enabled_trace:
            return

        reward_traces = np.array(
            self.training_env.get_attr(Environment.ATTR_REWARD_TRACE)).T
        num_rows = reward_traces.shape[0]

        for i in range(num_rows - 2, -1, -1):
            reward_traces[i] += reward_traces[i + 1] * self.REWARD_TRACE_GAMME
        self.buffer.rewards[:num_rows] += reward_traces

        # Clear Reward Trace
        self.training_env.set_attr(Environment.ATTR_REWARD_TRACE, [])

        # In Stable Baseline3, callback is called after return and advantage computaion for OnPolicyAlgorithm
        # File path: stable_baselines3/common/on_policy_algorithm.py
        ###
        # rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        # callback.on_rollout_end()
        if self.is_OnPolicyAlgorithm:
            with th.no_grad():
                values = self.model.policy.predict_values(
                    obs_as_tensor(self.locals['new_obs'], self.model.device))
            self.buffer.compute_returns_and_advantage(
                last_values=values, dones=self.locals['dones'])

    def _on_step(self):
        infos = self.locals["infos"]
        self.update_info_buffer(infos)
        # self.logger.dump(self.num_timesteps)
        ep_info_buffer=self.ep_info_buffer


        if len(ep_info_buffer) > 0 and self.INFO_KEY_CAVE in ep_info_buffer[0] and Environment.LOG_EP_VIOLATED in ep_info_buffer[0][self.INFO_KEY_CAVE]:
            ep_occurred = [info[self.INFO_KEY_CAVE][Environment.LOG_EP_OCCURRED] for info in ep_info_buffer]
            ep_violated = [info[self.INFO_KEY_CAVE][Environment.LOG_EP_VIOLATED] for info in ep_info_buffer]
            ep_reward_delta = [info[self.INFO_KEY_CAVE][Environment.LOG_EP_REWARD_DELTA] for info in ep_info_buffer]
            ep_occurred_mean = safe_mean(ep_occurred)
            ep_violated_mean = safe_mean(ep_violated)

            ep_len = [info[self.INFO_KEY_MONITOR]["l"] for info in ep_info_buffer]
            ep_len_mean = safe_mean(ep_len)

            self.logger.record("rollout/cave_eps_occurred_mean", ep_occurred_mean)
            self.logger.record("rollout/cave_eps_violated_mean", ep_violated_mean)
            self.logger.record("rollout/cave_epr_v_mean", safe_divison(ep_violated_mean,ep_violated_mean))
            self.logger.record("rollout/cave_epr_o_mean", safe_divison(ep_violated_mean,ep_len_mean))
            self.logger.record("rollout/cave_epr_vt_mean", safe_divison(ep_violated_mean,ep_len_mean))
            self.logger.record("rollout/cave_epd_reward_delta_mean", safe_divison(safe_mean(ep_reward_delta),ep_len_mean))
        return True
    
    def update_info_buffer(self, infos):
        for idx, info in enumerate(infos):            
            maybe_info = {k:info[k] for k in self.INFO_KEYS if k in info}

            if maybe_info:
                self.ep_info_buffer.extend([maybe_info])
            