import imp
from optparse import NO_DEFAULT
import os
import random
import inspect
from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from cave import Const
from cave import utils
from cave.Settings import messager
from libpycom.math import safe_div, ewma


def make_Environment_fn(env_id, env_kwargs, env_options: dict = {},
                        reward_api: str | dict[str, Callable] | None = None,
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
    LOG_EP_REWARD_RAW = 'ep_r_raw'
    LOG_SUM_OCCURRED = 'sum_occ'
    LOG_SUM_VIOLATED = 'sum_vio'

    # LOG_ID = "id"

    # log_per_episode
    LOGS_EP = (LOG_ALL, LOG_EPISODE, LOG_EP_OCCURRED,
               LOG_EP_VIOLATED, LOG_EP_REWARD_DELTA, LOG_EP_REWARD_RAW)
    # log_in_sum's dependency on which log_per_episode
    LOGS_SUM_DEPENDENCY = {
        LOG_SUM_OCCURRED: LOG_EP_OCCURRED, LOG_SUM_VIOLATED: LOG_EP_VIOLATED}
    # log_per_episode's contribution to which log_in_sum
    LOGS_EP_CONTRIBUTION = {v: k for k, v in LOGS_SUM_DEPENDENCY.items()}
    LOGS_SUM = LOGS_SUM_DEPENDENCY.keys()

    LOGS = (*LOGS_EP, *LOGS_SUM)
    LOG_FILENAMES = {LOG_ALL: "all.log", LOG_EPISODE: "episode.log",
                     LOG_EP_OCCURRED: "occurred.log", LOG_EP_VIOLATED: "violated.log"}

    LOG_DST_LOG_AND_INFO = 0
    LOG_DST_LOG_ONLY = 1
    LOG_DST_INFO_ONLY = 2

    LOG_MODE_ADD_1 = 1
    LOG_MODE_ADD_ARG = 2

    def __init__(self, env_id, env_kwargs, env_options: dict = {},
                 reward_api: str | dict[str, Callable] | None = None,
                 log_dirpath: str = ""):
        env = gym.make(env_id, **env_kwargs)

        super().__init__(env)

        self.reward_api = reward_api
        self.env_options = env_options

        self.fn_is_violated = None
        self.fn_get_reward = None

        self.log_dirpath = log_dirpath

        self.counter_step_per_episode = 0
        self.counter_episode = 0

        # Reward API
        if self.reward_api:
            try:
                if isinstance(self.reward_api, dict):
                    self.fn_is_violated = self.reward_api[self.IS_VIOLATED_FUNC_ID]
                    self.fn_get_reward = self.reward_api[self.GET_REWARD_FUNC_ID]
                elif isinstance(self.reward_api, str):
                    try:
                        exec(self.reward_api)
                        self.fn_is_violated = locals()[
                            self.IS_VIOLATED_FUNC_ID]
                        self.fn_get_reward = locals()[
                            self.GET_REWARD_FUNC_ID]
                    except BaseException as e:
                        with open(self.reward_api) as f:
                            exec(f.read())

                        self.fn_is_violated = locals()[
                            self.IS_VIOLATED_FUNC_ID]
                        self.fn_get_reward = locals()[
                            self.GET_REWARD_FUNC_ID]
            except BaseException as e:
                raise BaseException("Invalid reward_api.")
        self.messager = messager
        self.messager.debug(f"Reward API: {self.reward_api}")

        # Reward Trace
        setattr(self, self.ATTR_REWARD_TRACE, [])

        # logger
        self.enabled_infos = [self.LOG_ALL, self.LOG_EPISODE]
        if self.reward_api:
            self.enabled_infos.extend([self.LOG_EP_OCCURRED, self.LOG_EP_VIOLATED,
                                      self.LOG_EP_REWARD_DELTA, self.LOG_EP_REWARD_RAW,
                                      self.LOG_SUM_OCCURRED, self.LOG_SUM_VIOLATED])
        self.infos = {k: 0 for k in self.enabled_infos}

        self.loggers = {}
        self.enabled_loggers = []
        if self.log_dirpath:
            os.makedirs(self.log_dirpath, exist_ok=True)
            self.enabled_loggers = [self.LOG_ALL, self.LOG_EPISODE]
            if self.reward_api:
                self.enabled_loggers.extend(
                    [self.LOG_EP_OCCURRED, self.LOG_EP_VIOLATED])

            for k in self.enabled_loggers:
                self.loggers[k] = open(os.path.join(
                    log_dirpath, self.LOG_FILENAMES[k]), "w")

    def __del__(self):
        for logger in self.loggers.values():
            logger.close()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.counter_step_per_episode += 1

        # Add train
        # if random.random()<5e-4:
        #     obs[0]=random.uniform(-2.4,-2)
        #     obs[2]=random.uniform(0.15,0.21)

        _reward_ = self.call_reward_api(
            obs, action, reward, terminated, truncated)

        self.log(obs, action, reward, _reward_, logger=self.LOG_ALL)
        self.log(reward, logger=self.LOG_EP_REWARD_RAW,
                 log_mode=self.LOG_MODE_ADD_ARG)

        if terminated or truncated:
            self.counter_episode += 1
            self.log(
                f"Steps: {self.counter_step_per_episode} Total Episodes: {self.counter_episode}", logger=self.LOG_EPISODE)

            [logger.flush() if logger else None for logger in self.loggers.values()]

            info = self.edit_info(info, self.infos)
        return obs, _reward_, terminated, truncated, info

    def reset(self, *, seed=None, options={}):
        self.counter_step_per_episode = 0
        options.update(self.env_options)
        obs, info = super().reset(seed=seed, options=options)

        self.update_info_sum()

        info = self.edit_info(
            info, {k: self.infos[k] for k in self.LOGS_SUM if k in self.infos})

        for k in self.infos.keys():
            if k in self.LOGS_EP:
                self.infos[k] = 0

        return obs, info

    def call_reward_api(self, obs, action, reward, terminated, truncated):
        _reward_ = reward

        if self.reward_api:
            try:
                occured, violated = self.fn_is_violated(
                    obs.reshape(1, -1), action.reshape(1, -1))
            except Exception:
                print(inspect.getsource(self.fn_is_violated))
                print(obs.reshape(1, -1))
                print(action.reshape(1, -1))
                raise Exception
            try:
                _reward_ = self.fn_get_reward(obs.reshape(
                    1, -1), action.reshape(1, -1), reward, violated)
            except Exception:
                print(inspect.getsource(self.fn_get_reward))
                print(obs.reshape(1, -1))
                print(action.reshape(1, -1))
                raise Exception

            # if truncated
            _reward_delta = float(reward - _reward_)
            getattr(self, self.ATTR_REWARD_TRACE).append(_reward_delta)

            self.log(_reward_delta, logger=self.LOG_EP_REWARD_DELTA,
                     log_mode=self.LOG_MODE_ADD_ARG)
            if occured:
                self.log(obs, action, reward, _reward_,
                         logger=self.LOG_EP_OCCURRED, log_mode=self.LOG_MODE_ADD_1)
            if violated:
                self.log(obs, action, reward, _reward_,
                         logger=self.LOG_EP_VIOLATED, log_mode=self.LOG_MODE_ADD_1)

        return _reward_

    def log(self, *args, logger, log_dst=0, log_mode=0):
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

    def edit_info(self, info, cave_info):
        cave_info = dict(cave_info)
        # cave_info[self.CAVE][self.LOG_ID] = random.getrandbits(16)
        info[self.CAVE] = cave_info
        return info


class CallBack(BaseCallback):
    REWARD_TRACE_GAMME = 0.9
    REWARD_TRACE_DEPTH = 20
    REWARD_EWMA_ALPHA = 0.01

    TRACE = "Trace"
    STOP_TRAINING_ON_MAX_EPISODES = "StopOnEpisodes"
    STOP_TRAINING_ON_NO_MODEL_IMPROVEMENT = "StopOnNoImprovement"
    MODULES = [
        TRACE, STOP_TRAINING_ON_NO_MODEL_IMPROVEMENT,
        STOP_TRAINING_ON_MAX_EPISODES
    ]

    INFO_KEY_MONITOR = "episode"
    INFO_KEY_CAVE = Environment.CAVE

    INFO_KEYS = (INFO_KEY_CAVE, INFO_KEY_MONITOR)

    def __init__(self, enabled_trace: bool = True, verbose=0):
        super().__init__(verbose)
        self.enabled_trace = enabled_trace

    def _init_callback(self):
        self.is_OnPolicyAlgorithm = isinstance(self.model, OnPolicyAlgorithm)
        self.trajectory_buffer = self.model.rollout_buffer if self.is_OnPolicyAlgorithm else self.model.replay_buffer

        self.nproc = len(self.training_env.get_attr("action_space"))

        self.enabled_trace = (self.training_env.get_attr(
            Environment.ATTR_REWARD_API)[0] is not None) and (self.enabled_trace)

        self.ep_info_buffer = deque(maxlen=self.model._stats_window_size)

        self.sum_occurreds = [0 for i in range(self.nproc)]
        self.sum_violateds = [0 for i in range(self.nproc)]
        self.ep_reward_ewma = None
        self.ep_reward_raw_ewma = None

    def _on_rollout_end(self):
        self.module_trace()

    def _on_step(self):
        self.module_tensorboard_record()
        return True

    def module_trace(self):
        if not self.enabled_trace:
            return

        reward_traces = np.array(
            self.training_env.get_attr(Environment.ATTR_REWARD_TRACE)).T
        num_rows = reward_traces.shape[0]

        for i in range(num_rows - 2, -1, -1):
            reward_traces[i] += reward_traces[i + 1] * self.REWARD_TRACE_GAMME
        self.trajectory_buffer.rewards[:num_rows] += reward_traces

        # Clear Reward Trace
        self.training_env.set_attr(Environment.ATTR_REWARD_TRACE, [])

        # NOTE
        # In Stable Baseline3, callback is called after return and advantage computaion for OnPolicyAlgorithm
        # File path: stable_baselines3/common/on_policy_algorithm.py
        ###
        # rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        # callback.on_rollout_end()
        if self.is_OnPolicyAlgorithm:
            with th.no_grad():
                values = self.model.policy.predict_values(
                    obs_as_tensor(self.locals['new_obs'], self.model.device))
            self.trajectory_buffer.compute_returns_and_advantage(
                last_values=values, dones=self.locals['dones'])

    def module_tensorboard_record(self):
        infos = self.locals["infos"]
        buffer_shift_len = self.update_info_buffer(infos)
        self.logger.dump(self.num_timesteps - self.nproc)
        ep_info_buffer = self.ep_info_buffer

        if buffer_shift_len:
            ep_lens = [info[self.INFO_KEY_MONITOR]["l"]
                       for info in ep_info_buffer]
            ep_rewards = [info[self.INFO_KEY_MONITOR]["r"]
                          for info in ep_info_buffer]
            ep_len_mean = safe_mean(ep_lens)

            # ep_reward_ewma
            buffer_len = len(ep_info_buffer)
            self.ep_reward_ewma = ewma(
                ep_rewards, self.REWARD_EWMA_ALPHA,
                init_val=self.ep_reward_ewma,
                start=max(buffer_len - buffer_shift_len, 0),
                stop=buffer_len
            )
            # for i in range(max(buffer_len - buffer_shift, 0), buffer_len):
            #     self.ep_reward_ewma = ep_rewards[i] * self.REWARD_EWMA_ALPHA + \
            #         self.ep_reward_ewma * (1 - self.REWARD_EWMA_ALPHA)

            self.logger.record("rollout/ep_rew_ewma", self.ep_reward_ewma)

            if self.INFO_KEY_CAVE in ep_info_buffer[0] and Environment.LOG_EP_VIOLATED in ep_info_buffer[0][self.INFO_KEY_CAVE]:
                ep_rewards_raw = [
                    info[self.INFO_KEY_CAVE][Environment.LOG_EP_REWARD_RAW] for info in ep_info_buffer]
                ep_occurreds = [
                    info[self.INFO_KEY_CAVE][Environment.LOG_EP_OCCURRED] for info in ep_info_buffer]
                ep_violateds = [
                    info[self.INFO_KEY_CAVE][Environment.LOG_EP_VIOLATED] for info in ep_info_buffer]
                ep_rewards_delta = [
                    info[self.INFO_KEY_CAVE][Environment.LOG_EP_REWARD_DELTA] for info in ep_info_buffer]

                # calc
                self.ep_reward_raw_ewma = ewma(
                    ep_rewards_raw, self.REWARD_EWMA_ALPHA,
                    init_val=self.ep_reward_raw_ewma,
                    start=max(buffer_len - buffer_shift_len, 0),
                    stop=buffer_len
                )
                ep_reward_raw_mean = safe_mean(ep_rewards_raw)
                ep_occurred_mean = safe_mean(ep_occurreds)
                ep_violated_mean = safe_mean(ep_violateds)
                ep_reward_delta_mean = safe_mean(ep_rewards_delta)

                # record
                self.logger.record(
                    "rollout/ep_rew_raw_mean", ep_reward_raw_mean)
                self.logger.record(
                    "rollout/ep_rew_raw_ewma", self.ep_reward_raw_ewma)

                self.logger.record(
                    "rollout/cave_ep_occurred_mean", ep_occurred_mean)
                self.logger.record(
                    "rollout/cave_ep_violated_mean", ep_violated_mean)

                self.logger.record(
                    "rollout/cave_sum_occurred_mean", sum(self.sum_occurreds))
                self.logger.record(
                    "rollout/cave_sum_violated_mean", sum(self.sum_violateds))

                self.logger.record(
                    "rollout/cave_epr_occurred_mean", safe_div(ep_occurred_mean, ep_len_mean))
                self.logger.record(
                    "rollout/cave_epr_violated_mean", safe_div(ep_violated_mean, ep_len_mean))
                self.logger.record(
                    "rollout/cave_epr_vo_mean", safe_div(ep_violated_mean, ep_occurred_mean))

                self.logger.record("rollout/cave_epd_reward_delta_mean",
                                   safe_div(ep_reward_delta_mean, ep_violated_mean))

    def update_info_buffer(self, infos):
        buffer_shift_len = 0
        self.sum_occurred = []
        for proc_idx, info in enumerate(infos):
            # Extract Cave info
            maybe_info = {k: info[k] for k in self.INFO_KEYS if k in info}
            if maybe_info:
                buffer_shift_len += 1
                self.ep_info_buffer.extend([maybe_info])
                if self.INFO_KEY_CAVE in maybe_info:
                    self.sum_violateds[proc_idx] += maybe_info[self.INFO_KEY_CAVE].get(Environment.LOG_SUM_VIOLATED, 0)
                    self.sum_occurreds[proc_idx] += maybe_info[self.INFO_KEY_CAVE].get(Environment.LOG_SUM_OCCURRED, 0)

        return buffer_shift_len
