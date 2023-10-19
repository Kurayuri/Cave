from .import Trace
from .Link import Link
from .Network import Network
from .Sender import Sender
from ...CONSTANT import *
from ...util.lib import pcc_aurora_reward
from ...common import sender_obs
import gymnasium as gym
import numpy as np
from typing import Tuple, List
import heapq
import os
import random
import sys
import time
import warnings


class NetworkCCEnv(gym.Env):
    def __init__(self, history_len=10,
                 # features="sent latency inflation,latency ratio,send ratio",
                 features="sent latency inflation,latency ratio,recv ratio",
                 train_flag=True, delta_scale=1.0, traces=[],  env_config=None, real_trace_prob: float = 0,
                 record_pkt_log: bool = False):
        """Network environment used in simulation.
        congestion_control_type: aurora is pcc-rl. cubic is TCPCubic.
        """
        self.real_trace_prob = real_trace_prob
        self.trace_config = env_config
        self.traces = traces

        self.record_pkt_log = record_pkt_log
        self.delta_scale = delta_scale
        self.train_flag = train_flag
        

        if self.train_flag and self.traces:
            self.real_trace_configs = []
            for trace in self.traces:
                self.real_trace_configs.append(trace.real_trace_configs(True))
            self.real_trace_configs = np.array(self.real_trace_configs).reshape(-1, 4)
        else:
            self.real_trace_configs = None



        # Observation
        self.history_len = history_len
        self.features = features.split(",")
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = gym.spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                                np.tile(single_obs_max_vec, self.history_len),
                                                dtype=np.float32)

        # Action
        self.action_type = Sender.ACTION_RATE_DELTA
        if self.action_type == Sender.ACTION_CWND_DELTA:
            self.action_space = gym.spaces.Box(
                np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        if self.action_type == Sender.ACTION_RATE_DELTA:
            self.action_space = gym.spaces.Box(
                np.array([-1e12]), np.array([1e12]), dtype=np.float32)


        # Network
        self.episodes_run = -1
        self.reward_ewma = 0.0
        self.reward_sum = 0.0
        self.steps_taken = 0

        # self.reset()


        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

    def reset(self, seed=None, options=None):

        # choose real trace with a probability. otherwise, use synthetic trace
        if self.train_flag and self.trace_config:
            self.current_trace = Trace.generate_trace_from_config(self.trace_config)
            if random.uniform(0, 1) < self.real_trace_prob and self.traces:
                config_syn = np.array(self.current_trace.real_trace_configs(normalized=True)).reshape(1, -1)
                assert self.real_trace_configs is not None
                dists = np.linalg.norm(self.real_trace_configs - config_syn, axis=1)
                target_idx = np.argmin(dists)
                real_trace = self.traces[target_idx]
                # real_trace = np.random.choice(self.traces)  # randomly select a real trace
                real_trace.queue_size = self.current_trace.queue_size
                real_trace.loss_rate = self.current_trace.loss_rate
                self.current_trace = real_trace
        else:
            self.current_trace = np.random.choice(self.traces)
        self.current_trace.reset()

        # Newtwork
        self.links = [Link(self.current_trace), Link(self.current_trace)]
        self.senders = [Sender(
            10 / (self.current_trace.get_delay(0) * 2 / 1000),
            [self.links[0], self.links[1]], 0,
            self.features,
            history_len=self.history_len,
            delta_scale=self.delta_scale)]
        
        if not self.senders[0].rtt_samples:
            self.timestamp_granularity = 0.01
        self.net = Network(self.senders, self.links, self)



        self.episodes_run += 1        
        self.net.run_for_duration(self.timestamp_granularity)
        self.reward_ewma = 0.99 * self.reward_ewma + 0.01 * self.reward_sum
        
        # print(self.episodes_run,self.reward_ewma,self.reward_sum, self.steps_taken)
        self.reward_sum = 0.0
        self.steps_taken = 0
        return self.get_sender_obs(), {}

    def get_sender_obs(self):
        return np.array(self.senders[0].get_obs()).reshape(-1,)

    def step(self, actions):
        for i in range(len(actions)):
            self.senders[i].act(self.action_type, actions[i])
        reward = self.net.run_for_duration(self.timestamp_granularity, action=actions[0])
        self.steps_taken += 1
        sender_obs = self.get_sender_obs()
        terminated = self.current_trace.is_terminated(self.net.get_curr_time())
        truncated = self.observation_space.contains(sender_obs)
        self.reward_sum += reward
        # print(sender_obs[:3],reward)
        return sender_obs, reward, terminated, truncated, {}
    