from .Trace import generate_traces
from .Link import Link
from .Network import Network
from .Sender import Sender
from ...CONSTANT import *
from ...common.utils import pcc_aurora_reward
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
    def __init__(self, traces, history_len=10,
                 # features="sent latency inflation,latency ratio,send ratio",
                 features="sent latency inflation,latency ratio,recv ratio",
                 train_flag=False, delta_scale=1.0, config_file=None,
                 record_pkt_log: bool = False, real_trace_prob: float = 0, reward_api: str = ""):
        """Network environment used in simulation.
        congestion_control_type: aurora is pcc-rl. cubic is TCPCubic.
        """
        self.real_trace_prob = real_trace_prob
        self.record_pkt_log = record_pkt_log
        self.config_file = config_file
        self.delta_scale = delta_scale
        self.traces = traces
        self.train_flag = train_flag
        self.reward_api = reward_api
        if self.config_file:
            self.current_trace = generate_traces(self.config_file, 1, 30)[0]
        elif self.traces:
            self.current_trace = np.random.choice(self.traces)
        else:
            raise ValueError
        if self.train_flag and self.traces:
            self.real_trace_configs = []
            for trace in self.traces:
                self.real_trace_configs.append(trace.real_trace_configs(True))
            self.real_trace_configs = np.array(self.real_trace_configs).reshape(-1, 4)
        else:
            self.real_trace_configs = None
        self.use_cwnd = False

        self.history_len = history_len
        # print("History length: %d" % history_len)
        self.features = features.split(",")
        # print("Features: %s" % str(self.features))

        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links, self)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        if self.use_cwnd:
            self.action_space = gym.spaces.Box(
                np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(
                np.array([-1e12]), np.array([1e12]), dtype=np.float32)

        self.observation_space = None
        # use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = gym.spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec,
                                                    self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.episodes_run = -1

        # TODO
        self.reward_api = reward_api
        self.is_violated_func = None
        self.reward_func = None
        if self.reward_api:
            if not callable(self.reward_api):
                exec(open(self.reward_api).read())
                try:
                    self.is_violated_func = locals()["is_violated"]
                    self.reward_func = locals()["reward"]
                except BaseException:
                    pass

    def seed(self, seed=None):
        self.rand, seed = gym.seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        assert self.senders
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        return sender_obs

    def step(self, actions):
        assert self.senders
        #print("Actions: %s" % str(actions))
        # print(actions)
        for i in range(0, 1):  # len(actions)):
            #print("Updating rate for sender %d" % i)
            action = actions
            self.senders[i].apply_rate_delta(action[0])
            if self.use_cwnd:
                self.senders[i].apply_cwnd_delta(action[1])
        # print("Running for %fs" % self.run_dur)
        reward = self.net.run_for_dur(self.run_dur, action=actions[0])
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()

        should_stop = self.current_trace.is_finished(self.net.get_cur_time())
        self.reward_sum += reward
        # print('env step: {}s'.format(time.time() - t_start))

        # sender_obs = np.array([self.senders[0].send_rate,
        #         self.senders[0].avg_latency,
        #         self.senders[0].lat_diff, int(self.senders[0].start_stage),
        #         self.senders[0].max_tput, self.senders[0].min_rtt,
        #         self.senders[0].latest_rtt])
        if self.reward_api:
            reward = self.call_reward_api(sender_obs, action, reward)
        return sender_obs, reward, should_stop,should_stop, {}

    def call_reward_api(self, sender_obs, action, reward):
        if callable(self.reward_api):
            reward, violated = self.reward_api([list(sender_obs)], list(action))
        else:
            violated = self.is_violated_func([list(sender_obs)], list(action))
            reward = self.reward_func(violated, reward)
        return reward

    def create_new_links_and_senders(self):
        self.links = [Link(self.current_trace), Link(self.current_trace)]
        self.senders = [Sender(
            10 / (self.current_trace.get_delay(0) * 2 / 1000),
            [self.links[0], self.links[1]], 0,
            self.features,
            history_len=self.history_len,
            delta_scale=self.delta_scale)]
        # self.run_dur = 3 * lat
        # self.run_dur = 1 * lat
        if not self.senders[0].rtt_samples:
            # self.run_dur = 0.473
            # self.run_dur = 5 / self.senders[0].rate
            self.run_dur = 0.01
            # self.run_dur = self.current_trace.get_delay(0) * 2 / 1000

    def reset(self,seed,options):
        self.steps_taken = 0
        self.net.reset()
        print(seed)
        # old snippet start
        # self.current_trace = np.random.choice(self.traces)
        # old snippet end

        # choose real trace with a probability. otherwise, use synthetic trace
        if self.train_flag and self.config_file:
            self.current_trace = generate_traces(self.config_file, 1, duration=30)[0]
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

        # if self.train_flag and not self.config_file:
        #     bdp = np.max(self.current_trace.bandwidths) / BYTES_PER_PACKET / \
        #             BITS_PER_BYTE * 1e6 * np.max(self.current_trace.delays) * 2 / 1000
        #     self.current_trace.queue_size = max(2, int(bdp * np.random.uniform(0.2, 3.0))) # hard code this for now
        #     loss_rate_exponent = float(np.random.uniform(np.log10(0+1e-5), np.log10(0.5+1e-5), 1))
        #     if loss_rate_exponent < -4:
        #         loss_rate = 0
        #     else:
        #         loss_rate = 10**loss_rate_exponent
        #     self.current_trace.loss_rate = loss_rate

        self.current_trace.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links, self)
        self.episodes_run += 1

        # old code snippet start
        # if self.train_flag and self.config_file is not None and self.episodes_run % 100 == 0:
        #     self.traces = generate_traces(self.config_file, 10, duration=30)
        # old code snippet end
        self.net.run_for_dur(self.run_dur)
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        self.reward_sum = 0.0
        print(self.reward_ewma,end="\t")
        return self._get_all_sender_obs(),{}


