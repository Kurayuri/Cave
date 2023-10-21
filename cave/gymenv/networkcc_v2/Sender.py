# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .Trace import generate_traces
from .Link import Link
from ...CONSTANT import *
from ...util.lib import pcc_aurora_reward
from ...common import sender_obs
import numpy as np
from typing import Tuple, List
import heapq
import os
import random
import sys
import time
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = True
MAX_LATENCY_NOISE = 1.1

# DEBUG = True
DEBUG = False


def debug_print(msg):
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)


class Sender():
    ACTION_RATE_DELTA = 1
    ACTION_CWND_DELTA = 2
        

    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10,
                 delta_scale=1):
        self.id = Sender._get_next_id()
        self.delta_scale = delta_scale
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.rtt_samples_ts = []
        self.queue_delay_samples = []
        self.prev_rtt_samples = self.rtt_samples
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd
        self.use_cwnd = False
        self.rto = -1
        self.ssthresh = 0
        self.pkt_loss_wait_time = -1
        # self.estRTT = 1000000 / 1e6  # SynInterval in emulation
        # self.RTTVar = self.estRTT / 2  # RTT variance
        self.estRTT = None  # SynInterval in emulation
        self.RTTVar = None  # RTT variance
        self.got_data = False

        self.min_rtt = 10
        self.max_tput = 0
        self.start_stage = True
        self.lat_diff = 0
        self.recv_rate = 0
        self.send_rate = 0
        self.latest_rtt = 0

        # variables to track accross the connection session
        self.tot_sent = 0  # no. of packets
        self.tot_acked = 0  # no. of packets
        self.tot_lost = 0  # no. of packets
        self.cur_avg_latency = 0.0
        self.first_ack_ts = None
        self.last_ack_ts = None
        self.first_sent_ts = None
        self.last_sent_ts = None

        # variables to track binwise measurements
        self.bin_bytes_sent = {}
        self.bin_bytes_acked = {}
        self.lat_ts = []
        self.lats = []
        self.bin_size = 500  # ms

        self.act_func = {
            self.ACTION_RATE_DELTA: self.apply_rate_delta,
            self.ACTION_CWND_DELTA: self.apply_cwnd_delta
        }
    _next_id = 1

    @property
    def avg_sending_rate(self):
        """Average sending rate in packets/second."""
        assert self.last_ack_ts is not None and self.first_ack_ts is not None
        assert self.last_sent_ts is not None and self.first_sent_ts is not None
        return self.tot_sent / (self.last_sent_ts - self.first_sent_ts)

    @property
    def avg_throughput(self):
        """Average throughput in packets/second."""
        assert self.last_ack_ts is not None and self.first_ack_ts is not None
        assert self.last_sent_ts is not None and self.first_sent_ts is not None
        return self.tot_acked / (self.last_ack_ts - self.first_ack_ts)

    @property
    def avg_latency(self):
        """Average latency in second."""
        return self.cur_avg_latency

    @property
    def pkt_loss_rate(self):
        """Packet loss rate in one connection session."""
        return 1 - self.tot_acked / self.tot_sent

    @property
    def bin_tput(self) -> Tuple[List[float], List[float]]:
        tput_ts = []
        tput = []
        for bin_id in sorted(self.bin_bytes_acked):
            tput_ts.append(bin_id * self.bin_size / 1000)
            tput.append(
                self.bin_bytes_acked[bin_id] * BITS_PER_BYTE / self.bin_size * 1000 / 1e6)
        return tput_ts, tput

    @property
    def bin_sending_rate(self) -> Tuple[List[float], List[float]]:
        sending_rate_ts = []
        sending_rate = []
        for bin_id in sorted(self.bin_bytes_sent):
            sending_rate_ts.append(bin_id * self.bin_size / 1000)
            sending_rate.append(
                self.bin_bytes_sent[bin_id] * BITS_PER_BYTE / self.bin_size * 1000 / 1e6)
        return sending_rate_ts, sending_rate

    @property
    def latencies(self) -> Tuple[List[float], List[float]]:
        return self.lat_ts, self.lats

    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result
    

    def act(self,action_type,action):
        self.act_func[action_type](action)


    def apply_rate_delta(self, delta):
        # if self.got_data:
        delta *= self.delta_scale
        # print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def apply_cwnd_delta(self, delta):
        delta *= self.delta_scale
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_pkt(self):
        if self.use_cwnd:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def send_pkt(self):
        assert self.net
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET
        self.tot_sent += 1
        if self.first_sent_ts is None:
            self.first_sent_ts = self.net.ts
        self.last_sent_ts = self.net.ts

        bin_id = int((self.net.ts - self.first_sent_ts) * 1000 / self.bin_size)
        self.bin_bytes_sent[bin_id] = self.bin_bytes_sent.get(bin_id, 0) + BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        assert self.net
        self.cur_avg_latency = (self.cur_avg_latency * self.tot_acked + rtt) / (self.tot_acked + 1)
        self.tot_acked += 1
        if self.first_ack_ts is None:
            self.first_ack_ts = self.net.ts
        self.last_ack_ts = self.net.ts

        self.min_rtt = min(self.min_rtt, rtt)
        if self.estRTT is None and self.RTTVar is None:
            self.estRTT = rtt
            self.RTTVar = rtt / 2
        elif self.estRTT and self.RTTVar:
            self.estRTT = (7.0 * self.estRTT + rtt) / 8.0  # RTT of emulation way
            self.RTTVar = (self.RTTVar * 7.0 + abs(rtt - self.estRTT) * 1.0) / 8.0
        else:
            raise ValueError("srtt and rttvar shouldn't be None.")

        self.acked += 1
        self.rtt_samples.append(rtt)
        self.rtt_samples_ts.append(self.net.ts)
        # self.rtt_samples.append(self.estRTT)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET
        if not self.got_data:
            self.got_data = len(self.rtt_samples) >= 1
        # self.got_data = True

        bin_id = int((self.net.ts - self.first_ack_ts) * 1000 / self.bin_size)
        self.bin_bytes_acked[bin_id] = self.bin_bytes_acked.get(bin_id, 0) + BYTES_PER_PACKET
        self.lat_ts.append(self.net.ts)
        self.lats.append(rtt * 1000)


    def on_packet_lost(self, rtt):
        self.lost += 1
        self.tot_lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate
        # print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        # if self.cwnd > MAX_CWND:
        #     self.cwnd = MAX_CWND
        # if self.cwnd < MIN_CWND:
        #     self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        # if not self.got_data and smi.rtt_samples:
        #     self.got_data = True
        #     self.history.step(smi)
        # else:
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        assert self.net
        obs_end_time = self.net.ts

        #obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)
        # print(self.acked, self.sent)
        if not self.rtt_samples and self.prev_rtt_samples:
            rtt_samples = [np.mean(np.array(self.prev_rtt_samples))]
        else:
            rtt_samples = self.rtt_samples
        # if not self.rtt_samples:
        #     print(self.obs_start_time, obs_end_time, self.rate)
        # rtt_samples is empty when there is no packet acked in MI
        # Solution: inherit from previous rtt_samples.

        # recv_start = self.rtt_samples_ts[0] if len(
        #     self.rtt_samples) >= 2 else self.obs_start_time
        recv_start = self.history.back().recv_end if len(
            self.rtt_samples) >= 1 else self.obs_start_time
        recv_end = self.rtt_samples_ts[-1] if len(
            self.rtt_samples) >= 1 else obs_end_time
        bytes_acked = self.acked * BYTES_PER_PACKET
        if recv_start == 0:
            recv_start = self.rtt_samples_ts[0]
            bytes_acked = (self.acked - 1) * BYTES_PER_PACKET

        # bytes_acked = max(0, (self.acked-1)) * BYTES_PER_PACKET if len(
        #     self.rtt_samples) >= 2 else self.acked * BYTES_PER_PACKET
        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            # max(0, (self.acked-1)) * BYTES_PER_PACKET,
            # bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_acked=bytes_acked,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            # recv_start=self.obs_start_time,
            # recv_end=obs_end_time,
            recv_start=recv_start,
            recv_end=recv_end,
            rtt_samples=rtt_samples,
            queue_delay_samples=self.queue_delay_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        assert self.net
        self.sent = 0
        self.acked = 0
        self.lost = 0
        if self.rtt_samples:
            self.prev_rtt_samples = self.rtt_samples
        self.rtt_samples = []
        self.rtt_samples_ts = []
        self.queue_delay_samples = []
        self.obs_start_time = self.net.ts

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.estRTT = 1000000 / 1e6  # SynInterval in emulation
        self.RTTVar = self.estRTT / 2  # RTT variance

        self.got_data = False
        self.min_rtt = 10
        self.max_tput = 0
        self.start_stage = True
        self.lat_diff = 0
        self.recv_rate = 0
        self.send_rate = 0
        self.latest_rtt = 0

        self.tot_sent = 0  # no. of packets
        self.tot_acked = 0  # no. of packets
        self.tot_lost = 0  # no. of packets
        self.cur_avg_latency = 0.0
        self.first_ack_ts = None
        self.last_ack_ts = None

        self.bin_bytes_sent = {}
        self.bin_bytes_acked = {}
        self.lat_ts = []
        self.lats = []

    def timeout(self):
        # placeholder
        pass

    def on_pkt_acked(self, rtt):
        assert self.net
        self.cur_avg_latency = (self.cur_avg_latency * self.tot_acked + rtt) / (self.tot_acked + 1)
        self.tot_acked += 1
        if self.first_ack_ts is None:
            self.first_ack_ts = self.net.ts
        self.last_ack_ts = self.net.ts

        self.min_rtt = min(self.min_rtt, rtt)
        if self.estRTT is None and self.RTTVar is None:
            self.estRTT = rtt
            self.RTTVar = rtt / 2
        elif self.estRTT and self.RTTVar:
            self.estRTT = (7.0 * self.estRTT + rtt) / 8.0  # RTT of emulation way
            self.RTTVar = (self.RTTVar * 7.0 + abs(rtt - self.estRTT) * 1.0) / 8.0
        else:
            raise ValueError("srtt and rttvar shouldn't be None.")

        self.acked += 1
        self.rtt_samples.append(rtt)
        self.rtt_samples_ts.append(self.net.ts)
        # self.rtt_samples.append(self.estRTT)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET
        if not self.got_data:
            self.got_data = len(self.rtt_samples) >= 1
        # self.got_data = True

        bin_id = int((self.net.ts - self.first_ack_ts) * 1000 / self.bin_size)
        self.bin_bytes_acked[bin_id] = self.bin_bytes_acked.get(bin_id, 0) + BYTES_PER_PACKET
        self.lat_ts.append(self.net.ts)
        self.lats.append(rtt * 1000)

    def on_pkt_lost(self,ts):
        return self.on_packet_lost(ts)