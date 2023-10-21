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
from ...CONSTANT import *
from ...util.lib import pcc_aurora_reward
from ...common import sender_obs
from .Link import Link
from .Sender import Sender
from .Trace import Trace
import numpy as np
from typing import Tuple, List
import heapq
import os
import random
import sys
import time
import warnings
from collections import namedtuple


LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = True
MAX_LATENCY_NOISE = 1.1

DEBUG = True
DEBUG = False


def debug_print(msg):
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)


class Event:
    def __init__(self,ts,id,src,dst,type,next_hop,latency,dropped,rto,event_queue_delay):
        self.ts = ts
        self.id = id
        self.src = src
        self.dst = dst
        self.type = type
        self.next_hop = next_hop
        self.latency = latency
        self.rto = rto
        self.event_queue_delay = event_queue_delay
    
    def __lt__(self, other):
        return self.ts < other.ts
    
    def __repr__(self):
        return f"{__class__.__name__}({', '.join(f'{k}={v}' for k, v in vars(self).items())})" 

class EventQueue:
    def __init__(self):
        self.queue = []

    def top(self):
        return self.queue[0]

    def pop(self):
        return heapq.heappop(self.queue)

    def tail(self):
        return self.queue[-1]

    def push(self, element):
        heapq.heappush(self.queue, element)

    def __repr__(self):
        return repr(self.queue)

def counter():
    count = 0
    while True:
        yield count
        count += 1


class Network():
    def __init__(self, trace, sender_kwargs):
        self.trace = trace
        self.sender_kwargs = sender_kwargs
        self.reset()

    def reset(self):
        self.ts = 0.0
        self.links =   [Link(self.trace), Link(self.trace)]
        self.senders = [Sender(rate=10 / (self.trace.get_delay(self.ts) * 2 / 1000),
                               path=[self.links[0], self.links[1]],
                               dest=0, **self.sender_kwargs)]
        [sender.register_network(self) for sender in self.senders]
        self.event_queue = EventQueue()
        self.counter_event = counter()

        for sender in self.senders:
            self.event_queue.push(Event(ts=0, id=next(self.counter_event), 
                                        src=sender, dst=0, type=EVENT_TYPE_SEND, next_hop=0, latency=0.0, 
                                        dropped=False, rto=sender.rto, event_queue_delay=0.0))

    def run(self, duration):
        ts_str = self.ts
        ts_end = min(self.ts + duration, self.trace.ts_end)
        
        [sender.reset_obs() for sender in self.senders]

        while self.event_queue.top().ts < ts_end:
            event = self.event_queue.pop()
            # print(event)
            self.ts = event.ts
            # breakpoint()
            if event.type == EVENT_TYPE_ACK:
                if event.next_hop == len(event.src.path):
                    if event.rto >= 0 and event.latency > event.rto:
                        # Timeout
                        event.src.timeout()
                    elif event.dropped:
                        # Drop
                        event.src.on_pkt_lost(event.latency)
                    else:
                        #
                        event.src.on_pkt_acked(event.latency)
                else:
                    event=self.hop(event)
                    self.event_queue.push(event)

            elif event.type == EVENT_TYPE_SEND:
                if event.next_hop == event.dst:
                    # Arrive
                    if event.src.can_send_pkt():
                        event.src.send_pkt()
                    self.event_queue.push(Event(ts=self.ts + 1.0 / event.src.rate, id=next(self.counter_event), 
                                                src=event.src, dst=0, type=EVENT_TYPE_SEND, next_hop=0, latency=0.0, 
                                                dropped=False, rto=event.src.rto, event_queue_delay=0.0))

                if event.next_hop == event.dst:
                    event.type = EVENT_TYPE_ACK

                event=self.hop(event)
                event.dropped = not event.src.path[event.next_hop].on_pkt_enter(self.ts)
                self.event_queue.push(event)

        for sender in self.senders:
            sender.record_run()

        sender_mi = self.senders[0].history.back()
        throughput = sender_mi.get("recv rate")  # bits/sec
        latency = sender_mi.get("avg latency")  # second
        loss = sender_mi.get("loss ratio")

        reward = pcc_aurora_reward(throughput / BITS_PER_BYTE / BYTES_PER_PACKET, latency, loss)
        self.ts = ts_end
        return reward
    
    def hop(self, event):
        latency_propagation = event.src.path[event.next_hop].get_latency_propagation(self.ts)
        event.next_hop += 1
        event.latency += latency_propagation
        event.ts += latency_propagation
        return event


class Networker():
    def __init__(self, senders, links, env):
        self.event_count = 0
        self.senders = senders
        self.links = links
        self.env = env
        self.reset()

    def reset(self):
        self.pkt_log = []
        self.curr_timestamp = 0.0
        self.event_queue = []
        [link.reset() for link in self.links]
        # [sender.reset() for sender in self.senders]

        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.event_queue, (0, sender, EVENT_TYPE_SEND,
                                              0, 0.0, False, self.event_count, sender.rto, 0.0))
            self.event_count += 1

    def get_curr_time(self):
        return self.curr_timestamp

    def run_for_duration(self, duration, action=None):
        if self.senders[0].lat_diff != 0:
            self.senders[0].start_stage = False
        stime = self.curr_timestamp
        etime = min(self.curr_timestamp + duration, self.env.current_trace.timestamps[-1])
        # debug_print('MI from {} to {}, dur {}'.format(
        #     self.cur_time, end_time, dur))
        for sender in self.senders:
            sender.reset_obs()
        # set_obs_start = False
        extra_delays = []  # time used to put packet onto the network
        while True:
            event_time, sender, event_type, next_hop, cur_latency, dropped, \
                event_id, rto, event_queue_delay = self.event_queue[0]
            # if not sender.got_data and event_time >= end_time and event_type == EVENT_TYPE_ACK and next_hop == len(sender.path):
            #     end_time = event_time
            #     self.cur_time = end_time
            #     self.env.run_dur = end_time - start_time
            #     break
            if sender.got_data and event_time >= etime and event_type == EVENT_TYPE_SEND:
                etime = event_time
                self.curr_timestamp = etime
                break

            event_time, sender, event_type, next_hop, cur_latency, dropped, \
                event_id, rto, event_queue_delay = heapq.heappop(self.event_queue)
            self.curr_timestamp = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            new_event_queue_delay = event_queue_delay
            push_new_event = False
            debug_print("Got %d event %s, to link %d, latency %f at time %f, "
                        "next_hop %d, dropped %s, event_q length %f, "
                        "sender rate %f, duration: %f, queue_size: %f, "
                        "rto: %f, cwnd: %f, ssthresh: %f, sender rto %f, "
                        "pkt in flight %d, wait time %d" % (
                            event_id, event_type, next_hop, cur_latency,
                            event_time, next_hop, dropped, len(self.event_queue),
                            sender.rate, duration, self.links[0].queue_size,
                            rto, sender.cwnd, sender.ssthresh, sender.rto,
                            int(sender.bytes_in_flight / BYTES_PER_PACKET),
                            sender.pkt_loss_wait_time))
            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    # if cur_latency > 1.0:
                    #     sender.timeout(cur_latency)
                    # sender.on_packet_lost(cur_latency)
                    if rto >= 0 and cur_latency > rto and sender.pkt_loss_wait_time <= 0:
                        sender.timeout()
                        dropped = True
                        new_dropped = True
                    elif dropped:
                        sender.on_packet_lost(cur_latency)
                        if self.env.record_pkt_log:
                            self.pkt_log.append(
                                [self.curr_timestamp, event_id, 'lost',
                                 BYTES_PER_PACKET, cur_latency, event_queue_delay,
                                 self.links[0].pkt_in_queue,
                                 sender.rate * BYTES_PER_PACKET * BITS_PER_BYTE,
                                 self.links[0].get_bandwidth(self.curr_timestamp) * BYTES_PER_PACKET * BITS_PER_BYTE])
                    else:
                        sender.on_packet_acked(cur_latency)
                        # debug_print('Ack packet at {}'.format(self.cur_time))
                        # log packet acked
                        if self.env.record_pkt_log:
                            self.pkt_log.append(
                                [self.curr_timestamp, event_id, 'acked',
                                 BYTES_PER_PACKET, cur_latency,
                                 event_queue_delay, self.links[0].pkt_in_queue,
                                 sender.rate * BYTES_PER_PACKET * BITS_PER_BYTE,
                                 self.links[0].get_bandwidth(self.curr_timestamp) * BYTES_PER_PACKET * BITS_PER_BYTE])
                else:
                    # comment out to save disk usage
                    # if self.env.record_pkt_log:
                    #     self.pkt_log.append(
                    #         [self.cur_time, event_id, 'arrived',
                    #          BYTES_PER_PACKET, cur_latency, event_queue_delay,
                    #          self.links[0].pkt_in_queue,
                    #          sender.rate * BYTES_PER_PACKET * BITS_PER_BYTE,
                    #          self.links[0].get_bandwidth(self.cur_time) * BYTES_PER_PACKET * BITS_PER_BYTE])
                    new_next_hop = next_hop + 1
                    # new_event_queue_delay += sender.path[next_hop].get_curr_queue_delay(
                    #     self.cur_time)
                    link_latency = sender.path[next_hop].get_curr_propagation_latency(
                        self.curr_timestamp)
                    # link_latency *= self.env.current_trace.get_delay_noise_replay(self.cur_time)
                    # if USE_LATENCY_NOISE:
                    # link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            elif event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    if sender.can_send_packet():
                        sender.on_packet_sent()
                        # print('Send packet at {}'.format(self.cur_time))
                        if not self.env.train_flag and self.env.record_pkt_log:
                            self.pkt_log.append(
                                [self.curr_timestamp, event_id, 'sent',
                                 BYTES_PER_PACKET, cur_latency,
                                 event_queue_delay, self.links[0].pkt_in_queue,
                                 sender.rate * BYTES_PER_PACKET * BITS_PER_BYTE,
                                 self.links[0].get_bandwidth(self.curr_timestamp) * BYTES_PER_PACKET * BITS_PER_BYTE])
                        push_new_event = True
                    heapq.heappush(self.event_queue, (self.curr_timestamp + (1.0 / sender.rate),
                                                      sender, EVENT_TYPE_SEND, 0, 0.0,
                                                      False, self.event_count, sender.rto,
                                                      0.0))
                    self.event_count += 1

                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1

                prop_delay, new_event_queue_delay = sender.path[next_hop].get_curr_latency(
                    self.curr_timestamp)
                link_latency = prop_delay + new_event_queue_delay
                # if USE_LATENCY_NOISE:
                # link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                # link_latency += self.env.current_trace.get_delay_noise(
                #     self.cur_time, self.links[0].get_bandwidth(self.cur_time)) / 1000
                # link_latency += max(0, np.random.normal(0, 1) / 1000)
                # link_latency += max(0, np.random.uniform(0, 5) / 1000)
                rand = random.uniform(0, 1)
                if rand > 0.9:
                    noise = random.uniform(0, sender.path[next_hop].trace.delay_noise) / 1000
                else:
                    noise = 0
                new_latency += noise
                new_event_time += noise
                # link_latency *= self.env.current_trace.get_delay_noise_replay(self.cur_time)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].on_pkt_enter_link(
                    self.curr_timestamp)
                extra_delays.append(
                    1 / self.links[0].get_bandwidth(self.curr_timestamp))
                # new_latency += 1 / self.links[0].get_bandwidth(self.cur_time)
                # new_event_time += 1 / self.links[0].get_bandwidth(self.cur_time)
                if not new_dropped:
                    sender.queue_delay_samples.append(new_event_queue_delay)

            if push_new_event:
                heapq.heappush(self.event_queue, (new_event_time, sender, new_event_type,
                                                  new_next_hop, new_latency, new_dropped,
                                                  event_id, rto, float(new_event_queue_delay)))
        for sender in self.senders:
            sender.record_run()

        sender_mi = self.senders[0].history.back()  # get_run_data()
        throughput = sender_mi.get("recv rate")  # bits/sec
        latency = sender_mi.get("avg latency")  # second
        loss = sender_mi.get("loss ratio")
        # debug_print("thpt %f, delay %f, loss %f, bytes sent %f, bytes acked %f" % (
        #     throughput/1e6, latency, loss, sender_mi.bytes_sent, sender_mi.bytes_acked))
        avg_bw_in_mi = self.env.current_trace.get_avail_bits2send(stime, etime) / (etime - stime) / BITS_PER_BYTE / BYTES_PER_PACKET
        # avg_bw_in_mi = np.mean(self.env.current_trace.bandwidths) * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET
        reward = pcc_aurora_reward(
            throughput / BITS_PER_BYTE / BYTES_PER_PACKET, latency, loss,
            avg_bw_in_mi, np.mean(self.env.current_trace.delays) * 2 / 1e3)

        # self.env.run_dur = MI_RTT_PROPORTION * self.senders[0].estRTT # + np.mean(extra_delays)
        if latency > 0.0:
            self.env.run_dur = MI_RTT_PROPORTION * \
                sender_mi.get("avg latency") + np.mean(np.array(extra_delays))
        # elif self.env.run_dur != 0.01:
            # assert self.env.run_dur >= 0.03
            # self.env.run_dur = max(MI_RTT_PROPORTION * sender_mi.get("avg latency"), 5 * (1 / self.senders[0].rate))

        # self.senders[0].avg_latency = sender_mi.get("avg latency")  # second
        # self.senders[0].recv_rate = round(sender_mi.get("recv rate"), 3)  # bits/sec
        # self.senders[0].send_rate = round(sender_mi.get("send rate"), 3)  # bits/sec
        # self.senders[0].lat_diff = sender_mi.rtt_samples[-1] - sender_mi.rtt_samples[0]
        # self.senders[0].latest_rtt = sender_mi.rtt_samples[-1]
        # self.recv_rate_cache.append(self.senders[0].recv_rate)
        # if len(self.recv_rate_cache) > 6:
        #     self.recv_rate_cache = self.recv_rate_cache[1:]
        # self.senders[0].max_tput = max(self.recv_rate_cache)
        #
        # if self.senders[0].lat_diff == 0 and self.senders[0].start_stage:  # no latency change
        #     pass
        #     # self.senders[0].max_tput = max(self.senders[0].recv_rate, self.senders[0].max_tput)
        # elif self.senders[0].lat_diff == 0 and not self.senders[0].start_stage:  # no latency change
        #     pass
        #     # self.senders[0].max_tput = max(self.senders[0].recv_rate, self.senders[0].max_tput)
        # elif self.senders[0].lat_diff > 0:  # latency increase
        #     self.senders[0].start_stage = False
        #     # self.senders[0].max_tput = self.senders[0].recv_rate # , self.max_tput)
        # else:  # latency decrease
        #     self.senders[0].start_stage = False
        #     # self.senders[0].max_tput = max(self.senders[0].recv_rate, self.senders[0].max_tput)
        return reward * REWARD_SCALE
