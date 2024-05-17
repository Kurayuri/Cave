from ...CONST import BITS_PER_BYTE, BYTES_PER_PACKET
from .Trace import Trace
from typing import Tuple
import random


class Link():
    def __init__(self, trace: Trace):
        self.trace = trace
        self.reset()
    
    def reset(self):
        self.trace.reset()
        self.queue_size = self.trace.queue_size
        self.queue_delay_update_time = 0.0
        self.pkt_in_queue = 0

    def get_curr_queue_delay(self, event_time: float) -> float:
        self.pkt_in_queue = max(
            0, self.pkt_in_queue - self.trace.get_avail_bits2send(
                self.queue_delay_update_time, event_time) / BITS_PER_BYTE / BYTES_PER_PACKET)
        self.queue_delay_update_time = event_time

        curr_queue_delay = self.trace.get_sending_t_usage(
            self.pkt_in_queue * BYTES_PER_PACKET * BITS_PER_BYTE, event_time)
        return curr_queue_delay

    def on_pkt_enter_link(self, event_time: float) -> bool:
        # Random loss
        if random.random() < self.trace.get_loss_rate():
            return False
        # Full queue
        if self.pkt_in_queue + 1 > self.queue_size:
            return False
        self.pkt_in_queue += 1
        return True

    def get_curr_propagation_latency(self, event_time: float) -> float:
        return self.trace.get_delay(event_time) / 1000.0

    def get_curr_latency(self, event_time: float) -> Tuple[float, float]:
        q_delay = self.get_curr_queue_delay(event_time)
        return self.trace.get_delay(event_time) / 1000.0, q_delay

    def get_bandwidth(self, timestamp):
        return self.trace.get_bandwidth(timestamp) * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET
