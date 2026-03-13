import numpy as np

class StateBuilder:
    def __init__(self):
        self.prev_packets = None
        self.prev_bytes = None
        self.prev_action = 0

    def build(self, raw):
        packets = raw["total_packets"]
        bytes_ = raw["total_bytes"]
        latency = raw["latency"]
        flow_count = raw["flow_count"]

        if self.prev_packets is None:
            packet_rate = 0
            byte_rate = 0
        else:
            packet_rate = packets - self.prev_packets
            byte_rate = bytes_ - self.prev_bytes

        self.prev_packets = packets
        self.prev_bytes = bytes_

        state = np.array([
            packet_rate / 10000,
            byte_rate / 100000,
            flow_count / 100,
            latency / 100,
            raw.get("controller_load", 0),
            raw.get("attack_flag", 0),
            self.prev_action
        ], dtype=np.float32)

        return state