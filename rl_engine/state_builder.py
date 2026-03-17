import numpy as np

class StateBuilder:
    def __init__(self):
        self.prev_packets = None
        self.prev_bytes = None
        self.prev_action = 0

    def build(self, raw):
        packets = raw.get("total_packets", 0)
        bytes_ = raw.get("total_bytes", 0)
        latency = raw.get("latency", 0)
        flow_count = raw.get("flow_count", 0)
        packet_loss = raw.get("packet_loss", 0)
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
            packet_loss,
            raw.get("controller_load", 0),
            raw.get("attack_flag", 0),
            self.prev_action
        ], dtype=np.float32)

        return state