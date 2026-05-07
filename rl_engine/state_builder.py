from curses import raw

import numpy as np

class StateBuilder:

    def __init__(self):
        self.prev_action = 0

    def build(self, raw):

        packet_rate = raw.get("packet_rate", 0)
        byte_rate = raw.get("byte_rate", 0)
        flow_count = raw.get("flow_count", 0)
        latency = raw.get("latency", 0)
        packet_loss = raw.get("packet_loss", 0)

        # Nếu chưa có entropy thật → 0
        entropy = raw.get("src_ip_entropy", 0)

        # Heuristic giả lập congestion
        queue_length = min(1.0, packet_rate / 100000)
        controller_cpu = min(1.0, flow_count / 100)

        def build(self, raw):

            state = np.array([
                raw.get("packet_rate", 0),
                raw.get("byte_rate", 0),
                raw.get("flow_count", 0),
                raw.get("src_ip_entropy", 0),
                raw.get("latency", 0),
                raw.get("packet_loss", 0),
                raw.get("queue_length", 0),
                raw.get("controller_cpu", 0),
                self.prev_action
            ], dtype=np.float32)

            return state