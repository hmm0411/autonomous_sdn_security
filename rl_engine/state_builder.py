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

        state = np.array([
            packet_rate / 10000,     # giống offline
            byte_rate / 100000,
            flow_count / 100,
            entropy / 10,
            latency / 100,
            packet_loss,
            queue_length,
            controller_cpu,
            self.prev_action
        ], dtype=np.float32)

        return state