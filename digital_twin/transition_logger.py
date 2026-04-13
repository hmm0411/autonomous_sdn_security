import csv
import os

class TransitionLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.file_path):
            header = [
                "packet_rate", "byte_rate", "flow_count", "src_ip_entropy", 
                "latency", "packet_loss", "queue_length", "controller_cpu", "attack_indicator",
                "action", "next_latency", "next_packet_loss", "attack_type"
            ]
            with open(self.file_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def log(self, state, action, next_state, attack_type):
        # state có 9 phần tử
        # next_state cũng có 9 phần tử, ta lấy phần tử index 4 (latency) và 5 (packet_loss)
        row = list(state) + \
              [action] + \
              [next_state[4], next_state[5]] + \
              [attack_type]

        with open(self.file_path, "a", newline="") as f:
            csv.writer(f).writerow(row)