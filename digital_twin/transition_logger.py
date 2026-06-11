import csv
import os

class TransitionLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.file_path):
            header = [
                "packet_rate", "byte_rate", "flow_count", "flow_growth_rate", "src_ip_entropy", 
                "latency", "packet_loss", "controller_cpu",
                "action", "next_latency", "next_packet_loss", "attack_type"
            ]
            with open(self.file_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def log(self, state, action, next_state, attack_type):
        FIELDS = ["packet_rate","byte_rate","flow_count","flow_growth_rate",
              "src_ip_entropy","latency","packet_loss","controller_cpu"]
        row = [state.get(f, 0.0) for f in FIELDS] + \
              [action] + \
              [next_state.get("latency", 0.0), next_state.get("packet_loss", 0.0)] + \
              [attack_type]
        with open(self.file_path, "a", newline="") as f:
            csv.writer(f).writerow(row)