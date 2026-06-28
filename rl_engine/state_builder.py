import numpy as np


FEATURES_8 = [
    "packet_rate",
    "byte_rate",
    "flow_count",
    "flow_growth_rate",
    "src_ip_entropy",
    "latency",
    "packet_loss",
    "controller_cpu",
]


MAX_VALUES = {
    "packet_rate": 5000.0,
    "byte_rate": 5_000_000.0,
    "flow_count": 500.0,
    "flow_growth_rate": 100.0,
    "src_ip_entropy": 10.0,
    "latency": 300.0,
    "packet_loss": 1.0,
    "controller_cpu": 100.0,
}


def _clip01(value: float) -> float:
    if value != value:
        return 0.0

    return float(max(0.0, min(1.0, value)))


class StateBuilder:
    """
    Build state vector 8 chiều:

    0 packet_rate
    1 byte_rate
    2 flow_count
    3 flow_growth_rate
    4 src_ip_entropy
    5 latency
    6 packet_loss
    7 controller_cpu
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def build(self, raw: dict) -> np.ndarray:
        if raw is None:
            return np.zeros(8, dtype=np.float32)

        values = []

        for feature in FEATURES_8:
            value = float(raw.get(feature, 0.0) or 0.0)

            if self.normalize:
                value = _clip01(value / MAX_VALUES[feature])

            values.append(value)

        return np.array(values, dtype=np.float32)