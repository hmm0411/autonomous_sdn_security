import argparse
import os
import time
import urllib.request
from datetime import datetime

SIGNAL_FILE = "logs/current_attack.txt"
METRICS_URL = "http://localhost:8800/metrics"

METRICS_KEYWORDS = [
    "sdn_current_score",
    "sdn_threat_level",
    "sdn_rl_action",
    "sdn_packet_rate",
    "sdn_flow_count",
]

SCENARIOS = {
    "normal": {
        "signal": "Normal Traffic",
        "duration": 20,
        "expected_status": "SAFE",
        "expected_action": "NO ACTION",
        "expected_action_id": 0,
        "description": "Baseline normal traffic validation",
    },
    "ddos": {
        "signal": "DDoS Flood Attack",
        "duration": 40,
        "expected_status": "CRITICAL",
        "expected_action": "BLOCK",
        "expected_action_id": 1,
        "description": "DDoS flood scenario, expected defense action is BLOCK",
    },
    "spoof": {
        "signal": "IP Spoofing Detected",
        "duration": 35,
        "expected_status": "CRITICAL",
        "expected_action": "ISOLATE",
        "expected_action_id": 4,
        "description": "IP spoofing scenario, expected defense action is ISOLATE",
    },
    "packet_in": {
        "signal": "Packet-In Anomaly",
        "duration": 35,
        "expected_status": "WARNING/CRITICAL",
        "expected_action": "RATE LIMIT",
        "expected_action_id": 2,
        "description": "Packet-In anomaly scenario, expected defense action is RATE LIMIT",
    },
    "flow_overflow": {
        "signal": "Flow Table Overflow",
        "duration": 35,
        "expected_status": "CRITICAL",
        "expected_action": "RATE LIMIT",
        "expected_action_id": 2,
        "description": "Flow table overflow scenario, expected defense action is RATE LIMIT",
    },
    "port_scan": {
        "signal": "Port Scanning",
        "duration": 35,
        "expected_status": "WARNING",
        "expected_action": "REDIRECT",
        "expected_action_id": 3,
        "description": "Port scanning scenario, expected defense action is REDIRECT",
    },
    "recovery": {
        "signal": "Normal Traffic",
        "duration": 25,
        "expected_status": "SAFE",
        "expected_action": "NO ACTION",
        "expected_action_id": 0,
        "description": "Recovery phase, system should return to SAFE state",
    },
}


def now():
    return datetime.now().strftime("%H:%M:%S")


def ensure_log_dir():
    os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)


def write_attack_signal(signal: str):
    ensure_log_dir()
    with open(SIGNAL_FILE, "w", encoding="utf-8") as f:
        f.write(signal)


def read_metrics():
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=3) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return [f"[WARN] Cannot read metrics from {METRICS_URL}: {e}"]

    lines = []
    for line in body.splitlines():
        if any(key in line for key in METRICS_KEYWORDS):
            if not line.startswith("#"):
                lines.append(line)

    return lines if lines else ["[WARN] No SDN metrics found yet"]


def print_header(name: str, cfg: dict):
    print("\n" + "=" * 90)
    print(f"[{now()}] START SCENARIO: {name.upper()}")
    print("=" * 90)
    print(f"Signal file       : {SIGNAL_FILE}")
    print(f"Generated signal  : {cfg['signal']}")
    print(f"Description       : {cfg['description']}")
    print(f"Expected status   : {cfg['expected_status']}")
    print(f"Expected action   : {cfg['expected_action']} ({cfg['expected_action_id']})")
    print(f"Duration          : {cfg['duration']}s")
    print("=" * 90)


def run_phase(name: str, duration_override=None, watch_metrics=True):
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")

    cfg = SCENARIOS[name]
    duration = duration_override or cfg["duration"]

    print_header(name, {**cfg, "duration": duration})
    write_attack_signal(cfg["signal"])

    elapsed = 0
    interval = 5

    while elapsed < duration:
        remaining = duration - elapsed
        print(f"[{now()}] Running {name:<12} | remaining: {remaining:>3}s")

        if watch_metrics:
            for metric_line in read_metrics():
                print(f"  metric: {metric_line}")

        time.sleep(min(interval, remaining))
        elapsed += interval

    print(f"[{now()}] FINISHED SCENARIO: {name.upper()}")


def run_all(watch_metrics=True):
    sequence = [
        "normal",
        "ddos",
        "recovery",
        "spoof",
        "recovery",
        "packet_in",
        "recovery",
    ]

    print("\nSDN Realtime Validation Scenario Runner")
    print("This script generates controlled attack signals for Grafana/Prometheus validation.")
    print("Open Grafana dashboard and Prometheus Alerts before running this scenario.\n")

    for scenario in sequence:
        run_phase(scenario, watch_metrics=watch_metrics)

    write_attack_signal("Normal Traffic")
    print("\n" + "=" * 90)
    print(f"[{now()}] ALL SCENARIOS COMPLETED. System returned to Normal Traffic.")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Generate controlled SDN security validation scenarios"
    )

    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all"] + list(SCENARIOS.keys()),
        help="Scenario to run",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Override duration for a single scenario",
    )

    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Do not print metrics while running",
    )

    args = parser.parse_args()
    watch_metrics = not args.no_metrics

    if args.scenario == "all":
        run_all(watch_metrics=watch_metrics)
    else:
        run_phase(
            args.scenario,
            duration_override=args.duration,
            watch_metrics=watch_metrics,
        )


if __name__ == "__main__":
    main()