import argparse
import os
import time
from datetime import datetime

SIGNAL_FILE = "logs/current_attack.txt"

SCENARIOS = {
    "normal": {
        "signal": "Normal Traffic",
        "duration": 20,
        "expected_action": 0,
        "description": "Baseline normal traffic"
    },
    "ddos": {
        "signal": "DDoS Flood Attack",
        "duration": 35,
        "expected_action": 1,
        "description": "DDoS scenario, expected action: Block"
    },
    "spoof": {
        "signal": "IP Spoofing Detected",
        "duration": 30,
        "expected_action": 4,
        "description": "IP spoofing scenario, expected action: Isolate"
    },
    "packet_in": {
        "signal": "Packet-In Anomaly",
        "duration": 30,
        "expected_action": 2,
        "description": "Packet-In flood scenario, expected action: Rate Limit"
    },
    "recovery": {
        "signal": "Normal Traffic",
        "duration": 25,
        "expected_action": 0,
        "description": "Recovery phase, expected action: No Action"
    }
}


def ensure_dirs():
    os.makedirs("logs", exist_ok=True)


def write_signal(signal: str):
    with open(SIGNAL_FILE, "w") as f:
        f.write(signal)


def run_phase(name: str, signal: str, duration: int, expected_action: int, description: str):
    write_signal(signal)

    print("=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SCENARIO: {name.upper()}")
    print(f"Signal written to {SIGNAL_FILE}: {signal}")
    print(f"Description: {description}")
    print(f"Expected action id: {expected_action}")
    print(f"Duration: {duration}s")
    print("=" * 80)

    for remaining in range(duration, 0, -5):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running {name}... remaining {remaining}s")
        time.sleep(min(5, remaining))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished scenario: {name}")


def run_all():
    ensure_dirs()

    order = ["normal", "ddos", "recovery", "spoof", "recovery", "packet_in", "recovery"]

    for name in order:
        cfg = SCENARIOS[name]
        run_phase(
            name=name,
            signal=cfg["signal"],
            duration=cfg["duration"],
            expected_action=cfg["expected_action"],
            description=cfg["description"]
        )

    write_signal("Normal Traffic")
    print("\nAll validation scenarios completed. System returned to Normal Traffic.")


def run_one(name: str):
    ensure_dirs()

    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")

    cfg = SCENARIOS[name]
    run_phase(
        name=name,
        signal=cfg["signal"],
        duration=cfg["duration"],
        expected_action=cfg["expected_action"],
        description=cfg["description"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all"] + list(SCENARIOS.keys()),
        help="Scenario to run"
    )

    args = parser.parse_args()

    if args.scenario == "all":
        run_all()
    else:
        run_one(args.scenario)