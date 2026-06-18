import pandas as pd
import sys

df = pd.read_csv("logs/runtime_eval.csv")

expected_attacks = [
    "normal",
    "ddos_flood",
    "flow_overflow",
    "packet_in_flood",
    "ip_spoofing",
    "port_scanning",
]

expected_configs = [
    "no_defense",
    "rule",
    "rl_dqn",
    "rl_ppo",
    "rl_guard_ppo",
    "rl_twin_ppo",
    "full_system_ppo",
]

expected_phases = ["warmup", "attack", "recovery"]

print("\n=== Rows ===")
print(len(df))

print("\n=== attack_type ===")
print(df["attack_type"].value_counts())

print("\n=== eval_config ===")
print(df["eval_config"].value_counts())

print("\n=== phase ===")
print(df["phase"].value_counts())

print("\n=== Matrix attack × config × phase ===")
matrix = (
    df.groupby(["attack_type", "eval_config", "phase"])
    .size()
    .unstack(fill_value=0)
)

print(matrix.to_string())

errors = []

for attack in expected_attacks:
    if attack not in set(df["attack_type"]):
        errors.append(f"Missing attack_type={attack}")

for config in expected_configs:
    if config not in set(df["eval_config"]):
        errors.append(f"Missing eval_config={config}")

for phase in expected_phases:
    if phase not in set(df["phase"]):
        errors.append(f"Missing phase={phase}")

for attack in expected_attacks:
    for config in expected_configs:
        sub = df[(df["attack_type"] == attack) & (df["eval_config"] == config)]
        if sub.empty:
            errors.append(f"Missing rows for attack={attack}, config={config}")
            continue

        for phase in expected_phases:
            if phase not in set(sub["phase"]):
                errors.append(f"Missing phase={phase} for attack={attack}, config={config}")

if errors:
    print("\n[FAILED]")
    for e in errors:
        print("-", e)
    sys.exit(1)

print("\n[OK] Full benchmark matrix exists.")