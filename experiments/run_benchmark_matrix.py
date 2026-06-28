import os
import time
import subprocess

NAMESPACE = os.getenv("NAMESPACE", "sdn-security")
DEPLOYMENT = os.getenv("CONTROL_LOOP_DEPLOYMENT", "control-loop")

WARMUP_SECONDS = int(os.getenv("WARMUP_SECONDS", "30"))
ATTACK_SECONDS = int(os.getenv("ATTACK_SECONDS", "90"))
RECOVERY_SECONDS = int(os.getenv("RECOVERY_SECONDS", "30"))

RUNS = int(os.getenv("RUNS", "1"))

ATTACKS = [
    "normal",
    "ddos_flood",
    "flow_overflow",
    "packet_in_flood",
    "ip_spoofing",
    "port_scanning",
]

INTENSITIES = ["medium"]

CONFIGS = [
    {
        "eval_config": "no_defense",
        "mode": "no_defense",
        "model": "dqn",
        "guard": "false",
        "twin": "false",
        "llm": "false",
    },
    {
        "eval_config": "rule",
        "mode": "rule",
        "model": "dqn",
        "guard": "false",
        "twin": "false",
        "llm": "false",
    },
    {
        "eval_config": "rl_dqn",
        "mode": "rl",
        "model": "dqn",
        "guard": "false",
        "twin": "false",
        "llm": "false",
    },
    {
        "eval_config": "rl_ppo",
        "mode": "rl",
        "model": "ppo",
        "guard": "false",
        "twin": "false",
        "llm": "false",
    },
    {
        "eval_config": "rl_guard_ppo",
        "mode": "rl",
        "model": "ppo",
        "guard": "true",
        "twin": "false",
        "llm": "false",
    },
    {
        "eval_config": "rl_twin_ppo",
        "mode": "rl_twin",
        "model": "ppo",
        "guard": "true",
        "twin": "true",
        "llm": "false",
    },
    {
        "eval_config": "full_system_ppo",
        "mode": "full",
        "model": "ppo",
        "guard": "true",
        "twin": "true",
        "llm": "true",
    },
]


ATTACK_COMMANDS = {
    "normal": None,
    "ddos_flood": "py net.manager.ddos_flood()",
    "flow_overflow": "py net.manager.flow_overflow()",
    "packet_in_flood": "py net.manager.packet_in_flood()",
    "ip_spoofing": "py net.manager.ip_spoofing()",
    "port_scanning": "py net.manager.port_scanning()",
}


def run(cmd: str):
    print("[CMD]", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def set_env(envs: dict):
    parts = []
    for k, v in envs.items():
        parts.append(f"{k}={v}")

    cmd = (
        f"kubectl set env deployment/{DEPLOYMENT} "
        f"-n {NAMESPACE} "
        + " ".join(parts)
    )
    run(cmd)


def restart_control_loop():
    run(f"kubectl rollout restart deployment/{DEPLOYMENT} -n {NAMESPACE}")
    run(f"kubectl rollout status deployment/{DEPLOYMENT} -n {NAMESPACE} --timeout=120s")


def mininet_send(command: str):
    """
    Cần bạn chạy Mininet trong tmux session tên 'mn':
    tmux new -s mn
    cd ~/autonomous_sdn_security/traffic_generator
    sudo python3 run.py

    Script này sẽ gửi lệnh vào tmux.
    """
    if not command:
        return

    safe_cmd = command.replace('"', '\\"')
    run(f'tmux send-keys -t mn "{safe_cmd}" Enter')


def stop_attack():
    mininet_send("py net.manager.stop_all()")


def run_one(attack, intensity, run_id, config):
    print(
        f"\n=== RUN attack={attack} intensity={intensity} "
        f"run={run_id} config={config['eval_config']} ===",
        flush=True,
    )

    common_env = {
        "EVAL_CONFIG": config["eval_config"],
        "MODE": config["mode"],
        "MODEL_TYPE": config["model"],
        "ENABLE_GUARD": config["guard"],
        "ENABLE_TWIN": config["twin"],
        "ENABLE_LLM": config["llm"],
        "ACTION_DRY_RUN": "true",
        "STRICT_RL": "true",
        "ALLOW_UNTRAINED_FALLBACK": "false",
        "ATTACK_TYPE": attack,
        "ATTACK_INTENSITY": intensity,
        "RUN_ID": run_id,
        "SURROGATE_MODEL": "models/surrogate_model.pkl",
    }

    # Warmup
    envs = dict(common_env)
    envs["PHASE"] = "warmup"
    set_env(envs)
    restart_control_loop()
    stop_attack()
    print(f"[PHASE] warmup {WARMUP_SECONDS}s")
    time.sleep(WARMUP_SECONDS)

    # Attack
    envs = dict(common_env)
    envs["PHASE"] = "attack"
    set_env(envs)
    restart_control_loop()

    attack_cmd = ATTACK_COMMANDS.get(attack)
    if attack_cmd:
        mininet_send(attack_cmd)

    print(f"[PHASE] attack {ATTACK_SECONDS}s")
    time.sleep(ATTACK_SECONDS)

    # Recovery
    envs = dict(common_env)
    envs["PHASE"] = "recovery"
    set_env(envs)
    restart_control_loop()
    stop_attack()

    print(f"[PHASE] recovery {RECOVERY_SECONDS}s")
    time.sleep(RECOVERY_SECONDS)


def main():
    for run_idx in range(1, RUNS + 1):
        for intensity in INTENSITIES:
            for attack in ATTACKS:
                for config in CONFIGS:
                    run_one(
                        attack=attack,
                        intensity=intensity,
                        run_id=str(run_idx),
                        config=config,
                    )

    print("\n[OK] Benchmark matrix finished.")


if __name__ == "__main__":
    main()