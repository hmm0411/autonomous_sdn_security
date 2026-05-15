#!/usr/bin/env python3

import os
import random


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIGNAL_FILE = os.path.join(PROJECT_ROOT, "logs", "current_attack.txt")


class AttackManager:
    def __init__(self, net):
        self.net = net

        self.normal_users = [net.get(f"h{i}") for i in range(1, 5)]
        self.attackers = [net.get(f"h{i}") for i in range(5, 8)]

        self.victim = net.get("h8")
        self.honeypot = net.get("h9")

        self.active_pids = []

        os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
        self._set_state("Normal Traffic")

    def _set_state(self, state_name):
        try:
            with open(SIGNAL_FILE, "w") as f:
                f.write(state_name)
        except Exception as e:
            print(f"[WARN] Cannot write state file: {e}")

    def _run_bg(self, host, cmd, log_name):
        log_file = f"/tmp/{host.name}_{log_name}.log"
        full_cmd = f"{cmd} > {log_file} 2>&1 & echo $!"
        pid = host.cmd(full_cmd).strip()

        print(f"    {host.name}: pid={pid}, log={log_file}")

        if pid:
            self.active_pids.append((host, pid))

        return pid

    def start_servers(self):
        print("[*] Starting victim and honeypot services...")

        self.victim.cmd("pkill -9 iperf 2>/dev/null || true")
        self.honeypot.cmd("pkill -9 iperf 2>/dev/null || true")

        self.victim.cmd("iperf -s -p 5001 > /tmp/h8_victim_iperf.log 2>&1 &")
        self.honeypot.cmd("iperf -s -p 5001 > /tmp/h9_honeypot_iperf.log 2>&1 &")

        print("[+] Victim service:   h8 10.0.0.8:5001")
        print("[+] Honeypot service: h9 10.0.0.9:5001")

    def stop_all(self):
        self._set_state("Normal Traffic")
        print("[*] Stopping all generated traffic...")

        for host, pid in self.active_pids:
            host.cmd(f"kill -9 {pid} 2>/dev/null || true")

        self.active_pids.clear()

        for host in self.normal_users + self.attackers + [self.victim, self.honeypot]:
            host.cmd("pkill -9 hping3 2>/dev/null || true")
            host.cmd("pkill -9 iperf 2>/dev/null || true")
            host.cmd("pkill -9 ping 2>/dev/null || true")
            host.cmd("pkill -9 nmap 2>/dev/null || true")
            host.cmd("pkill -9 python3 2>/dev/null || true")

        print("[+] All traffic stopped.")

    # =========================
    # NORMAL TRAFFIC
    # =========================
    def normal_low(self):
        self._set_state("Normal Low")
        print("[*] Normal LOW: h1-h2 ping -> h8")

        for h in self.normal_users[:2]:
            self._run_bg(
                h,
                f"ping {self.victim.IP()}",
                "normal_low"
            )

    def normal_medium(self):
        self._set_state("Normal Medium")
        print("[*] Normal MEDIUM: h1-h3 iperf -> h8:5001")

        for h in self.normal_users[:3]:
            self._run_bg(
                h,
                f"iperf -c {self.victim.IP()} -p 5001 -t 900 -i 1",
                "normal_medium"
            )

    def normal_high(self):
        self._set_state("Normal High")
        print("[*] Normal HIGH: h1-h4 iperf -> h8:5001")

        for h in self.normal_users:
            self._run_bg(
                h,
                f"iperf -c {self.victim.IP()} -p 5001 -t 900 -i 1",
                "normal_high"
            )

    # =========================
    # ATTACK TRAFFIC
    # =========================
    def ddos_flood(self, num_attackers=1, intensity="low"):
        self._set_state("DDoS Flood")

        selected = random.sample(
            self.attackers,
            min(num_attackers, len(self.attackers))
        )

        if intensity == "low":
            mode = "--fast"
        elif intensity == "medium":
            mode = "--faster"
        else:
            mode = "--flood"

        print(f"[*] DDoS {intensity}: {len(selected)} attacker(s) -> h8:80")

        for attacker in selected:
            self._run_bg(
                attacker,
                f"hping3 -S {mode} -p 80 {self.victim.IP()}",
                f"ddos_{intensity}"
            )

    def ip_spoofing(self, num_attackers=1, intensity="medium"):
        self._set_state("IP Spoofing")

        selected = random.sample(
            self.attackers,
            min(num_attackers, len(self.attackers))
        )

        if intensity == "low":
            mode = "--fast"
        elif intensity == "medium":
            mode = "--faster"
        else:
            mode = "--flood"

        print(f"[*] IP Spoofing {intensity}: {len(selected)} attacker(s) -> h8:80")

        for attacker in selected:
            self._run_bg(
                attacker,
                f"hping3 --rand-source -S {mode} -p 80 {self.victim.IP()}",
                f"spoof_{intensity}"
            )

    def packet_in_flood(self, num_attackers=1, intensity="medium"):
        self._set_state("Packet-In Flood")

        selected = random.sample(
            self.attackers,
            min(num_attackers, len(self.attackers))
        )

        if intensity == "low":
            mode = "--fast"
        elif intensity == "medium":
            mode = "--faster"
        else:
            mode = "--flood"

        print(f"[*] Packet-In Flood {intensity}: {len(selected)} attacker(s) -> random ports on h8")

        for attacker in selected:
            self._run_bg(
                attacker,
                f"hping3 --rand-source -S {mode} -p ++1 {self.victim.IP()}",
                f"packetin_{intensity}"
            )

    def flow_overflow(self, num_attackers=1, flows_per_attacker=3000):
        self._set_state("Flow Table Overflow")

        selected = random.sample(
            self.attackers,
            min(num_attackers, len(self.attackers))
        )

        print(f"[*] Flow Overflow: {len(selected)} attacker(s), {flows_per_attacker} flows each")

        for attacker in selected:
            cmd = (
                f"for p in $(seq 10000 {10000 + flows_per_attacker}); do "
                f"hping3 -S -c 1 -p $p {self.victim.IP()} >/dev/null 2>&1; "
                f"done"
            )

            self._run_bg(
                attacker,
                cmd,
                f"flow_overflow_{flows_per_attacker}"
            )

    def port_scanning(self, attacker_index=0, start_port=1, end_port=1000):
        self._set_state("Port Scanning")

        attacker = self.attackers[attacker_index % len(self.attackers)]

        print(f"[*] Port Scanning: {attacker.name} -> h8 ports {start_port}-{end_port}")

        cmd = (
            f"for p in $(seq {start_port} {end_port}); do "
            f"hping3 -S -c 1 -p $p {self.victim.IP()} >/dev/null 2>&1; "
            f"done"
        )

        self._run_bg(
            attacker,
            cmd,
            f"port_scan_{start_port}_{end_port}"
        )