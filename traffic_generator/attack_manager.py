#!/usr/bin/env python3

import random
import time


class AttackManager:
    def __init__(self, net):
        self.net = net

        self.normal_users = [net.get(f"h{i}") for i in range(1, 5)]
        self.attackers = [net.get(f"h{i}") for i in range(5, 8)]

        self.victim = net.get("h8")
        self.honeypot = net.get("h9")

        self.active_pids = []

    # ==================================================
    # INTERNAL UTILS
    # ==================================================
    def _run_bg(self, host, cmd):
        """
        Run command in background and store PID.
        """
        full_cmd = f"{cmd} > /tmp/{host.name}_{len(self.active_pids)}.log 2>&1 & echo $!"
        pid = host.cmd(full_cmd).strip()

        if pid:
            self.active_pids.append((host, pid))

        return pid

    def stop_all(self):
        """
        Stop all generated traffic.
        """
        print("[*] Stopping all generated traffic...")

        for host, pid in self.active_pids:
            host.cmd(f"kill -9 {pid} 2>/dev/null || true")

        self.active_pids = []

        # Kill remaining tools just in case
        for h in self.normal_users + self.attackers + [self.victim, self.honeypot]:
            h.cmd("pkill -9 hping3 2>/dev/null || true")
            h.cmd("pkill -9 iperf 2>/dev/null || true")
            h.cmd("pkill -9 nmap 2>/dev/null || true")
            h.cmd("pkill -9 python3 2>/dev/null || true")

        print("[+] All traffic stopped.")

    def start_servers(self):
        self.victim.cmd("pkill -9 iperf 2>/dev/null || true")
        self.honeypot.cmd("pkill -9 iperf 2>/dev/null || true")

        self.victim.cmd("iperf -s -p 5001 > /tmp/h8_victim_iperf.log 2>&1 &")
        self.honeypot.cmd("iperf -s -p 5001 > /tmp/h9_honeypot_iperf.log 2>&1 &")

        print("[+] Victim h8 service started at 10.0.0.8:5001")
        print("[+] Honeypot h9 service started at 10.0.0.9:5001")


    def normal_low(self):
        self._set_state("Normal Low")
        print("[*] Normal LOW: h1-h2 ping -> h8")
        for h in self.normal_users[:2]:
            pid = h.cmd(f"ping {self.victim.IP()} > /tmp/{h.name}_normal_low.log 2>&1 & echo $!").strip()
            self.active_pids.append((h, pid))


    def normal_medium(self):
        self._set_state("Normal Medium")
        print("[*] Normal MEDIUM: h1-h3 iperf -> h8")
        for h in self.normal_users[:3]:
            pid = h.cmd(f"iperf -c {self.victim.IP()} -p 5001 -t 600 -i 1 > /tmp/{h.name}_normal_medium.log 2>&1 & echo $!").strip()
            self.active_pids.append((h, pid))


    def normal_high(self):
        self._set_state("Normal High")
        print("[*] Normal HIGH: h1-h4 iperf -> h8")
        for h in self.normal_users:
            pid = h.cmd(f"iperf -c {self.victim.IP()} -p 5001 -t 600 -i 1 > /tmp/{h.name}_normal_high.log 2>&1 & echo $!").strip()
            self.active_pids.append((h, pid))

    # ==================================================
    # ATTACK TRAFFIC
    # ==================================================
    def ddos_flood(self, num_attackers=1, intensity="low"):
        self._set_state("DDoS Flood")

        selected = random.sample(self.attackers, min(num_attackers, len(self.attackers)))

        if intensity == "low":
            mode = "--fast"
            processes = 1
        elif intensity == "medium":
            mode = "--faster"
            processes = 2
        else:
            mode = "--flood"
            processes = 2

        print(f"[*] DDoS {intensity}: {len(selected)} attacker(s) -> h8")

        for attacker in selected:
            for _ in range(processes):
                cmd = f"hping3 -S {mode} -p 80 {self.victim.IP()} > /tmp/{attacker.name}_ddos.log 2>&1 & echo $!"
                pid = attacker.cmd(cmd).strip()
                self.active_pids.append((attacker, pid))

    def packet_in_flood(self, num_attackers=1, intensity="medium"):
        self._set_state("Packet-In Flood")
        selected = random.sample(self.attackers, min(num_attackers, len(self.attackers)))

        mode = "--fast" if intensity == "low" else "--faster"

        for attacker in selected:
            cmd = f"hping3 --rand-source -S {mode} -p 12345 {self.victim.IP()} > /tmp/{attacker.name}_packetin.log 2>&1 & echo $!"
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))


    def ip_spoofing(self, num_attackers=1, intensity="medium"):
        self._set_state("IP Spoofing")
        selected = random.sample(self.attackers, min(num_attackers, len(self.attackers)))

        mode = "--fast" if intensity == "low" else "--faster"

        for attacker in selected:
            cmd = f"hping3 --rand-source -S {mode} -p 80 {self.victim.IP()} > /tmp/{attacker.name}_spoof.log 2>&1 & echo $!"
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))


    def flow_overflow(self, num_attackers=1, flows_per_attacker=2000):
        self._set_state("Flow Table Overflow")
        selected = random.sample(self.attackers, min(num_attackers, len(self.attackers)))

        for attacker in selected:
            cmd = (
                f"for p in $(seq 10000 {10000 + flows_per_attacker}); do "
                f"hping3 -S -c 1 -p $p {self.victim.IP()} >/dev/null 2>&1; "
                f"done > /tmp/{attacker.name}_flow_overflow.log 2>&1 & echo $!"
            )
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))


    def port_scanning(self, attacker_index=0, start_port=1, end_port=1000):
        self._set_state("Port Scanning")
        attacker = self.attackers[attacker_index % len(self.attackers)]

        cmd = (
            f"for p in $(seq {start_port} {end_port}); do "
            f"hping3 -S -c 1 -p $p {self.victim.IP()} >/dev/null 2>&1; "
            f"done > /tmp/{attacker.name}_port_scan.log 2>&1 & echo $!"
        )
        pid = attacker.cmd(cmd).strip()
        self.active_pids.append((attacker, pid))

    def stop_all(self):
        self._set_state("Normal Traffic")
        print("[*] Stopping all generated traffic...")

        for host, pid in self.active_pids:
            host.cmd(f"kill -9 {pid} 2>/dev/null || true")

        self.active_pids.clear()

        for host in self.normal_users + self.attackers + [self.victim, self.honeypot]:
            host.cmd("pkill -9 hping3 2>/dev/null || true")
            host.cmd("pkill -9 iperf 2>/dev/null || true")
            host.cmd("pkill -9 nmap 2>/dev/null || true")
            host.cmd("pkill -9 ping 2>/dev/null || true")
            host.cmd("pkill -9 python3 2>/dev/null || true")

        print("[+] All traffic stopped.")