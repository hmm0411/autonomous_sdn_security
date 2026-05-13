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
        """
        Start victim and honeypot services.
        Victim h8 is the real protected service.
        Honeypot h9 is the decoy service.
        """
        print("[*] Starting victim and honeypot services...")

        self.victim.cmd("pkill -9 iperf 2>/dev/null || true")
        self.honeypot.cmd("pkill -9 iperf 2>/dev/null || true")

        self.victim.cmd("iperf -s -p 5001 > /tmp/h8_victim_iperf.log 2>&1 &")
        self.honeypot.cmd("iperf -s -p 5001 > /tmp/h9_honeypot_iperf.log 2>&1 &")

        print("[+] Victim service:   h8 10.0.0.8:5001")
        print("[+] Honeypot service: h9 10.0.0.9:5001")

    # ==================================================
    # NORMAL TRAFFIC
    # ==================================================
    def normal_low(self):
        """
        Light normal traffic from legitimate users to victim.
        """
        print("[*] Normal LOW traffic: h1-h2 -> h8")

        for h in self.normal_users[:2]:
            self._run_bg(h, f"ping -i 1 {self.victim.IP()}")

    def normal_medium(self):
        """
        Medium normal traffic.
        """
        print("[*] Normal MEDIUM traffic: h1-h3 -> h8")

        for h in self.normal_users[:3]:
            self._run_bg(h, f"iperf -c {self.victim.IP()} -p 5001 -t 300 -i 1")

    def normal_high(self):
        """
        High but legitimate traffic.
        This is important so the model does not learn:
        high traffic = attack.
        """
        print("[*] Normal HIGH traffic: h1-h4 -> h8")

        for h in self.normal_users:
            self._run_bg(h, f"iperf -c {self.victim.IP()} -p 5001 -t 300 -i 1")

    # ==================================================
    # ATTACK TRAFFIC
    # ==================================================
    def ddos_flood(self, num_attackers=1, intensity="low"):
        """
        SYN flood to victim h8.
        Attack does NOT go to honeypot directly.
        Controller should redirect suspicious traffic to h9.
        """
        num_attackers = min(num_attackers, len(self.attackers))
        selected = random.sample(self.attackers, num_attackers)

        if intensity == "low":
            mode = "--fast"
            processes = 1
        elif intensity == "medium":
            mode = "--faster"
            processes = 2
        else:
            mode = "--flood"
            processes = 2

        print(f"[*] DDoS {intensity}: {num_attackers} attacker(s) -> victim h8")

        for h in selected:
            for _ in range(processes):
                self._run_bg(h, f"hping3 -S {mode} -p 80 {self.victim.IP()}")

    def packet_in_flood(self, num_attackers=1, intensity="low"):
        """
        Generate many unusual packets so switch/controller sees many unknown flows.
        """
        num_attackers = min(num_attackers, len(self.attackers))
        selected = random.sample(self.attackers, num_attackers)

        if intensity == "low":
            mode = "--fast"
            processes = 1
        elif intensity == "medium":
            mode = "--faster"
            processes = 2
        else:
            mode = "--flood"
            processes = 2

        print(f"[*] Packet-In Flood {intensity}: {num_attackers} attacker(s) -> victim h8")

        for h in selected:
            for _ in range(processes):
                self._run_bg(h, f"hping3 --rand-source -S {mode} -p 12345 {self.victim.IP()}")

    def ip_spoofing(self, num_attackers=1, intensity="medium"):
        """
        Random source IP attack to victim.
        Useful for src_ip_entropy.
        """
        num_attackers = min(num_attackers, len(self.attackers))
        selected = random.sample(self.attackers, num_attackers)

        if intensity == "low":
            mode = "--fast"
        elif intensity == "medium":
            mode = "--faster"
        else:
            mode = "--flood"

        print(f"[*] IP Spoofing {intensity}: {num_attackers} attacker(s) -> victim h8")

        for h in selected:
            self._run_bg(h, f"hping3 --rand-source -S {mode} -p 80 {self.victim.IP()}")

    def port_scanning(self, attacker_index=0, start_port=1, end_port=1000):
        """
        Port scan from one attacker to victim.
        """
        attacker = self.attackers[attacker_index % len(self.attackers)]

        print(f"[*] Port scanning: {attacker.name} -> h8 ports {start_port}-{end_port}")

        cmd = (
            f"for p in $(seq {start_port} {end_port}); do "
            f"hping3 -S -c 1 -p $p {self.victim.IP()} >/dev/null 2>&1; "
            f"done"
        )

        self._run_bg(attacker, cmd)

    def flow_overflow(self, num_attackers=1, flows_per_attacker=2000):
        """
        Generate many different flows by changing destination ports.
        """
        num_attackers = min(num_attackers, len(self.attackers))
        selected = random.sample(self.attackers, num_attackers)

        print(f"[*] Flow Table Overflow: {num_attackers} attacker(s), {flows_per_attacker} flows each")

        for h in selected:
            cmd = (
                f"for p in $(seq 10000 {10000 + flows_per_attacker}); do "
                f"hping3 -S -c 1 -p $p {self.victim.IP()} >/dev/null 2>&1; "
                f"done"
            )
            self._run_bg(h, cmd)

    def mixed_ddos_with_normal(self):
        """
        Realistic scenario:
        normal users still access victim while attackers attack victim.
        """
        print("[*] Mixed scenario: normal users + DDoS attackers")

        self.normal_medium()
        time.sleep(2)
        self.ddos_flood(num_attackers=2, intensity="medium")