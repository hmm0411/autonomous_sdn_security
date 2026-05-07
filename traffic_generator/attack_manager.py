import random
import os

# Đường dẫn file tín hiệu để giao tiếp với Dashboard
SIGNAL_FILE = "logs/current_attack.txt"

class AttackManager:
    def __init__(self, net):
        self.net = net
        self.attackers = [net.get(f'h{i}') for i in range(1, 7)]
        self.victim = net.get('h8')
        self.active_pids = []
        
        # Tạo thư mục logs và set trạng thái Normal khi khởi động
        os.makedirs("logs", exist_ok=True)
        self._set_state("Normal Traffic")

    def _set_state(self, state_name):
        """Hàm ghi trạng thái tấn công ra file để Dashboard đọc"""
        try:
            with open(SIGNAL_FILE, "w") as f:
                f.write(state_name)
        except Exception as e:
            print(f"[!] Lỗi ghi file trạng thái: {e}")

    # ================================
    # DDoS (High Diversity)
    # ================================
    def ddos_flood(self):
        self._set_state("DDoS Flood") # <--- Báo tín hiệu
        print(f"[*] INTENSE DDoS: {len(self.attackers)} hosts -> {self.victim.IP()}")
        for attacker in self.attackers:
            for _ in range(10):
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
        self._set_state("Packet-In Flood") # <--- Báo tín hiệu
        attacker = random.choice(self.attackers)
        print(f"[*] INTENSE Packet-In Flood từ {attacker.name}")
        for _ in range(10):
            cmd = (
                f'python3 -c "from scapy.all import *;'
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
        self._set_state("Flow Table Overflow") # <--- Báo tín hiệu
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
        self._set_state("IP Spoofing") # <--- Báo tín hiệu
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
        self._set_state("Port Scanning") # <--- Báo tín hiệu
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
        self._set_state("Normal Traffic") # <--- Trả về Normal
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