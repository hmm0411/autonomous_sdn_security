from mininet.topo import Topo

class SDNResearchTopo(Topo):
    def build(self):
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3') # Switch Honeypot

        # 6 Attackers kết nối vào s1
        for i in range(1, 7):
            h = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            self.addLink(h, s1)

        # Normal user h7
        h7 = self.addHost('h7', ip='10.0.0.7/24')
        self.addLink(h7, s1)

        # Victim h8 kết nối vào s2
        h8 = self.addHost('h8', ip='10.0.0.8/24')
        self.addLink(h8, s2)

        # Honeypot h9 kết nối vào s3
        h9 = self.addHost('h9', ip='10.0.0.9/24')
        self.addLink(h9, s3)

        # KẾT NỐI KHÔNG VÒNG LẶP:
        self.addLink(s1, s2) # Đường truyền chính (Attacker -> Victim)
        self.addLink(s1, s3) # Đường truyền phụ (Để RL học Action Redirect sang Honeypot)
        # KHÔNG nối s3 với s2 nữa để tránh Loop.