from mininet.topo import Topo

class SDNResearchTopo(Topo):

    def build(self):

        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        # 6 attackers
        for i in range(1, 7):
            h = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            self.addLink(h, s1)

        # normal user
        h7 = self.addHost('h7', ip='10.0.0.7/24')
        self.addLink(h7, s1)

        # victim
        victim = self.addHost('h8', ip='10.0.0.8/24')
        self.addLink(victim, s2)

        self.addLink(s1, s2)
