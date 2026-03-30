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
        h6 = self.addHost('h6', ip='10.0.0.6/24')
        self.addLink(h6, s1)

        # victim
        victim = self.addHost('h7', ip='10.0.0.7/24')
        self.addLink(victim, s2)

        self.addLink(s1, s2)
