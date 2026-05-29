#!/usr/bin/env python3

from mininet.topo import Topo


class SDNSecurityTopo(Topo):
    """
    h1-h4: normal users
    h5-h7: attackers
    h8   : victim server
    h9   : honeypot server

    Port mapping:
    s1-eth1 -> h1
    s1-eth2 -> h2
    s1-eth3 -> h3
    s1-eth4 -> h4
    s1-eth5 -> h5
    s1-eth6 -> h6
    s1-eth7 -> h7
    s1-eth8 -> s2 -> h8 victim
    s1-eth9 -> s3 -> h9 honeypot
    """

    def build(self):
        s1 = self.addSwitch("s1", dpid="0000000000000001")
        s2 = self.addSwitch("s2", dpid="0000000000000002")
        s3 = self.addSwitch("s3", dpid="0000000000000003")

        for i in range(1, 5):
            self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24",
                mac=f"00:00:00:00:00:0{i}"
            )

        for i in range(5, 8):
            self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24",
                mac=f"00:00:00:00:00:0{i}"
            )

        self.addHost("h8", ip="10.0.0.8/24", mac="00:00:00:00:00:08")
        self.addHost("h9", ip="10.0.0.9/24", mac="00:00:00:00:00:09")

        self.addLink("h1", s1, port2=1)
        self.addLink("h2", s1, port2=2)
        self.addLink("h3", s1, port2=3)
        self.addLink("h4", s1, port2=4)

        self.addLink("h5", s1, port2=5)
        self.addLink("h6", s1, port2=6)
        self.addLink("h7", s1, port2=7)

        self.addLink(s1, s2, port1=8, port2=1)
        self.addLink(s1, s3, port1=9, port2=1)

        self.addLink("h8", s2, port2=2)
        self.addLink("h9", s3, port2=2)