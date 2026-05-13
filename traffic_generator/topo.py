#!/usr/bin/env python3

from mininet.topo import Topo


class SDNSecurityTopo(Topo):
    """
    Topology:
        h1-h4: normal users
        h5-h7: attackers
        h8   : victim server
        h9   : honeypot server

        h1-h7 --- s1 --- s2 --- h8 victim
                    \
                     s3 --- h9 honeypot
    """

    def build(self):
        # Switches
        s1 = self.addSwitch("s1", dpid="0000000000000001")
        s2 = self.addSwitch("s2", dpid="0000000000000002")
        s3 = self.addSwitch("s3", dpid="0000000000000003")

        # Normal users: h1-h4
        for i in range(1, 5):
            self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24",
                mac=f"00:00:00:00:00:0{i}"
            )

        # Attackers: h5-h7
        for i in range(5, 8):
            self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24",
                mac=f"00:00:00:00:00:0{i}"
            )

        # Victim and honeypot
        self.addHost(
            "h8",
            ip="10.0.0.8/24",
            mac="00:00:00:00:00:08"
        )

        self.addHost(
            "h9",
            ip="10.0.0.9/24",
            mac="00:00:00:00:00:09"
        )

        # Attach normal users to s1
        self.addLink("h1", s1, port2=1)
        self.addLink("h2", s1, port2=2)
        self.addLink("h3", s1, port2=3)
        self.addLink("h4", s1, port2=4)

        # Attach attackers to s1
        self.addLink("h5", s1, port2=5)
        self.addLink("h6", s1, port2=6)
        self.addLink("h7", s1, port2=7)

        # s1 -> s2 victim path
        # s1 port 8 goes to s2
        self.addLink(s1, s2, port1=8, port2=1)

        # s1 -> s3 honeypot path
        # s1 port 9 goes to s3
        self.addLink(s1, s3, port1=9, port2=1)

        # victim server on s2
        self.addLink("h8", s2, port2=2)

        # honeypot server on s3
        self.addLink("h9", s3, port2=2)