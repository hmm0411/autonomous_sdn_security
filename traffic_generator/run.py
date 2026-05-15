#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

from traffic_generator.topo import SDNSecurityTopo
from traffic_generator.attack_manager import AttackManager


def run():
    topo = SDNSecurityTopo()

    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=False,
        autoStaticArp=True
    )

    c0 = RemoteController(
        "c0",
        ip="127.0.0.1",
        port=6653
    )

    net.addController(c0)

    info("[*] Starting Mininet...\n")
    net.start()

    for sw in net.switches:
        sw.cmd(f"ovs-vsctl set bridge {sw.name} protocols=OpenFlow13")

    manager = AttackManager(net)
    manager.start_servers()
    net.manager = manager

    info("\n========== SDN SECURITY TOPO READY ==========\n")
    info("Normal users : h1 h2 h3 h4\n")
    info("Attackers    : h5 h6 h7\n")
    info("Victim       : h8 10.0.0.8\n")
    info("Honeypot     : h9 10.0.0.9\n")
    info("Redirect port on s1 to honeypot: 9\n")
    info("=============================================\n")
    info("Commands:\n")
    info("  pingall\n")
    info("  py net.manager.normal_low()\n")
    info("  py net.manager.normal_medium()\n")
    info("  py net.manager.normal_high()\n")
    info("  py net.manager.ddos_flood(num_attackers=1, intensity='low')\n")
    info("  py net.manager.ip_spoofing(num_attackers=1, intensity='medium')\n")
    info("  py net.manager.packet_in_flood(num_attackers=1, intensity='medium')\n")
    info("  py net.manager.flow_overflow(num_attackers=1, flows_per_attacker=3000)\n")
    info("  py net.manager.port_scanning(attacker_index=0, start_port=1, end_port=1000)\n")
    info("  py net.manager.stop_all()\n")
    info("  py net.manager.start_servers()\n")
    info("=============================================\n\n")

    CLI(net)

    info("[*] Stopping network...\n")
    manager.stop_all()
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    run()