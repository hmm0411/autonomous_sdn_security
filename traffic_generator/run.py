from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from topo import SDNResearchTopo
from attack_manager import AttackManager
import time


def run_experiment():

    topo = SDNResearchTopo()

    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        switch=lambda name, **opts: OVSSwitch(name, protocols='OpenFlow13', **opts)
    )

    print("[*] Starting Mininet...")
    net.start()

    print("[*] Waiting controller connection (5s)...")
    time.sleep(5)

    manager = AttackManager(net)
    net.manager = manager

    print("="*40)
    print("SYSTEM READY")
    print("="*40)
    print("pingall → test network")
    print("py net.manager.ddos_flood()")
    print("py net.manager.packet_in_flood()")
    print("py net.manager.flow_overflow()")
    print("py net.manager.ip_spoofing()")
    print("py net.manager.port_scanning()")
    print("py net.manager.stop_all()")
    print("="*40)

    CLI(net)

    print("[*] Stopping network...")
    net.stop()


if __name__ == "__main__":
    run_experiment()
