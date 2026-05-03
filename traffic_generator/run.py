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
        controller=lambda name: RemoteController(name, ip='34.126.64.185', port=6653),
        switch=lambda name, **opts: OVSSwitch(name, protocols='OpenFlow13', **opts)
    )

    print("[*] Starting Mininet...")
    net.start()
    print("[*] Enabling STP on OVS bridges to prevent Broadcast Storm...")
    for sw in net.switches:
        # Ra lệnh cho switch bật STP
        sw.cmd('ovs-vsctl set bridge', sw.name, 'stp_enable=true')

    # ------------------------
    print("[*] Waiting for STP Convergence (20s)...")
    time.sleep(20)
    print("[*] Connecting Root Namespace to Mininet (Fixing 500.0 Latency)...")

    import os
    os.system('sudo ip addr add 10.0.0.100/24 dev s1')
    os.system('sudo ip link set dev s1 up')

    print("[*] Initializing Attack Manager...")
    manager = AttackManager(net)
    net.manager = manager
    print("[*] Waiting controller connection (5s)...")
    time.sleep(5)

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
    os.system('sudo ip addr del 10.0.0.100/24 dev s1 2>/dev/null')
    print("[*] Stopping network...")
    net.stop()

if __name__ == "__main__":
    run_experiment()