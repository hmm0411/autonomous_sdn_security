#!/bin/bash

set -e

DATA_DIR="data"
mkdir -p $DATA_DIR

echo "===== CLEAN ENV ====="
sudo mn -c || true

run_scenario () {
    NAME=$1
    LABEL=$2
    CMD=$3

    echo "===== RUN SCENARIO: $NAME ====="

    # Start Mininet + traffic
    sudo env "PYTHONPATH=$(pwd)" python3 - <<EOF &
from traffic_generator.run import main
import time

net = main()

time.sleep(5)
net.manager.start_servers()

$CMD

time.sleep(300)
EOF

    MN_PID=$!

    sleep 10

    # Run collector
    PYTHONPATH=$(pwd) python3 -m traffic_generator.onos_collector \
        --label $LABEL \
        --samples 200 \
        --interval 1 \
        --output data/${NAME}.csv

    echo "[*] Stop Mininet"
    sudo mn -c || true
    kill $MN_PID 2>/dev/null || true

    sleep 5
}

# ========================
# RUN ALL SCENARIOS
# ========================

echo "===== START DATA COLLECTION ====="

run_scenario "normal" 0 "py net.manager.normal_medium()"
run_scenario "ddos" 1 "py net.manager.ddos_flood(num_attackers=2, intensity='medium')"
run_scenario "spoof" 2 "py net.manager.ip_spoofing(num_attackers=2, intensity='medium')"
run_scenario "packet_in" 3 "py net.manager.packet_in_flood(num_attackers=2, intensity='medium')"
run_scenario "flow_overflow" 4 "py net.manager.flow_overflow(num_attackers=2, flows_per_attacker=300)"
run_scenario "port_scan" 5 "py net.manager.port_scanning(attacker_index=0, start_port=1, end_port=500)"

# ========================
# MERGE DATA
# ========================

echo "===== MERGING DATA ====="

PYTHONPATH=$(pwd) python3 <<EOF
import pandas as pd
import os

data_dir = "data"

files = [
    "normal.csv",
    "ddos.csv",
    "spoof.csv",
    "packet_in.csv",
    "flow_overflow.csv",
    "port_scan.csv"
]

dfs = []
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))

df = pd.concat(dfs)
df.to_csv("data/train_data.csv", index=False)

print("[+] DONE: data/train_data.csv")
EOF

echo "===== ALL DONE ====="