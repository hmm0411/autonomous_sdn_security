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

    # Start Mininet
    PYTHONPATH=$(pwd) sudo python3 -m traffic_generator.run <<EOF &
sleep 5
py net.manager.start_servers()
$CMD
EOF

    MN_PID=$!

    sleep 5

        # Run collector
        PYTHONPATH=$(pwd) sudo python3 -m traffic_generator.onos_collector \
        --label $LABEL \
        --samples 200 \
        --interval 1 \
        --output $DATA_DIR/${NAME}.csv

    echo "[*] Stop Mininet"
    sudo mn -c || true

    kill $MN_PID 2>/dev/null || true

    sleep 5
}

# ========================
# RUN ALL SCENARIOS
# ========================

run_scenario () {
    NAME=$1
    LABEL=$2
    CMD=$3

    echo "===== RUN SCENARIO: $NAME ====="

    # Start Mininet in background (giữ sống)
    sudo env "PYTHONPATH=$(pwd)" python3 -m traffic_generator.run > /tmp/mn_$NAME.log 2>&1 &
    MN_PID=$!

    echo "[*] Waiting Mininet to start..."
    sleep 10

    # Inject command vào CLI Mininet
    echo "$CMD" | sudo mnexec -a $(pgrep -f "mininet:") sh

    sleep 5

    # Run collector
    PYTHONPATH=$(pwd) python3 -m traffic_generator.onos_collector \
        --label $LABEL \
        --samples 500 \
        --interval 1 \
        --output data/${NAME}.csv

    echo "[*] Stop Mininet"
    sudo mn -c || true
    kill $MN_PID 2>/dev/null || true

    sleep 5
}
# ========================
# MERGE DATA
# ========================

echo "===== MERGING DATA ====="

PYTHONPATH=$(pwd) sudo python3 <<EOF
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