import pandas as pd
import os

df = pd.read_csv("data/processed/test_data.csv")

os.makedirs("data/processed", exist_ok=True)

df[df.attack_indicator == 0.0].to_csv("data/processed/test_normal.csv", index=False)
df[df.attack_indicator == 0.2].to_csv("data/processed/test_ddos.csv", index=False)
df[df.attack_indicator == 0.4].to_csv("data/processed/test_spoofing.csv", index=False)
df[df.attack_indicator == 0.6].to_csv("data/processed/test_overflow.csv", index=False)
df[df.attack_indicator == 0.8].to_csv("data/processed/test_packet_in.csv", index=False)
df[df.attack_indicator == 1.0].to_csv("data/processed/test_port_scan.csv", index=False)

print("Scenario files generated.")