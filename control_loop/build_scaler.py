import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

FILES = [
    r"C:\Users\myngu\Desktop\autonomous_sdn_security\data\processed\train_data.csv",
    r"C:\Users\myngu\Desktop\autonomous_sdn_security\data\processed\normal.csv",
    r"C:\Users\myngu\Desktop\autonomous_sdn_security\data\processed\ddos.csv",
    r"C:\Users\myngu\Desktop\autonomous_sdn_security\data\processed\flow_overflow.csv",
    r"C:\Users\myngu\Desktop\autonomous_sdn_security\data\processed\packet_in_flood.csv",
    r"C:\Users\myngu\Desktop\autonomous_sdn_security\data\processed\ip_spoofing.csv",
    r"C:\Users\myngu\Desktop\autonomous_sdn_security\data\processed\port_scanning.csv"
]

dfs = []

for f in FILES:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

FEATURES = [
    "packet_rate",
    "byte_rate",
    "flow_count",
    "src_ip_entropy",
    "latency",
    "packet_loss",
    "queue_length",
    "controller_cpu"
]

scaler = MinMaxScaler()
scaler.fit(data[FEATURES])

joblib.dump(scaler, r"C:\Users\myngu\Desktop\autonomous_sdn_security\models\scaler.pkl")

print("New scaler saved successfully.")