from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route("/state")
def get_state():
    return jsonify({
        "packet_rate": random.randint(100, 1000),
        "byte_rate": random.randint(10000, 1000000),
        "flow_count": random.randint(10, 200),
        "src_ip_entropy": round(random.uniform(0.5, 2.0), 3),
        "latency": random.randint(20, 50),
        "packet_loss": round(random.uniform(0.0, 0.05), 3),
        "queue_length": random.randint(0, 100),
        "controller_cpu": round(random.uniform(0.1, 0.5), 3),
        "attack_indicator": random.choice([0, 1]),
        "previous_action": random.choice([0, 1, 2, 3, 4])
    })

if __name__ == "__main__":
    app.run(host="34.126.64.185", port=8181)