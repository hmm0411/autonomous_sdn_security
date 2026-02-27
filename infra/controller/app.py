from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route("/state")
def get_state():
    return jsonify({
        "latency": random.randint(20, 50),
        "throughput": random.randint(700, 1000),
        "packet_loss": round(random.uniform(0.0, 0.05), 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)