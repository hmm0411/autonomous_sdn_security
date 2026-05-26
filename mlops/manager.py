# mlops/manager.py
from time import time

import mlflow
from mlflow.tracking import MlflowClient
import requests
from kubernetes import client, config

config.load_incluster_config()
job_api = client.BatchV1Api()
mlflow_client = MlflowClient()

def promote_model(model_name):
    # Lấy bản tốt nhất từ Staging/None
    best_model, _ = get_best_model(model_name)
    mlflow_client.transition_model_version_stage(
        name=best_model.name, version=best_model.version, stage="Production"
    )
    # Reload Serving
    requests.post(f"http://rl-serving-{model_name.lower().split('_')[1]}:8000/reload")

def trigger_retrain(model_name):
    # Định nghĩa Job để tạo Retrain
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": f"retrain-{model_name.lower()}-{int(time.time())}"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [{"name": "trainer", "image": "hmm0411/sdn-rl-agent:latest", "command": ["python", f"rl_engine/agent/train_{model_name.lower()}.py"]}],
                    "restartPolicy": "OnFailure"
                }
            }
        }
    }
    job_api.create_namespaced_job(namespace="sdn-security", body=job_manifest)