from mlops.mlflow_manager import get_best_model
from mlflow.tracking import MlflowClient

client = MlflowClient()

def promote_best():
    best_model, score = get_best_model()

    client.transition_model_version_stage(
        name=best_model.name,
        version=best_model.version,
        stage="Production"
    )

    print(f"Promote {best_model.name} v{best_model.version} | score={score}")

if __name__ == "__main__":
    promote_best()