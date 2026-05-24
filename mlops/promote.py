from mlflow.tracking import MlflowClient

client = MlflowClient()

def promote_staging_to_production():
    models_to_check = ["SDN_DQN_Model", "SDN_PPO_Model"]
    
    for model_name in models_to_check:
        try:
            # Lấy version đang nằm ở Staging
            versions = client.get_latest_versions(model_name, stages=["Staging"])
            if not versions:
                continue
                
            for v in versions:
                # Transition lên Production và tự động gỡ bản cũ xuống Archived
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Production",
                    archive_existing_versions=True # RẤT QUAN TRỌNG: Gỡ bản cũ
                )
                print(f"[+] Đã thăng cấp {model_name} version {v.version} lên PRODUCTION!")
        except Exception as e:
            print(f"[-] Lỗi thăng cấp {model_name}: {e}")

if __name__ == "__main__":
    promote_staging_to_production()