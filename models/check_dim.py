import joblib
import torch

def check_scaler(file_path):
    print(f"\n🔍 CHECKING SCALER: {file_path}")
    try:
        scaler = joblib.load(file_path)
        # Thư viện scikit-learn lưu số features đầu vào trong thuộc tính n_features_in_
        if hasattr(scaler, 'n_features_in_'):
            print(f"✅ Kết quả: Scaler này được train với {scaler.n_features_in_} features (dims).")
        else:
            print("⚠️ Không tìm thấy thuộc tính n_features_in_ (có thể do phiên bản scikit-learn cũ).")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def check_pytorch_model(file_path):
    print(f"\n🔍 CHECKING PYTORCH MODEL: {file_path}")
    try:
        # Load mô hình vào RAM (dùng map_location='cpu' để tránh lỗi nếu máy không có GPU)
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
        
        # Lấy state_dict (danh sách các trọng số)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Duyệt qua các layer để tìm layer Linear đầu tiên
        for key, tensor in state_dict.items():
            if "weight" in key and len(tensor.shape) == 2:
                # Trong PyTorch, shape của nn.Linear là [out_features, in_features]
                in_features = tensor.shape[1]
                print(f"✅ Kết quả: Layer đầu tiên '{key}' có Input Dimension = {in_features}")
                break
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    # Đảm bảo bạn đang chạy file này ở thư mục chứa thư mục 'models'
    check_scaler("models/scaler.pkl")
    check_pytorch_model("models/dqn_model.pth")
    check_pytorch_model("models/ppo_model.pth")