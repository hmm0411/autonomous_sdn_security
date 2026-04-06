import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Cố định toàn bộ các bộ sinh số ngẫu nhiên để đảm bảo Reproducibility.
    """
    # 1. Cố định seed cho Python Hash (ảnh hưởng đến thứ tự dict/set)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Cố định thư viện random chuẩn của Python
    random.seed(seed)
    
    # 3. Cố định Numpy
    np.random.seed(seed)
    
    # 4. Cố định PyTorch trên CPU và GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Dành cho trường hợp chạy Multi-GPU
        
    # 5. Ép cuDNN chạy ở chế độ Deterministic (Bắt buộc để GPU luôn ra 1 kết quả)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🌱 Đã cố định toàn cục Seed = {seed}")