# safety.py

def validate(predicted_qos):
    """
    Kiểm tra xem hành động RL đề xuất có gây sập mạng hoặc 
    giảm chất lượng dịch vụ (QoS) quá mức cho phép không.
    """
    if predicted_qos is None:
        return False
        
    LATENCY_THRESHOLD = 150.0  # Ví dụ: Không cho phép trễ quá 150ms
    PACKET_LOSS_THRESHOLD = 0.1 # Ví dụ: Không cho phép rớt gói quá 10%
    
    if predicted_qos["latency"] > LATENCY_THRESHOLD:
        print(f"Twin CẢNH BÁO: Độ trễ dự đoán quá cao ({predicted_qos['latency']}ms)")
        return False
        
    if predicted_qos["packet_loss"] > PACKET_LOSS_THRESHOLD:
        print(f"Twin CẢNH BÁO: Tỉ lệ rớt gói dự đoán quá cao ({predicted_qos['packet_loss']})")
        return False
        
    return True