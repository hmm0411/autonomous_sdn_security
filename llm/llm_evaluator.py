import pandas as pd
import re
import os
import sys

# Đảm bảo import được llm_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.llm_service import call_llm

def evaluate_llm_logs(log_file="logs/llm_reports.log", output_file="logs/llm_auto_eval.csv"):
    if not os.path.exists(log_file):
        print(f"Không tìm thấy file log tại: {log_file}")
        return

    # 1. Đọc file txt
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Lọc bỏ các dòng trống
    reports = [line.strip() for line in lines if line.strip()]
    
    if len(reports) == 0:
        print("File log trống. Chưa có báo cáo nào để chấm điểm.")
        return
        
    print(f"[*] Tìm thấy tổng cộng {len(reports)} báo cáo trong file log...")
    
    # ---------------------------------------------------------
    # GIỚI HẠN SỐ LƯỢNG MẪU ĐỂ KHÔNG BỊ HUGGING FACE KHÓA API
    # ---------------------------------------------------------
    EVAL_LIMIT = 20 # Đánh giá 20 dòng đầu tiên (Bạn có thể tăng lên 50 sau)
    reports_to_eval = reports[:EVAL_LIMIT]
    print(f"[*] Đang trích xuất {len(reports_to_eval)} báo cáo đầu tiên để chấm điểm...")

    results = []
    
    # 2. Vòng lặp chấm điểm
    for index, text in enumerate(reports_to_eval):
        # Trích xuất Action nếu có trong chuỗi
        action_match = re.search(r'(Action|Hành động)[\s:]*(\d+)', text, re.IGNORECASE)
        action = action_match.group(2) if action_match else "Tự phân tích từ văn bản"
        
        judge_prompt = f"""
        Bạn là một chuyên gia mạng SDN. Hãy đánh giá đoạn báo cáo an ninh sau đây trên thang điểm 1-5 cho 3 tiêu chí:
        1. Accuracy (Tính chính xác của báo cáo, đối chiếu với hành động: {action})
        2. Clarity (Rõ ràng, dễ hiểu)
        3. Usefulness (Tính hữu ích cho người quản trị mạng)
        
        Báo cáo cần đánh giá: "{text}"
        
        CHỈ trả về kết quả dưới định dạng: Accuracy: [điểm], Clarity: [điểm], Usefulness: [điểm]. Không giải thích thêm.
        """
        
        print(f"  -> Đang chấm điểm dòng {index + 1}/{len(reports_to_eval)}...")
        try:
            score_text = call_llm(judge_prompt)
            print(f"     => Kết quả: {score_text.strip()}")
            
            results.append({
                "action": action,
                "explanation": text,
                "scores": score_text.strip()
            })
        except Exception as e:
            print(f"     [!] Lỗi khi gọi API: {e}")

    # 3. Lưu kết quả ra file CSV
    eval_df = pd.DataFrame(results)
    eval_df.to_csv(output_file, index=False)
    print(f"\nHoàn tất! Kết quả đã được lưu tại: {output_file}")

if __name__ == "__main__":
    evaluate_llm_logs()