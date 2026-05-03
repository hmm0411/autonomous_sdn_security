import streamlit as st
import os
import time

st.set_page_config(page_title="SDN AI Control Panel", layout="wide")


# SIDEBAR

with st.sidebar:
    st.subheader("Quick Links")
    st.markdown("Truy cập nhanh các dịch vụ hệ thống:")
    
    # Tạo các nút bấm mở ra Tab mới
    st.link_button("ONOS Controller GUI", "http://localhost:8181/onos/ui", use_container_width=True)
    st.link_button("MLflow (AI Tracking)", "http://localhost:5000", use_container_width=True)
    st.link_button("Grafana Dashboards", "http://localhost:3000", use_container_width=True)
    st.link_button("Prometheus Metrics", "http://localhost:9090", use_container_width=True)
    st.link_button("Alertmanager", "http://localhost:9093", use_container_width=True)
    
    st.divider()
    st.caption("Autonomous SDN Security Project")

# MAIN DASHBOARD

st.title("Autonomous SDN Security Dashboard")

# Cấu hình đường dẫn file trigger
TRIGGER_FILE = "logs/llm_on"

# Giao diện điều khiển
col1, col2 = st.columns(2)

with col1:
    st.header("LLM Control")
    is_on = os.path.exists(TRIGGER_FILE)
    
    if is_on:
        st.success("Trạng thái: LLM đang BẬT")
        if st.button("TẮT Lớp Nhận Thức LLM"):
            if os.path.exists(TRIGGER_FILE):
                os.remove(TRIGGER_FILE)
            st.rerun()
    else:
        st.error("Trạng thái: LLM đang TẮT")
        if st.button("BẬT Lớp Nhận Thức LLM"):
            # Tạo thư mục logs nếu chưa có để tránh lỗi
            os.makedirs("logs", exist_ok=True)
            with open(TRIGGER_FILE, "w") as f:
                f.write("ON")
            st.rerun()

with col2:
    st.header("System Logs")
    if st.button("Xóa Log cũ"):
        if os.path.exists("logs/llm_reports.log"):
            os.remove("logs/llm_reports.log")
            st.info("Đã dọn dẹp nhật ký.")

# Hiển thị báo cáo LLM mới nhất
st.divider()
st.subheader("LLM Security Report")
if os.path.exists("logs/llm_reports.log"):
    with open("logs/llm_reports.log", "r", encoding="utf-8") as f:
        content = f.read()
        with st.container(height=400): 
            st.code(content, language="text")
else:
    st.info("Chưa có báo cáo nào được ghi nhận.")