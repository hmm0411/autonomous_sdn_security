FROM python:3.9-slim
WORKDIR /app
# Cài đặt công cụ quản lý mô hình và web server để nhận tín hiệu tự động hóa
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir mlflow requests flask prometheus_client 
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . /app
ENV PYTHONPATH=/app
EXPOSE 5000 5001
# Mặc định chạy script trigger nếu không có lệnh override
CMD ["python", "auto_trigger.py"]