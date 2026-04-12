FROM python:3.9-slim
WORKDIR /app
# Cài đặt công cụ quản lý mô hình và web server để nhận tín hiệu tự động hóa
RUN pip install --no-cache-dir mlflow requests flask prometheus_client pandas torch numpy
COPY . /app
ENV PYTHONPATH=/app
EXPOSE 5000 5001
# Mặc định chạy script trigger nếu không có lệnh override
CMD ["python", "auto_trigger.py"]