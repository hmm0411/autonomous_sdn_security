# Nhom04-NT548 Microservices CI/CD Pipeline

## 1. Giới thiệu

Dự án này được thực hiện cho Câu 3 - Bài tập thực hành 02 môn NT548: Công nghệ DevOps và Ứng dụng.

Mục tiêu của dự án là xây dựng quy trình CI/CD cho ứng dụng microservices bằng Jenkins. Pipeline tự động thực hiện các bước: lấy mã nguồn từ GitHub, phân tích chất lượng mã nguồn bằng SonarQube, build Docker images, push images lên Docker Hub và deploy ứng dụng lên Kubernetes.

## 2. Kiến trúc CI/CD

Luồng CI/CD của hệ thống:

```text
GitHub
   ↓
Jenkins Pipeline
   ↓
SonarQube Scan
   ↓
Docker Build
   ↓
Docker Hub
   ↓
Kubernetes / K3s
```

Các công cụ sử dụng:

* Jenkins: quản lý pipeline CI/CD
* GitHub: lưu trữ mã nguồn
* Docker: đóng gói ứng dụng thành container images
* Docker Hub: lưu trữ Docker images
* SonarQube: kiểm tra chất lượng mã nguồn
* Kubernetes / K3s: triển khai và quản lý container
* Flask Python: xây dựng microservices

## 3. Cấu trúc thư mục

```text
Nhom04-NT548-microservices
│   Jenkinsfile
│   README.md
│
├───k8s
│       product-deployment.yaml
│       user-deployment.yaml
│
├───service-product
│       app.py
│       Dockerfile
│       requirements.txt
│
└───service-user
        app.py
        Dockerfile
        requirements.txt
```

## 4. Mô tả các thành phần

### 4.1. service-user

`service-user` là microservice đơn giản được viết bằng Flask Python. Service này chạy ở port 5000 và trả về thông tin trạng thái hoạt động của user service.

Kết quả mong đợi:

```json
{"service":"user","status":"running"}
```

### 4.2. service-product

`service-product` là microservice thứ hai được viết bằng Flask Python. Service này chạy ở port 5001 và trả về thông tin trạng thái hoạt động của product service.

Kết quả mong đợi:

```json
{"service":"product","status":"running"}
```

### 4.3. k8s

Thư mục `k8s` chứa các file YAML dùng để deploy ứng dụng lên Kubernetes. Mỗi microservice có một file deployment riêng, bao gồm Deployment và Service dạng NodePort.

### 4.4. Jenkinsfile

`Jenkinsfile` định nghĩa toàn bộ pipeline CI/CD, bao gồm các stage:

* Checkout SCM
* SonarQube Scan
* Build Docker Images
* Push Docker Images
* Deploy Kubernetes

## 5. Yêu cầu môi trường

Máy chủ triển khai sử dụng Ubuntu trên AWS EC2.

Các phần mềm cần cài đặt:

```bash
sudo apt update
sudo apt install -y git docker.io openjdk-21-jdk
```

Cài đặt Jenkins, SonarQube và Kubernetes K3s.

Kiểm tra các công cụ:

```bash
git --version
docker --version
java --version
kubectl version
```

## 6. Cài đặt và chạy ứng dụng cục bộ

Clone repository:

```bash
git clone https://github.com/van2352/Nhom04-NT548-microservices.git
cd Nhom04-NT548-microservices
```

Build Docker image cho user service:

```bash
docker build -t user-service ./service-user
```

Build Docker image cho product service:

```bash
docker build -t product-service ./service-product
```

Chạy thử user service:

```bash
docker run -d --name user-test -p 5000:5000 user-service
```

Chạy thử product service:

```bash
docker run -d --name product-test -p 5001:5001 product-service
```

Kiểm tra kết quả:

```bash
curl http://localhost:5000
curl http://localhost:5001
```

Kết quả mong đợi:

```json
{"service":"user","status":"running"}
```

```json
{"service":"product","status":"running"}
```

Dừng container test:

```bash
docker rm -f user-test product-test
```

## 7. Build và push Docker images

Đăng nhập Docker Hub:

```bash
docker login
```

Build images:

```bash
docker build -t tuongvan23521768/user-service:latest ./service-user
docker build -t tuongvan23521768/product-service:latest ./service-product
```

Push images lên Docker Hub:

```bash
docker push tuongvan23521768/user-service:latest
docker push tuongvan23521768/product-service:latest
```

Docker Hub repositories:

```text
tuongvan23521768/user-service
tuongvan23521768/product-service
```

## 8. Deploy lên Kubernetes

Cài đặt K3s:

```bash
curl -sfL https://get.k3s.io | sh -
```

Cấu hình kubectl:

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config
export KUBECONFIG=$HOME/.kube/config
```

Kiểm tra node:

```bash
kubectl get nodes
```

Deploy ứng dụng:

```bash
kubectl apply -f k8s/
```

Kiểm tra pods:

```bash
kubectl get pods -o wide
```

Kiểm tra services:

```bash
kubectl get svc
```

Kết quả service mong đợi:

```text
user-service      NodePort   5000:30080/TCP
product-service   NodePort   5001:30081/TCP
```

## 9. Kiểm tra kết quả triển khai

Kiểm tra user service:

```bash
curl http://localhost:30080
```

Kết quả:

```json
{"service":"user","status":"running"}
```

Kiểm tra product service:

```bash
curl http://localhost:30081
```

Kết quả:

```json
{"service":"product","status":"running"}
```

Nếu đã mở port trên AWS Security Group, có thể kiểm tra bằng trình duyệt:

```text
http://32.196.117.220:30080
http://32.196.117.220:30081
```

## 10. Cấu hình Jenkins Pipeline

Tạo Jenkins Pipeline Job với tên:

```text
Nhom04-NT548-Pipeline
```

Cấu hình Pipeline:

```text
Pipeline script from SCM
SCM: Git
Repository URL: https://github.com/van2352/Nhom04-NT548-microservices.git
Branch: */main
Script Path: Jenkinsfile
```

Các plugin Jenkins đã sử dụng:

* Pipeline
* Git
* Docker Pipeline
* SonarQube Scanner
* Kubernetes CLI

## 11. Cấu hình SonarQube

SonarQube được chạy bằng Docker:

```bash
docker run -d --name sonarqube -p 9000:9000 sonarqube:lts-community
```

Truy cập SonarQube:

```text
http://32.196.117.220:9000
```

Project key:

```text
nhom04-nt548
```

Kết quả phân tích mã nguồn:

```text
Quality Gate: Passed
Bugs: 0
Vulnerabilities: 0
Security Hotspots: 0
Code Smells: 0
```

## 12. Jenkins Pipeline Stages

Pipeline gồm các stage chính:

### 12.1. Checkout SCM

Jenkins lấy mã nguồn từ GitHub repository.

### 12.2. SonarQube Scan

Jenkins sử dụng SonarScanner để phân tích chất lượng mã nguồn. Kết quả được hiển thị trên SonarQube Dashboard.

### 12.3. Build Docker Images

Jenkins build Docker images cho hai microservices:

```bash
docker build -t tuongvan23521768/user-service:latest ./service-user
docker build -t tuongvan23521768/product-service:latest ./service-product
```

### 12.4. Push Docker Images

Jenkins push Docker images lên Docker Hub:

```bash
docker push tuongvan23521768/user-service:latest
docker push tuongvan23521768/product-service:latest
```

### 12.5. Deploy Kubernetes

Jenkins deploy ứng dụng lên Kubernetes:

```bash
kubectl apply -f k8s/
kubectl rollout restart deployment/user-service
kubectl rollout restart deployment/product-service
```

## 13. Kiểm tra sau khi pipeline chạy

Kiểm tra trạng thái Jenkins Pipeline:

```text
Finished: SUCCESS
```

Kiểm tra Kubernetes:

```bash
kubectl get pods
kubectl get svc
```

Kiểm tra API:

```bash
curl http://localhost:30080
curl http://localhost:30081
```

Kết quả mong đợi:

```json
{"service":"user","status":"running"}
```

```json
{"service":"product","status":"running"}
```

## 14. Kết quả đạt được

Dự án đã xây dựng thành công quy trình CI/CD tự động cho ứng dụng microservices. Khi pipeline chạy, Jenkins tự động lấy mã nguồn từ GitHub, quét chất lượng mã bằng SonarQube, build Docker images, push images lên Docker Hub và deploy ứng dụng lên Kubernetes.

Kết quả đạt được:

* Mã nguồn được quản lý trên GitHub
* Ứng dụng được container hóa bằng Docker
* Docker images được lưu trữ trên Docker Hub
* Mã nguồn được phân tích bằng SonarQube
* Ứng dụng được triển khai trên Kubernetes
* Jenkins Pipeline chạy thành công toàn bộ quy trình CI/CD

## 15. Link liên quan

GitHub Repository:

```text
https://github.com/van2352/Nhom04-NT548-microservices
```

Docker Hub:

```text
https://hub.docker.com/r/tuongvan23521768/user-service
https://hub.docker.com/r/tuongvan23521768/product-service
```

SonarQube Dashboard:

```text
http://32.196.117.220:9000/dashboard?id=nhom04-nt548
```

Application URLs:

```text
http://32.196.117.220:30080
http://32.196.117.220:30081
```
