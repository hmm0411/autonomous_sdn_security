# README HƯỚNG DẪN LÀM VIỆC NHÓM

(Autonomous SDN Security Platform)

---

# 1️. Tổng quan dự án

Dự án xây dựng hệ thống phòng vệ mạng SDN tự động gồm:

* NT549 → Reinforcement Learning decision engine
* NT114 → Digital Twin validation layer
* NT548 → MLOps + CI/CD + Monitoring

Hệ thống được container hóa bằng Docker và chạy bằng:

```bash
docker compose up --build
```

---

# 2. Cấu trúc thư mục

```
infra/controller/     → SDN Controller (mock hoặc Mininet)
rl_engine/            → RL Agent & Environment
digital_twin/         → Digital Twin
mlops/                → CI/CD, MLflow, Monitoring
docker-compose.yml    → Điều phối toàn bộ hệ thống
```

---

# 3. Quy trình làm việc 

## Bước 1 – Cập nhật code mới nhất

Trước khi bắt đầu làm:

```bash
git checkout main
git pull origin main
```

Sau đó chuyển về branch của mình:

```bash
git checkout feature/nt549-rl
```

Rồi cập nhật main vào branch:

```bash
git merge main
```

---

## Bước 2 – Làm phần của mình

Chỉ sửa đúng thư mục phụ trách.

Ví dụ NT549 chỉ sửa:

```
rl_engine/
```

---

## Bước 3 – Commit rõ ràng

Không commit chung chung kiểu "update". Ghi rõ loại commit (fix, update, docs, ...) và ghi nội dung commit cụ thể

Ví dụ:

```bash
git add .
git commit -m "Update(rl): Implement DQN replay buffer"
```

Ví dụ không nên commit:

```bash
git commit -m "update"
```

---

## Bước 4 – Push lên branch riêng

```bash
git push origin feature/nt549-rl
```

---

## Bước 5 – Tạo Pull Request

Vào GitHub:

* Compare branch với main
* Tạo Pull Request
* Thành viên khác review
* Sau khi build CI pass → merge

---

# 4. Nguyên tắc hoạt động chung

* Không push trực tiếp vào main
* Không sửa code module người khác
* Không đổi docker-compose nếu chưa trao đổi
* Luôn pull main trước khi làm

---

# 5. Quy trình tích hợp hệ thống

Mỗi tuần sẽ có 1 ngày:

* Merge tất cả branch vào main
* Test toàn hệ thống bằng:

```bash
docker compose up --build
```

Nếu hệ thống chạy ổn → tag version:

```bash
git tag v1.0
git push origin v1.0
```

---

# 7. Flow tổng thể của hệ thống

```
Controller → RL Agent → Digital Twin → MLflow
```

RL Agent:

* Lấy state từ controller
* Quyết định action
* Tính reward
* Log metric

Digital Twin:

* Mirror state
* Simulate action
* Đánh giá trước khi áp dụng thật

---

# 8. Cách clone và chạy

Khởi tạo project trên máy chỉ cần:

```bash
git clone https://github.com/hmm0411/autonomous_sdn_security.git
cd autonomous_sdn_security
docker compose up --build
```

Không cần cài Python.
Không cần cài library riêng.

---

# 9. Lộ trình phát triển

Giai đoạn 1:

* Mock controller
* Random agent

Giai đoạn 2:

* DQN thật
* Reward chuẩn QoS

Giai đoạn 3:

* Digital Twin ML-based
* Stability analysis

Giai đoạn 4:

* CI/CD hoàn chỉnh
* Deploy cloud

---