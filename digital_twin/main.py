# vòng lặp chạy twin
# 1. Gọn sync
# 2. Update twin
# 3. Simulate
# 4. Log kết quả
from asyncio import sleep
import time

from sync import sync_from_controller
from twin import DigitalTwin

twin = DigitalTwin()

while True:
    state = sync_from_controller()
    if state:
        twin.update_state(state)

        predicted = twin.simulate("block")
        print("Predicted latency:", predicted)
        # Log kết quả (có thể dùng MLflow hoặc đơn giản là print)
        time.sleep(3)  # sleep để tránh quá tải controller