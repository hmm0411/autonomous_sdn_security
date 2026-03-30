import requests

CONTROLLER_URL = "http://localhost:8181/onos/v1"  # URL của SDN controller

def sync_from_controller():
    """
    Lấy trạng thái hiện tại từ SDN controller
    """
    try:
        response = requests.get(f"{CONTROLLER_URL}/state")
        return response.json()
    except Exception as e:
        print("Error syncing state:", e)
        return None