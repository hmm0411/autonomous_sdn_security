import requests

CONTROLLER_URL = "http://controller:8080"

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