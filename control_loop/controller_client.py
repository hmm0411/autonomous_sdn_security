import json
import os
import time

import requests


ONOS_URL = os.getenv("ONOS_URL", "http://controller:8181/onos/v1")
AUTH = (
    os.getenv("ONOS_USER", "onos"),
    os.getenv("ONOS_PASS", "rocks"),
)

DEVICE_ID = os.getenv("ONOS_DEVICE_ID", "of:0000000000000001")


def _headers():
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _install_drop_ipv4_rule(timeout=20):
    """
    Cài flow drop IPv4 tạm thời.
    Không dùng DELETE /flows/{deviceId} vì rất nguy hiểm.
    """
    flow_rule = {
        "priority": 50000,
        "timeout": int(timeout),
        "isPermanent": False,
        "deviceId": DEVICE_ID,
        "treatment": {
            "instructions": []
        },
        "selector": {
            "criteria": [
                {
                    "type": "ETH_TYPE",
                    "ethType": "0x0800"
                }
            ]
        }
    }

    try:
        url = f"{ONOS_URL}/flows/{DEVICE_ID}"

        response = requests.post(
            url,
            auth=AUTH,
            headers=_headers(),
            data=json.dumps(flow_rule),
            timeout=3,
        )

        print(
            f"[ONOS_ACTION] install_drop_ipv4 status={response.status_code} "
            f"body={response.text[:200]}",
            flush=True,
        )

        return response.status_code in (200, 201)

    except Exception as e:
        print(f"[ONOS_ACTION_ERROR] drop rule error={e}", flush=True)
        return False


def execute_action(action, raw=None):
    """
    Action map:
    0 no_action
    1 block_suspicious_flow
    2 limit_bandwidth
    3 redirect_traffic
    4 isolate_device

    Lưu ý:
    - Khi đánh giá nghiên cứu, nên dùng ACTION_DRY_RUN=true.
    - Chỉ bật apply thật khi demo và đã xác nhận đúng device/port.
    """
    action = int(action)

    if action == 0:
        print("[ACTION] no_action", flush=True)
        return True

    if action == 1:
        print("[ACTION] block_suspicious_flow", flush=True)
        return _install_drop_ipv4_rule(timeout=20)

    if action == 2:
        print(
            "[ACTION] limit_bandwidth requested. "
            "No destructive ONOS rule installed by default.",
            flush=True,
        )
        return True

    if action == 3:
        print(
            "[ACTION] redirect_traffic requested. "
            "Implement honeypot flow after confirming output ports.",
            flush=True,
        )
        return True

    if action == 4:
        print(
            "[ACTION] isolate_device requested. "
            "No destructive isolation rule installed by default.",
            flush=True,
        )
        return True

    print(f"[ACTION] unknown action={action}", flush=True)
    return False