import os


LATENCY_THRESHOLD = float(os.getenv("TWIN_LATENCY_THRESHOLD", "150.0"))
PACKET_LOSS_THRESHOLD = float(os.getenv("TWIN_PACKET_LOSS_THRESHOLD", "0.10"))


def validate(predicted_qos):
    """
    Safety validator cho Digital Twin.

    Return:
      True  = action an toàn
      False = action nên bị reject/fallback
    """
    if predicted_qos is None:
        return False

    latency = float(predicted_qos.get("latency", 999999.0))
    packet_loss = float(predicted_qos.get("packet_loss", 1.0))

    if latency > LATENCY_THRESHOLD:
        print(
            f"[TWIN_REJECT] latency too high: "
            f"{latency:.4f} > {LATENCY_THRESHOLD}",
            flush=True,
        )
        return False

    if packet_loss > PACKET_LOSS_THRESHOLD:
        print(
            f"[TWIN_REJECT] packet_loss too high: "
            f"{packet_loss:.4f} > {PACKET_LOSS_THRESHOLD}",
            flush=True,
        )
        return False

    return True