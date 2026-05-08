def compute_gap(predicted, real_state):

    real_latency = real_state[4]
    real_loss = real_state[5]

    gap_latency = abs(real_latency - predicted["latency"])
    gap_loss = abs(real_loss - predicted["packet_loss"])

    return gap_latency, gap_loss