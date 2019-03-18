
print("Importing module!")

def give_sample(flow_id, bytes_sent, bytes_acked, bytes_lost,
                send_start_time, send_end_time, recv_start_time,
                recv_end_time, rtt_samples, packet_size, utility):
    print("Got Sample:")
    print("\tflow_id: %d" % flow_id)
    print("\tbytes_sent: %d" % bytes_sent)
    print("\tbytes_acked: %d" % bytes_acked)
    print("\tbytes_lost: %d" % bytes_lost)
    print("\tsend_start_time: %f" % send_start_time)
    print("\tsend_end_time: %f" % send_end_time)
    print("\trecv_start_time: %f" % recv_start_time)
    print("\trecv_end_time: %f" % recv_end_time)
    print("\trtt_samples: %s" % rtt_samples)
    print("\tpacket_size: %d" % packet_size)
    print("\tutility: %f" % utility)
    
def reset(flow_id):
    pass

def get_rate(flow_id):
    return 3e6

def init(flow_id):
    pass
