print("Beginning module import!")
import socket

class PccShimDriver():
    
    flow_lookup = {}
    
    def __init__(self, flow_id):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", 9787))
        self.replay_rate = False
        self.last_rate = None
        PccShimDriver.flow_lookup[flow_id] = self
    
    def get_rate(self):
        if self.replay_rate:
            return self.last_rate
        self.replay_rate = True
        self.last_rate = float(self.sock.recv(1024).decode())
        return self.last_rate

    def reset(self):
        pass # Nothing to reset in the shim driver.

    def give_sample(self, flow_id, bytes_sent, bytes_acked, bytes_lost,
                    send_start_time, send_end_time, recv_start_time,
                    recv_end_time, rtt_samples, packet_size, utility):
        if not self.replay_rate:
            print("Detected repeat sample! Ignoring.")
            return
        self.sock.send(("%d;%d;%d;%d;%f;%f;%f;%f;%s;%d;%f\n" % (
            flow_id,
            bytes_sent,
            bytes_acked,
            bytes_lost,
            send_start_time,
            send_end_time,
            recv_start_time,
            recv_end_time,
            rtt_samples,
            packet_size,
            utility)).encode())
        self.replay_rate = False

    def get_by_flow_id(flow_id):
        return PccShimDriver.flow_lookup[flow_id]

def give_sample(flow_id, bytes_sent, bytes_acked, bytes_lost,
                send_start_time, send_end_time, recv_start_time,
                recv_end_time, rtt_samples, packet_size, utility):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    driver.give_sample(flow_id,
        bytes_sent,
        bytes_acked,
        bytes_lost,
        send_start_time,
        send_end_time,
        recv_start_time,
        recv_end_time,
        rtt_samples,
        packet_size,
        utility)
    
def reset(flow_id):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    driver.reset()

def get_rate(flow_id):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    return driver.get_rate() * 1e6

def init(flow_id):
    driver = PccShimDriver(flow_id)
