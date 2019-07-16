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

    def give_sample(self, sending_rate, recv_rate, latency, loss, lat_infl, utility):
        if not self.replay_rate:
            print("Detected repeat sample! Ignoring.")
            return
        self.sock.send(("%f,%f,%f,%f,%f,%f\n" % (sending_rate, recv_rate, latency,
            loss, lat_infl, utility)).encode())
        self.replay_rate = False

    def get_by_flow_id(flow_id):
        return PccShimDriver.flow_lookup[flow_id]

def give_sample(flow_id, sending_rate, recv_rate, latency, loss, lat_infl, utility):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    #print("Sent reward")
    driver.give_sample(sending_rate, recv_rate, latency, loss, lat_infl, utility)
    
def reset(flow_id):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    driver.reset()

def get_rate(flow_id):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    #print("Waiting for rate")
    return driver.get_rate() * 1e6

def init(flow_id):
    driver = PccShimDriver(flow_id)
