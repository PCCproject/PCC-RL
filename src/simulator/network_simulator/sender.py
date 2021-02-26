import sys

import numpy as np

from common import config, sender_obs
from simulator.network_simulator.constants import BYTES_PER_PACKET
from simulator.network_simulator.monitor_interval import MonitorInterval

MIN_PKTS_PER_MI = 5

class Sender():

    _next_id = 1

    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10):
        """Create a sender object.

        Args
            rate: start sending rate. (Unit: packets per second).
            path: network path a packet needs to traverse.
            dest: id of destination device.
            features: a list of features used by RL model.
            cwnd: size of congestion window. (Unit: number of packets)
            history_len: length of history array. History array is used as the
                input of RL model.
        """
        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_sample = 0
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd
        self.use_cwnd = False
        self.rto = -1
        self.mi_cache = {}  # a queue holding monitor intervals
        self.mi_count = 0
        self.got_data = False

    @staticmethod
    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta):
        delta *= config.DELTA_SCALE
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))
        # print("current rate {} after applying delta {}".format(self.rate, delta))
        # print("rate %f" % delta)

    def apply_cwnd_delta(self, delta):
        delta *= config.DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if self.use_cwnd:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self, mi_id):
        self.bytes_in_flight += BYTES_PER_PACKET
        self.mi_cache[mi_id].on_packet_sent(self.net.get_cur_time())

    def on_packet_acked(self, rtt, mi_id):
        self.rtt_sample = rtt
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET
        try:
            self.mi_cache[mi_id].on_packet_acked(self.net.get_cur_time(), rtt)
        except:
            import ipdb
            ipdb.set_trace()
        self.got_data = True

    def on_packet_lost(self, mi_id):
        self.bytes_in_flight -= BYTES_PER_PACKET
        self.mi_cache[mi_id].on_packet_lost()

    def set_rate(self, new_rate):
        self.rate = new_rate
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        # if self.rate > MAX_RATE:
        #     self.rate = MAX_RATE
        # if self.rate < MIN_RATE:
        #     self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        # if self.cwnd > MAX_CWND:
        #     self.cwnd = MAX_CWND
        # if self.cwnd < MIN_CWND:
        #     self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        # smi.debug_print()
        self.history.step(smi)
        print(self.history.as_array(), file=sys.stderr)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        # obs_end_time = self.net.get_cur_time()

        #obs_dur = obs_end_time - self.obs_start_time
        # print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        # print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        # print("self.rate = %f" % self.rate)
        # print(self.acked, self.sent)
        # print(self.rtt_samples)
        mi = self.mi_cache[min(self.mi_cache)]
        if mi.rtt_samples:
            rtt_samples = [[mi.rtt_samples[0], mi.rtt_samples[-1]]]
        else:
            rtt_samples = mi.rtt_samples

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=mi.bytes_sent,
            bytes_acked=mi.bytes_acked,
            bytes_lost=mi.bytes_lost,
            send_start=mi.first_packet_send_time,
            send_end=mi.last_packet_send_time,
            recv_start=mi.first_packet_ack_time,
            recv_end=mi.last_packet_ack_time,
            rtt_samples=rtt_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        pass

    # def print_debug(self):
    #     print("Sender:")
    #     print("Obs: %s" % str(self.get_obs()))
    #     print("Rate: %f" % self.rate)
    #     print("Sent: %d" % self.sent)
    #     print("Acked: %d" % self.acked)
    #     print("Lost: %d" % self.lost)
    #     print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.got_data = False

    def timeout(self):
        # placeholder
        pass

    def compute_mi_duration(self):
        return round(max(0.5 * self.rtt_sample,
                   MIN_PKTS_PER_MI / self.rate), 5)

    def create_mi(self, start_t, end_t):
        mi_id = self.mi_count
        # print(f"create mi {mi_id}, {start_t}, {end_t}")
        self.mi_cache[mi_id] = MonitorInterval(start_t, end_t)
        self.mi_count += 1

        return mi_id

    def has_finished_mi(self):
        has = False
        cur_time = self.net.get_cur_time()
        mi_id_list = list(sorted(self.mi_cache))
        for mi_id in mi_id_list:
            mi = self.mi_cache[mi_id]
            print(mi_id, len(mi.rtt_samples))
            if mi.is_finished(cur_time):
                smi = sender_obs.SenderMonitorInterval(
                    self.id, bytes_sent=mi.bytes_sent,
                    bytes_acked=mi.bytes_acked,
                    bytes_lost=mi.bytes_lost,
                    send_start=mi.first_packet_send_time,
                    send_end=mi.last_packet_send_time,
                    recv_start=mi.first_packet_ack_time,
                    recv_end=mi.last_packet_ack_time,
                    rtt_samples=mi.rtt_samples,
                    packet_size=BYTES_PER_PACKET)
                # print("mi {}: [{}, {}, {}, {}, {}, {}, {}, [{}, {}], {}]".format(
                #     mi_id, mi.bytes_sent, mi.bytes_acked,
                #     mi.bytes_lost, mi.first_packet_send_time,
                #     mi.last_packet_send_time,
                #     mi.first_packet_ack_time,
                #     mi.last_packet_ack_time,
                #     mi.rtt_samples[0], mi.rtt_samples[-1],
                #     BYTES_PER_PACKET))
                self.history.step(smi)
                # print(f'delete mi {mi_id}')
                del self.mi_cache[mi_id]
                has = True
        return has


class TCPCubicSender(Sender):
    """Mimic TCPCubic sender behavior.

    Args:
        rate
        path
        dest
        features
        cwnd: congestion window size. Unit: number of packets.
        history_len:
    """
    # used by TCP Cubic
    tcp_friendliness = 1
    fast_convergence = 1
    beta = 0.2
    C = 0.4

    # used by srtt
    ALPHA = 0.8
    BETA = 1.5

    def __init__(self, rate, path, dest, features, cwnd=10, history_len=10):
        super().__init__(rate, path, dest, features, cwnd, history_len)
        # TCP inital cwnd value is configured to 10 MSS. Refer to
        # https://developers.google.com/speed/pagespeed/service/tcp_initcwnd_paper.pdf

        # slow start threshold, arbitrarily high at start
        self.ssthresh = MAX_CWND
        self.pkt_loss_wait_time = 0
        self.use_cwnd = True
        self.rto = 3  # retransmission timeout (seconds)
        # initialize rto to 3s for waiting SYC packets of TCP handshake
        self.srtt = None # (self.ALPHA * self.srtt) + (1 - self.ALPHA) * rtt)

        self.cubic_reset()

    def cubic_reset(self):
        self.W_last_max = 0
        self.epoch_start = 0
        self.origin_point = 0
        self.dMin = 0
        self.W_tcp = 0  # used in tcp friendliness
        self.K = 0
        # self.acked = 0 # TODO is this one used in Cubic?
        self.ack_cnt = 0

        # In standard, TCP cwnd_cnt is an additional state variable that tracks
        # the number of segments acked since the last cwnd increment; cwnd is
        # incremented only when cwnd_cnt > cwnd; then, cwnd_cnt is set to 0.
        # initialie to 0
        self.cwnd_cnt = 0
        self.pkt_loss_wait_time = 0

    def apply_rate_delta(self, delta):
        # place holder
        #  do nothing
        pass

    def apply_cwnd_delta(self, delta):
        # place holder
        #  do nothing
        pass

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

        # Added by Zhengxu Xia
        if self.dMin:
            self.dMin = min(self.dMin, rtt)
        else:
            self.dMin = rtt
        if self.cwnd <= self.ssthresh:  # in slow start region
            # print("slow start, inc cwnd by 1")
            self.cwnd += 1
        else:  # in congestion avoidance or max bw probing region
            cnt = self.cubic_update()
            if self.cwnd_cnt > cnt:
                # print("in congestion avoidance, inc cwnd by 1")
                self.cwnd += 1
                self.cwnd_cnt = 0
            else:
                # print("in congestion avoidance, inc cwnd_cnt by 1")
                self.cwnd_cnt += 1
        if not self.rtt_samples and not self.prev_rtt_samples:
            print(self.srtt)
            raise RuntimeError("prev_rtt_samples is empty. TCP session is not constructed successfully!")
        elif not self.rtt_samples:
            avg_sampled_rtt = float(np.mean(np.array(self.prev_rtt_samples)))
        else:
            avg_sampled_rtt = float(np.mean(np.array(self.rtt_samples)))
        self.rate = self.cwnd / avg_sampled_rtt
        if self.pkt_loss_wait_time > 0:
            self.pkt_loss_wait_time -= 1

        # TODO: update RTO
        if self.srtt is None:
            self.srtt = rtt
        else:
            self.srtt = (self.ALPHA * self.srtt) + (1 - self.ALPHA) * rtt
        self.rto = max(1, min(self.BETA * self.srtt, 60))

    def on_packet_lost(self, rtt):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

        if self.pkt_loss_wait_time <= 0:
            self.epoch_start = 0
            if self.cwnd < self.W_last_max and self.fast_convergence:
                self.W_last_max = self.cwnd * (2 - self.beta) / 2
            else:
                self.W_last_max = self.cwnd
            old_cwnd = self.cwnd
            self.cwnd = self.cwnd * (1 - self.beta)
            self.ssthresh = self.cwnd
            # print("packet lost: cwnd change from", old_cwnd, "to", self.cwnd)
            if not self.rtt_samples and not self.prev_rtt_samples:
                raise RuntimeError("prev_rtt_samples is empty. TCP session is not constructed successfully!")
            elif not self.rtt_samples:
                avg_sampled_rtt = float(np.mean(np.array(self.prev_rtt_samples)))
            else:
                avg_sampled_rtt = float(np.mean(np.array(self.rtt_samples)))
            self.rate = self.cwnd / avg_sampled_rtt
            self.pkt_loss_wait_time = int(self.cwnd)

    def cubic_update(self):
        self.ack_cnt += 1
        # assume network current time is tcp_time_stamp
        assert self.net is not None
        tcp_time_stamp = self.net.get_cur_time()
        if self.epoch_start <= 0:
            self.epoch_start = tcp_time_stamp # TODO: check the unit of time
            if self.cwnd < self.W_last_max:
                self.K = np.cbrt((self.W_last_max - self.cwnd)/self.C)
                self.origin_point = self.W_last_max
            else:
                self.K = 0
                self.origin_point = self.cwnd
            self.ack_cnt = 1
            self.W_tcp = self.cwnd
        t = tcp_time_stamp + self.dMin - self.epoch_start
        target = self.origin_point + self.C * (t - self.K)**3
        if target > self.cwnd:
            cnt = self.cwnd / (target - self.cwnd)
        else:
            cnt = 100 * self.cwnd
        # TODO: call friendliness
        return cnt

    def reset(self):
        super().reset()
        self.cubic_reset()
        self.rto = 3  # retransmission timeout (seconds)
        # initialize rto to 3s for waiting SYC packets of TCP handshake
        self.srtt = None # (self.ALPHA * self.srtt) + (1 - self.ALPHA) * rtt)

    def timeout(self):
        # if self.pkt_loss_wait_time <= 0:
        # Refer to https://tools.ietf.org/html/rfc8312#section-4.7
        # self.ssthresh = max(int(self.bytes_in_flight / BYTES_PER_PACKET / 2), 2)
        self.sshthresh = self.cwnd * (1 - self.beta)
        self.cwnd = 1
        if not self.rtt_samples and not self.prev_rtt_samples:
            print(self.srtt)
            # raise RuntimeError("prev_rtt_samples is empty. TCP session is not constructed successfully!")
            avg_sampled_rtt = self.srtt
        elif not self.rtt_samples:
            avg_sampled_rtt = float(np.mean(np.array(self.prev_rtt_samples)))
        else:
            avg_sampled_rtt = float(np.mean(np.array(self.rtt_samples)))
        self.rate = self.cwnd / avg_sampled_rtt
        self.cubic_reset()
        # self.pkt_loss_wait_time = int(self.cwnd)
        # print('timeout rate', self.rate, self.cwnd)
        # return True

    def cubic_tcp_friendliness(self):
        raise NotImplementedError
