from enum import Enum
import math
import random
from simulator.constants import BYTES_PER_PACKET

from simulator.network_simulator.sender import Sender
from simulator.network_simulator import packet

# A constant specifying the minimum gain value that will
# allow the sending rate to double each round (2/ln(2) ~= 2.89), used
# in Startup mode for both BBR.pacing_gain and BBR.cwnd_gain.
BBR_HIGH_GAIN = 2.89
BTLBW_FILTER_LEN = 10 # packet-timed round trips.

RTPROP_FILTER_LEN = 10 # seconds
PROBE_RTT_DURATION = 10 # seconds
INITIAL_CWND = 10
BBR_MIN_PIPE_CWND = 4 # packets
BBR_GAIN_CYCLE_LEN = 8


class BBRPacket(packet.Packet):
    def __init__(self, ts: float, sender: Sender, pkt_id: int):
        super().__init__(ts, sender, pkt_id)
        self.delivered = 0
        self.delivered_time = 0.0
        self.first_sent_time = 0
        self.is_app_limited = False
        self.sent_time = 0


class BBRRateSample:
    def __init__(self):

        self.delivery_rate = 0 # The delivery rate sample (in most cases rs.delivered / rs.interval).
        self.is_app_limited = False # The P.is_app_limited from the most recent packet delivered; indicates whether the rate sample is application-limited.
        self.interval = 0 # The length of the sampling interval.
        self.delivered = 0# The amount of data marked as delivered over the sampling interval.
        self.prior_delivered = 0# The P.delivered count from the most recent packet delivered.
        self.prior_time = 0# The P.delivered_time from the most recent packet delivered.
        self.send_elapsed = 0# Send time interval calculated from the most recent packet delivered (see the "Send Rate" section above).
        self.ack_elapsed = 0# ACK time interval calculated from the most recent packet delivered (see the "ACK Rate" section above).

class BBRBtlBwFilter:
    def __init__(self, btlbw_filter_len: int):
        self.btlbw_filter_len = btlbw_filter_len
        self.cache = []

    def update(self, delivery_rate: float, round_count: int) -> None:
        # TODO: not finished yet
        self.cache.append(delivery_rate)

    def get_btlbw(self) -> float:
        return max(self.cache)


class BBRMode(Enum):
    BBR_STARTUP = "BBR_STARTUP"  # ramp up sending rate rapidly to fill pipe
    BBR_DRAIN = "BBR_DRAIN"  # drain any queue created during startup
    BBR_PROBE_BW = "BBR_PROBE_BW"  # discover, share bw: pace around estimated bw
    BBR_PROBE_RTT = "BBR_PROBE_RTT"  # cut inflight to min to probe min_rtt


class BBRSender(Sender):
    """

    Reference:
        https://datatracker.ietf.org/doc/html/draft-cardwell-iccrg-bbr-congestion-control
        https://datatracker.ietf.org/doc/html/draft-cheng-iccrg-delivery-rate-estimation#section-3.1.3
    """

    def __init__(self, sender_id: int, dest: int):
        super().__init__(sender_id, dest)

        self.cwnd = INITIAL_CWND

        self.rs = BBRRateSample()
        self.btlbw = 0

        self.init()


        self.app_limited_until = 0
        self.next_send_time = 0

        # Connection state used to estimate rates
        self.delivered = 0 # The total amount of data (tracked in octets or in packets) delivered so far over the lifetime of the transport connection.

        self.delivered_time = 0# The wall clock time when C.delivered was last updated.

        self.first_sent_time = 0# If packets are in flight, then this holds the send
        # time of the packet that was most recently marked as delivered.  Else,
        # if the connection was recently idle, then this holds the send time of
        # most recently sent packet.

        self.app_limited = 0 # The index of the last transmitted packet marked as
        # application-limited, or 0 if the connection is not currently
        #    application-limited.


        self.write_seq = 0# The data sequence number one higher than that of the
        # last octet queued for transmission in the transport layer write
        # buffer.

        self.pending_transmissions = 0 # The number of bytes queued for transmission
        # on the sending host at layers lower than the transport layer (i.e.
        # network layer, traffic shaping layer, network device layer).

        self.lost_out = 0# The number of packets in the current outstanding window
        # that are marked as lost.

        self.retrans_out = 0# The number of packets in the current outstanding
        # window that are being retransmitted.

        self.pipe = 0# The sender's estimate of the number of packets outstanding in
        # the network; i.e. the number of packets in the current outstanding
        # window that are being transmitted or retransmitted and have not been
        # SACKed or marked lost (e.g. "pipe" from [RFC6675]).

    def init(self):
        self.btlbw_filter = BBRBtlBwFilter(BTLBW_FILTER_LEN) # init_windowed_max_filter(filter=BBR.BtlBwFilter, value=0, time=0)
        self.rtprop = SRTT ? SRTT: math.inf
        self.rtprop_stamp = self.get_cur_time() # Now()
        self.rtprop_expired = False
        self.probe_rtt_done_stamp = 0
        self.probe_rtt_round_done = False
        self.packet_conservation = False
        self.prior_cwnd = 0
        self.idle_restart = False
        self.init_round_counting()
        self.init_full_pipe()
        self.init_pacing_rate()
        self.enter_startup()


    def init_round_counting(self):
        self.next_round_delivered = 0
        self.round_start = False
        self.round_count = 0

    def init_full_pipe(self):
        self.filled_pipe = False
        self.full_bw = 0
        self.full_bw_count = 0

    def init_pacing_rate(self):
        # nominal_bandwidth = InitialCwnd / (SRTT ? SRTT : 1ms)
        # InitialCwnd = 10 packets as Cubic
        nominal_bandwidth = self.cwnd / (SRTT ? SRTT : 1ms)
        self.pacing_rate =  self.pacing_gain * nominal_bandwidth


    def set_pacing_rate_with_gain(self, pacing_gain):
        rate = pacing_gain * self.btlbw
        if self.filled_pipe or rate > self.pacing_rate:
            self.pacing_rate = rate

    def set_pacing_rate(self):
        self.set_pacing_rate_with_gain(self.pacing_gain)

    def enter_startup(self):
        self.state = BBRMode.BBR_STARTUP
        self.pacing_gain = BBR_HIGH_GAIN
        self.cwnd_gain = BBR_HIGH_GAIN

    def check_full_pipe(self):
        if self.filled_pipe or not self.round_start or rs.is_app_limited:
            return  # no need to check for a full pipe now
        if self.btlbw >= self.full_bw * 1.25:  # BBR.BtlBw still growing?
            self.full_bw = self.btlbw    # record new baseline level
            self.full_bw_count = 0
            return
        self.full_bw_count += 1   # another round w/o much growth
        if self.full_bw_count >= 3:
             self.filled_pipe = True

    def update_round(self, pkt):
        self.delivered += pkt.pkt_size
        if pkt.delivered >= self.next_round_delivered:
            self.next_round_delivered = self.delivered
            self.round_count += 1
            self.round_start = True
        else:
            self.round_start = False

    def update_btlbw(self, pkt):
        self.update_round(pkt)
        if self.rs.delivery_rate >= self.btlbw or not self.rs.is_app_limited:
            self.btlbw_filter.update(self.rs.delivery_rate, self.round_count)
            self.btlbw = self.btlbw_filter.get_btlbw()
            # self.btl_bw = update_windowed_max_filter(
            #              filter=BBR.BtlBwFilter,
            #              value=self.rs.delivery_rate,
            #              time=self.round_count,
            #              window_length=BtlBwFilterLen)

    def update_rtprop(self, pkt):
        self.rtprop_expired = self.get_cur_time() > self.rtprop_stamp + RTPROP_FILTER_LEN
        if (pkt.rtt >= 0 and (pkt.rtt <= self.RTprop or self.rtprop_expired)):
            self.RTprop = pkt.rtt
            self.rtprop_stamp = self.get_cur_time()

    def set_send_quantum(self):
        if self.pacing_rate < 1.2: #  1.2Mbps
            self.send_quantum = 1 * BYTES_PER_PACKET # MSS
        elif self.pacing_rate < 24 : #Mbps
          self.send_quantum  = 2 * BYTES_PER_PACKET # MSS
        else:
            self.send_quantum  = min(self.pacing_rate * 1ms, 64KBytes)


    def inflight(self, gain):
        if self.rtprop > 0 and math.isinf(self.rtprop):
            return INITIAL_CWND # no valid RTT samples yet
        quanta = 3 * self.send_quantum
        estimated_bdp = self.btlbw * self.rtprop
        return gain * estimated_bdp + quanta

    def update_target_cwnd(self):
        self.target_cwnd = self.inflight(self.cwnd_gain)


    def modulate_cwnd_for_probe_rtt(self):
        if (self.state == BBRMode.BBR_PROBE_RTT):
            self.cwnd = min(self.cwnd, BBR_MIN_PIPE_CWND)

    def save_cwnd(self):
        if not InLossRecovery() and self.state != BBRMode.BBR_PROBE_RTT:
            return self.cwnd
        else:
            return max(self.prior_cwnd, self.cwnd)

    def restore_cwnd(self):
        self.BBcwnd = max(self.cwnd, self.prior_cwnd)

    def set_cwnd(self):
        self.update_target_cwnd()
        self.modulate_cwnd_for_recovery()
        if not self.packet_conservation:
            if self.filled_pipe:
                self.cwnd = min(self.cwnd + packets_delivered, self.target_cwnd)
            elif self.cwnd < self.target_cwnd or self.delivered < INITIAL_CWND:
                self.cwnd = self.cwnd + packets_delivered
            self.cwnd = max(self.cwnd, BBR_MIN_PIPE_CWND)

        self.modulate_cwnd_for_probe_rtt()

    def modulate_cwnd_for_recovery(self):
        pass



    def enter_drain(self):
        self.state = BBRMode.BBR_DRAIN
        self.pacing_gain = 1 / BBR_HIGH_GAIN  # pace slowly
        self.cwnd_gain = BBR_HIGH_GAIN    # maintain cwnd TODO: is this a bug?


    def check_drain(self):
        if self.state == BBRMode.BBR_STARTUP and self.filled_pipe:
            self.enter_drain()
        if self.state == BBRMode.BBR_DRAIN and packets_in_flight <= self.inflight(1.0):
            self.enter_probe_bw()  # we estimate queue is drained

    def enter_probe_bw(self):
        self.state = BBRMode.BBR_PROBE_BW
        self.pacing_gain = 1
        self.cwnd_gain = 2
        self.cycle_index = BBR_GAIN_CYCLE_LEN - 1 - random.randint(0, 6)
        self.advance_cycle_phase()

    def check_cycle_phase(self):
        if self.state == BBRMode.BBR_PROBE_BW and self.is_next_cycle_phase():
            self.advance_cycle_phase()

    def advance_cycle_phase(self):
        self.cycle_stamp = self.get_cur_time()
        self.cycle_index = (self.cycle_index + 1) % BBR_GAIN_CYCLE_LEN
        pacing_gain_cycle = [5/4, 3/4, 1, 1, 1, 1, 1, 1]
        self.pacing_gain = pacing_gain_cycle[self.cycle_index]

    def is_next_cycle_phase(self):
        is_full_length = (self.get_cur_time() - self.cycle_stamp) > self.rtprop
        if (self.pacing_gain == 1):
            return is_full_length
        if (self.pacing_gain > 1):
            return is_full_length and (packets_lost > 0 or prior_inflight >= self.inflight(self.pacing_gain))
        else:  #  (BBR.pacing_gain < 1)
            return is_full_length or prior_inflight <= self.inflight(1)

    def handle_restart_from_idle(self):
        if packets_in_flight == 0 and self.app_limited:
            self.idle_start = True
        if self.state == BBRMode.BBR_PROBE_BW:
           self.set_pacing_rate_with_gain(1)


    def check_probe_rtt(self):
        if self.state != BBRMode.BBR_PROBE_RTT and self.rtprop_expired and not self.idle_restart:
            self.enter_probe_rtt()
            self.save_cwnd()
            self.probe_rtt_done_stamp = 0
        if self.state == BBRMode.BBR_PROBE_RTT:
            self.handle_probe_rtt()
        self.idle_restart = False

    def enter_probe_rtt(self):
        self.state = BBRMode.BBR_PROBE_RTT
        self.pacing_gain = 1
        self.cwnd_gain = 1

    def handle_probe_rtt(self):
        # Ignore low rate samples during ProbeRTT: */
        self.app_limited = (BW.delivered + packets_in_flight) ? : 1
        if self.probe_rtt_done_stamp == 0 and packets_in_flight <= BBR_MIN_PIPE_CWND:
            self.probe_rtt_done_stamp = self.get_cur_time() + PROBE_RTT_DURATION
            self.probe_rtt_round_done = False
            self.next_round_delivered = self.delivered
        elif self.probe_rtt_done_stamp != 0:
            if self.round_start:
                self.probe_rtt_round_done = True
            if self.probe_rtt_round_done and self.get_cur_time() > self.probe_rtt_done_stamp:
                self.rtprop_stamp = self.get_cur_time()
                self.restore_cwnd()
                self.exit_probe_rtt()

    def exit_probe_rtt(self):
        if self.filled_pipe:
            self.enter_probe_bw()
        else:
            self.enter_startup()

    def update_on_ack(self):
       self.update_model_and_state()
       self.update_control_parameters()

    def update_model_and_state(self):
        self.update_btlbw()
        self.check_cycle_phase()
        self.check_full_pipe()
        self.check_drain()
        self.update_rtprop()
        self.check_probe_rtt()

    def update_control_parameters(self):
       self.set_pacing_rate()
       self.set_send_quantum()
       self.set_cwnd()

    def on_transmit(self):
        self.handle_restart_from_idle()

    def send_packet(self, pkt):
        if self.pipe == 0:
            self.first_sent_time = self.get_cur_time()
            self.delivered_time = self.get_cur_time()
        pkt.first_sent_time = self.first_sent_time
        pkt.delivered_time = self.delivered_time
        pkt.delivered = self.delivered
        pkt.is_app_limited = (self.app_limited != 0)

    # Upon receiving ACK, fill in delivery rate sample rs.
    def generate_rate_sample(self, rs):
        # TODO: uncomment this one
        # for each newly SACKed or ACKed packet P:
        #     self.update_rate_sample(P, rs)

        # Clear app-limited field if bubble is ACKed and gone.
        if self.app_limited and self.delivered > self.app_limited:
            self.app_limited = 0

        if rs.prior_time == 0:
            return False  # nothing delivered on this ACK

       # Use the longer of the send_elapsed and ack_elapsed
        rs.interval = max(rs.send_elapsed, rs.ack_elapsed)

        rs.delivered = self.delivered - rs.prior_delivered

        # Normally we expect interval >= MinRTT.
        # Note that rate may still be over-estimated when a spuriously
        # retransmitted skb was first (s)acked because "interval"
        # is under-estimated (up to an RTT). However, continuously
        # measuring the delivery rate during loss recovery is crucial
        # for connections suffer heavy or prolonged losses.
        #
        if rs.interval <  MinRTT(tp):
            rs.interval = -1
            return False  # no reliable sample

        if (rs.interval != 0):
            rs.delivery_rate = rs.delivered / rs.interval

        return True  # we filled in rs with a rate sample */

    # Update rs when packet is SACKed or ACKed. */
    def update_rate_sample(self, pkt, rs):
        if pkt.delivered_time == 0:
            return # P already SACKed

        self.delivered += pkt.data_length
        self.delivered_time = self.get_cur_time()

        # Update info using the newest packet:
        if (pkt.delivered > rs.prior_delivered):
            rs.prior_delivered = pkt.delivered
            rs.prior_time = pkt.delivered_time
            rs.is_app_limited = pkt.is_app_limited
            rs.send_elapsed = pkt.sent_time - pkt.first_sent_time
            rs.ack_elapsed = self.delivered_time - pkt.delivered_time
            self.first_sent_time = pkt.sent_time

        # Mark the packet as delivered once it's SACKed to
        # avoid being used again when it's cumulatively acked.

        pkt.delivered_time = 0

    # def on_packet_sent(self, pkt: BBRPacket) -> None:
    #     if not self.net:
    #         raise RuntimeError("network is not registered in sender.")
    #     bdp = BtlBwFilter.currentMax * RTpropFilter.currentMin
    #     if self.bytes_in_flight >= self.cwnd_gain * bdp:
    #         # wait for ack or timeout
    #         return
    #     if self.net.get_cur_time() >= self.next_send_time:
    #         # packet = nextPacketToSend() # assume always a packet to send from app
    #         if not pkt:
    #             self.app_limited_until = self.bytes_in_flight
    #             return
    #         pkt.app_limited = self.app_limited_until > 0
    #         pkt.send_time = self.net.get_cur_time()
    #         pkt.delivered = self.delivered
    #         pkt.delivered_time = self.delivered_time
    #         # ship(packet) # no need to do this in the simulator.
    #         super().on_packet_sent(pkt)
    #         self.next_send_time = self.net.get_cur_time() + pkt.pkt_size / \
    #             (self.pacing_gain * BtlBwFilter.currentMax)
    #         next_pkt = BBRPacket(self.next_send_time, self, 0)
    #         self.net.add_packet(self.next_send_time, next_pkt)
    #     # timerCallbackAt(send, nextSendTime)
    #     # TODO: potential bug here if previous call return at if inflight < cwnd
    #
    # def on_packet_acked(self, pkt: BBRPacket) -> None:
    #     if not self.net:
    #         raise RuntimeError("network is not registered in sender.")
    #     super().on_packet_acked(pkt)
    #     RTpropFilter = None
    #     BtlBwFilter = None
    #     rtt = pkt.cur_latency
    #     self.update_min_filter(RTpropFilter, rtt)
    #     self.delivered += pkt.pkt_size
    #     self.delivered_time = pkt.ts
    #     delivery_rate = (self.delivered - pkt.delivered) / \
    #         (self.net.get_cur_time() - pkt.delivered_time)
    #     if delivery_rate > BtlBwFilter.currentMax or not pkt.app_limited:
    #         self.update_max_filter(BtlBwFilter, delivery_rate)
    #     if self.app_limited_until > 0:
    #         self.app_limited_until -= pkt.pkt_size
    #
    # def on_packet_lost(self, pkt: BBRPacket) -> None:
    #     if not self.net:
    #         raise RuntimeError("network is not registered in sender.")
    #     super().on_packet_lost(pkt)

    # def update_max_filter(self, BtlBwFilter, delivery_rate: float):
    #     raise NotImplementedError
    #
    # def update_min_filter(self, RTpropFilter, rtt: float):
    #     raise NotImplementedError

    def reset(self):
        raise NotImplementedError
