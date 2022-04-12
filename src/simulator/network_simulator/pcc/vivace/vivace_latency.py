import csv
import os
import random
from enum import Enum
from typing import Tuple

import numpy as np

from common.utils import pcc_aurora_reward
from plot_scripts.plot_packet_log import plot
from plot_scripts.plot_time_series import plot as plot_simulation_log
from simulator.network_simulator.constants import (
    BITS_PER_BYTE,
    BYTES_PER_PACKET,
)
from simulator.network_simulator.link import Link
from simulator.network_simulator.network import Network
from simulator.network_simulator.sender import Sender
from simulator.network_simulator import packet
from simulator.trace import Trace
from simulator.network_simulator.pcc import monitor_interval_queue, utility_manager

FLAGS_max_rtt_fluctuation_tolerance_ratio_in_starting = 100.0
FLAGS_max_rtt_fluctuation_tolerance_ratio_in_decision_made = 1.0

kInitialRtt = 0.1
kInitialCwnd = 10
kNumIntervalGroupsInProbingPrimary = 3
kMinReliabilityRatio = 0.8
# Step size for rate change in PROBING mode.
kProbingStepSize = 0.05
# Base percentile step size for rate change in DECISION_MADE mode.
kDecisionMadeStepSize = 0.02
# Maximum percentile step size for rate change in DECISION_MADE mode.
kMaxDecisionMadeStepSize = 0.10

class UtilityInfo:
    def __init__(self, sending_rate: float, utility: float):
        self.sending_rate = sending_rate
        self.utility = utility


class PccSenderMode(Enum):
    # Initial phase of the connection. Sending rate gets doubled as
    # long as utility keeps increasing, and the sender enters
    # PROBING mode when utility decreases.
    STARTING = "STARTING"
    # Sender tries different sending rates to decide whether higher
    # or lower sending rate has greater utility. Sender enters
    # DECISION_MADE mode once a decision is made.
    PROBING = "PROBING"
    #  Sender keeps increasing or decreasing sending rate until
    #  utility decreases, then sender returns to PROBING mode.
    DECISION_MADE = "DECISION_MADE"


class RateChangeDirection(Enum):
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"


class VivaceLatencySender(Sender):
    def __init__(self, sender_id: int, dest: int):
        super().__init__(sender_id, dest)

        self.pacing_rate = kInitialCwnd * BYTES_PER_PACKET * BITS_PER_BYTE / kInitialRtt
        self.mi_q = monitor_interval_queue.MonitorIntervalQueue(self)
        self.monitor_duration = 0.0
        self.latest_rtt = 0.0
        self.avg_rtt = 0.0
        self.min_rtt = 0.0
        self.rtt_deviation = 0.0
        self.min_rtt_deviation = 0.0
        self.mode = PccSenderMode.STARTING
        self.has_seen_valid_rtt = False
        self.rounds = 1
        self.conn_start_time = -1  # 0.0
        self.rtt_on_inflation_start = 0.0
        self.latest_sent_timestamp = 0.0
        self.latest_ack_timestamp = 0.0
        self.latest_utility = 0.0
        self.utility_manager = utility_manager.UtilityManager()
        self.cwnd = 0

    def on_packet_sent(self, pkt: "packet.Packet") -> None:
        if self.conn_start_time == 0.0:
            self.conn_start_time = pkt.sent_time
            self.latest_sent_timestamp = pkt.sent_time

        if self.create_new_interval(pkt.sent_time):
            self.maybe_set_sending_rate()
            # Set the monitor duration to 1.0 of min rtt.
            self.monitor_duration = self.min_rtt * 1.0

            is_useful = self.create_useful_interval()
            if is_useful:
                self.mi_q.enqueue_new_monitor_interval(
                    self.pacing_rate, is_useful,
                    self.get_max_rtt_fluctuation_tolerance(), self.avg_rtt)
            else:
                self.mi_q.enqueue_new_monitor_interval(
                    self.get_sending_rate_for_non_useful_interval(),
                    is_useful, self.get_max_rtt_fluctuation_tolerance(), self.avg_rtt)

        super().on_packet_sent(pkt)
        self.mi_q.on_packet_sent(
            pkt, pkt.sent_time - self.latest_sent_timestamp)
        self.latest_sent_timestamp = pkt.sent_time

    def on_packet_acked(self, pkt: "packet.Packet") -> None:
        if self.latest_ack_timestamp == 0.0:
            self.latest_ack_timestamp = pkt.ts

          # if (exit_starting_based_on_sampled_bandwidth_) {
          #   // UpdateBandwidthSampler(event_time, acked_packets, lost_packets);
          # }

        ack_interval = 0.0
        if pkt.rtt:
            ack_interval = pkt.ts - self.latest_ack_timestamp
            self.update_rtt(pkt.ts, pkt.rtt)

        avg_rtt = self.avg_rtt
        if not self.has_seen_valid_rtt:
            self.has_seen_valid_rtt = True
            # Update sending rate if the actual RTT is smaller than initial rtt
            # value in RttStats, so PCC can start with larger rate and ramp up
            # faster.
            if self.latest_rtt < kInitialRtt:
                self.pacing_rate = self.pacing_rate * \
                    (kInitialRtt / self.latest_rtt)

        if self.mode == PccSenderMode.STARTING and self.check_for_rtt_inflation():
            # Directly enter PROBING when rtt inflation already exceeds the
            # tolerance ratio, so as to reduce packet losses and mitigate rtt
            # inflation.
            self.mi_q.on_rtt_inflation_in_starting()
            self.enter_probing()
            return

        self.mi_q.on_packet_acked(pkt, ack_interval, self.latest_rtt, avg_rtt,
                                  self.min_rtt)
        return super().on_packet_acked(pkt)

    def on_packet_lost(self, pkt: "packet.Packet") -> None:
        self.mi_q.on_packet_lost(pkt, self.avg_rtt, self.min_rtt)
        return super().on_packet_lost(pkt)

    def create_new_interval(self, event_time: float) -> bool:
        """Determine whether to create a new MI."""
        if self.mi_q.empty():
            return True
        #  Do not start new monitor interval before latest RTT is available.
        if self.latest_rtt == 0.0:
            return False

        # Start a (useful) interval if latest RTT is available but the queue
        # does not contain useful interval.
        if self.mi_q.num_useful_intervals == 0:
            return True

        cur_mi = self.mi_q.current()
        # Do not start new interval if there is non-useful interval in the
        # tail.
        if not cur_mi.is_useful:
            return False

        # Do not start new interval until current useful interval has enough
        # reliable RTT samples, and its duration exceeds the monitor_duration.
        if not cur_mi.has_enough_reliable_rtt or event_time - cur_mi.first_packet_sent_time < self.monitor_duration:
            return False

        if cur_mi.num_reliable_rtt / len(cur_mi.packet_rtt_samples) > kMinReliabilityRatio:
            # Start a new interval if current useful interval has an RTT
            # reliability ratio larger than kMinReliabilityRatio.
            return True
        elif cur_mi.is_monitor_duration_extended:
            # Start a new interval if current useful interval has been extended
            # once.
            return True
        else:
            # Extend the monitor duration if the current useful interval has
            # not been extended yet, and its RTT reliability ratio is lower
            # than kMinReliabilityRatio
            self.monitor_duration = self.monitor_duration * 2.0
            self.mi_q.extend_current_interval()
            return False

    def create_useful_interval(self) -> bool:
        if self.avg_rtt == 0.0:
            # Create non useful intervals upon starting a connection, until there is
            # valid rtt stats.
            assert self.mode == PccSenderMode.STARTING
            return False

        # In STARTING and DECISION_MADE mode, there should be at most one useful
        # intervals in the queue; while in PROBING mode, there should be at most
        # 2 * GetNumIntervalGroupsInProbing().
        max_num_useful = 2 * \
            self.get_num_interval_groups_in_probing(
            ) if self.mode == PccSenderMode.PROBING else 1
        return self.mi_q.num_useful_intervals < max_num_useful

    def get_num_interval_groups_in_probing(self):
        return kNumIntervalGroupsInProbingPrimary

    def maybe_set_sending_rate(self):
        if (self.mode != PccSenderMode.PROBING or (
            self.mi_q.num_useful_intervals == 2 * self.get_num_interval_groups_in_probing()
                and not self.mi_q.current().is_useful)):
            # Do not change sending rate when (1) current mode is STARTING or
            # DECISION_MADE (since sending rate is already changed in
            # OnUtilityAvailable), or (2) more than 2 *
            # GetNumIntervalGroupsInProbing() intervals have been created in
            # PROBING mode.
            return

        if self.mi_q.num_useful_intervals != 0:
            # Restore central sending rate.
            self.restore_central_sending_rate()

            if self.mi_q.num_useful_intervals == 2 * self.get_num_interval_groups_in_probing():
                # This is the first not useful monitor interval, its sending
                # rate is the central rate.
                return

        # Sender creates several groups of monitor intervals. Each group comprises an
        # interval with increased sending rate and an interval with decreased sending
        # rate. Which interval goes first is randomly decided.
        if self.mi_q.num_useful_intervals % 2 == 0:
            self.direction = RateChangeDirection.INCREASE if random.randint(
                0, 9) % 2 == 1 else RateChangeDirection.DECREASE
        else:
            self.direction = RateChangeDirection.DECREASE if (
                self.direction == RateChangeDirection.INCREASE) else RateChangeDirection.INCREASE

        if self.direction == RateChangeDirection.INCREASE:
            self.pacing_rate = self.pacing_rate * (1 + kProbingStepSize)
        else:
            self.pacing_rate = self.pacing_rate * (1 - kProbingStepSize)

    def restore_central_sending_rate(self):
        if self.mode == PccSenderMode.STARTING:
            # The sending rate upon exiting STARTING is set separately. This
            # function should not be called while sender is in STARTING mode.
            assert False
        elif self.mode == PccSenderMode.PROBING:
            # Change sending rate back to central probing rate.
            if self.mi_q.current().is_useful:
                if self.direction == RateChangeDirection.INCREASE:
                    self.pacing_rate = self.pacing_rate * \
                        (1.0 / (1 + kProbingStepSize))
                else:
                    self.pacing_rate = self.pacing_rate * \
                        (1.0 / (1 - kProbingStepSize))
        elif self.mode == PccSenderMode.DECISION_MADE:
            if self.direction == RateChangeDirection.INCREASE:
                self.pacing_rate = self.pacing_rate * \
                    (1.0 / (1 + min(self.rounds *
                     kDecisionMadeStepSize, kMaxDecisionMadeStepSize)))
            else:
                self.pacing_rate = self.pacing_rate * \
                    (1.0 / (1 - min(self.rounds *
                     kDecisionMadeStepSize, kMaxDecisionMadeStepSize)))

    def check_for_rtt_inflation(self) -> bool:
        if self.mi_q.empty() or self.mi_q.front().rtt_on_monitor_start == 0.0 or self.latest_rtt <= self.avg_rtt:
            # RTT is not inflated if latest RTT is no larger than smoothed RTT.
            self.rtt_on_inflation_start = 0.0
            return False

        # Once the latest RTT exceeds the smoothed RTT, store the corresponding
        # smoothed RTT as the RTT at the start of inflation. RTT inflation will
        # continue as long as latest RTT keeps being larger than smoothed RTT.
        if self.rtt_on_inflation_start == 0.0:
            self.rtt_on_inflation_start = self.avg_rtt

        max_inflation_ratio = 1 + self.get_max_rtt_fluctuation_tolerance()
        rtt_on_monitor_start = self.mi_q.current().rtt_on_monitor_start
        is_inflated = max_inflation_ratio * rtt_on_monitor_start < self.avg_rtt

        if is_inflated:
            # RTT is inflated by more than the tolerance, and early termination
            # will be triggered. Reset the rtt on inflation start.
            self.rtt_on_inflation_start = 0.0

        return is_inflated

    def get_max_rtt_fluctuation_tolerance(self) -> float:
        if self.mode == PccSenderMode.STARTING:
            return FLAGS_max_rtt_fluctuation_tolerance_ratio_in_starting
        return FLAGS_max_rtt_fluctuation_tolerance_ratio_in_decision_made

        # if (FLAGS_enable_rtt_deviation_based_early_termination) {
        #   float tolerance_gain = 0.0;
        #   if (mode_ == STARTING) {
        #     tolerance_gain = FLAGS_rtt_fluctuation_tolerance_gain_in_starting;
        #   } else if (mode_ == PROBING) {
        #     tolerance_gain = FLAGS_rtt_fluctuation_tolerance_gain_in_probing;
        #   } else {
        #     tolerance_gain = FLAGS_rtt_fluctuation_tolerance_gain_in_decision_made;
        #   }
        #   tolerance_ratio = std::min(
        #       tolerance_ratio,
        #       tolerance_gain *
        #           static_cast<float>(rtt_deviation_.ToMicroseconds()) /
        #           static_cast<float>((avg_rtt_.IsZero()? kInitialRtt : avg_rtt_)
        #                                  .ToMicroseconds()));
        # }
        #
        # return tolerance_ratio;

    def enter_probing(self):
        if self.mode == PccSenderMode.STARTING:
            # Fall back to the minimum between halved sending rate and
            # max bandwidth * (1 - 0.05) if there is valid bandwidth sample.
            # Otherwise, simply halve the current sending rate.
            self.pacing_rate = self.pacing_rate * 0.5
        elif self.mode == PccSenderMode.DECISION_MADE or self.mode == PccSenderMode.PROBING:
            # Reset sending rate to central rate when sender does not have enough
            # data to send more than 2 * GetNumIntervalGroupsInProbing() intervals.
            self.restore_central_sending_rate()

        if self.mode == PccSenderMode.PROBING:
            self.rounds += 1
            return

        self.mode = PccSenderMode.PROBING
        self.rounds = 1

    def get_sending_rate_for_non_useful_interval(self) -> float:
        if self.mode == PccSenderMode.STARTING:
            # Use halved sending rate for non-useful intervals in STARTING.
            return self.pacing_rate * 0.5
        elif self.mode == PccSenderMode.PROBING:
            # Use the smaller probing rate in PROBING.
            return self.pacing_rate * (1 - kProbingStepSize)
        elif self.mode == PccSenderMode.DECISION_MADE:
            # Use the last (smaller) sending rate if the sender is increasing
            # sending rate in DECISION_MADE. Otherwise, use the current sending
            # rate.
            if self.direction == RateChangeDirection.DECREASE:
                return self.pacing_rate
            return self.pacing_rate * (1.0 / (1 + min(self.rounds * kDecisionMadeStepSize, kMaxDecisionMadeStepSize)))
        assert False

    def update_rtt(self, event_time: float, rtt: float):
        self.latest_rtt = rtt
        if self.rtt_deviation == 0:
            self.rtt_deviation = rtt / 2
        else:
            self.rtt_deviation = 0.75 * self.rtt_deviation + \
                0.25 * abs(self.avg_rtt - rtt)
        if self.min_rtt_deviation == 0 or self.rtt_deviation < self.min_rtt_deviation:
            self.min_rtt_deviation = self.rtt_deviation
        if self.avg_rtt == 0:
            self.avg_rtt = rtt
        else:
            self.avg_rtt * 0.875 + rtt * 0.125
        if self.min_rtt == 0 or rtt < self.min_rtt:
            self.min_rtt = rtt

        self.latest_ack_timestamp = event_time

    def on_utility_available(self, useful_intervals, event_time: float):
        # Calculate the utilities for all available intervals.
        utility_info = []
        for mi in useful_intervals:
            utility_info.append(UtilityInfo(mi.sending_rate,
                    self.utility_manager.calculate_utility(
                        mi, event_time - self.conn_start_time)))

        if self.mode == PccSenderMode.STARTING:
            assert len(utility_info) == 1
            if (utility_info[0].utility > self.latest_utility):
                # Stay in STARTING mode. Double the sending rate and update
                # latest_utility.
                self.pacing_rate = self.pacing_rate * 2
                self.latest_utility = utility_info[0].utility
                self.rounds += 1
            else:
                # Enter PROBING mode if utility decreases.
                self.enter_probing()
        elif self.mode == PccSenderMode.PROBING:
            if self.can_make_decision(utility_info):
                assert len(utility_info) == 2 * self.get_num_interval_groups_in_probing()
                # Enter DECISION_MADE mode if a decision is made.
                if utility_info[0].utility > utility_info[1].utility:
                    if utility_info[0].sending_rate > utility_info[1].sending_rate:
                        self.direction = RateChangeDirection.INCREASE
                    else:
                        self.direction = RateChangeDirection.DECREASE
                else:
                    if utility_info[0].sending_rate > utility_info[1].sending_rate:
                        self.direction = RateChangeDirection.DECREASE
                    else:
                        self.direciton = RateChangeDirection.INCREASE
                self.latest_utility = max(
                    utility_info[2*self.get_num_interval_groups_in_probing()-2].utility,
                    utility_info[2*self.get_num_interval_groups_in_probing()-1].utility)
                self.enter_decision_made()
            else:
                # Stays in PROBING mode.
                self.enter_probing()
        elif self.mode == PccSenderMode.DECISION_MADE:
            assert len(utility_info) == 1
            if (utility_info[0].utility > self.latest_utility):
                # Remain in DECISION_MADE mode. Keep increasing or decreasing
                # the sending rate.
                self.rounds += 1
                if self.direction == RateChangeDirection.INCREASE:
                    self.pacing_rate = self.pacing_rate * (1 + min(
                        self.rounds * kDecisionMadeStepSize, kMaxDecisionMadeStepSize))
                else:
                    self.pacing_rate = self.pacing_rate * (1 - min(
                        self.rounds * kDecisionMadeStepSize, kMaxDecisionMadeStepSize))

                self.latest_utility = utility_info[0].utility
            else:
                # Enter PROBING mode if utility decreases.
                self.enter_probing()

    def can_make_decision(self, utility_info):
        # Determine whether increased or decreased probing rate has better utility.
        # Cannot make decision if number of utilities are less than
        # 2 * GetNumIntervalGroupsInProbing(). This happens when sender does not have
        # enough data to send.
        if len(utility_info) < 2 * self.get_num_interval_groups_in_probing():
            return False

        increase = False
        # All the probing groups should have consistent decision. If not, directly
        # return false.
        i = 0
        while i < self.get_num_interval_groups_in_probing():
            if utility_info[2*i].utility > utility_info[2*i+1].utility:
                increase_i =  utility_info[2*i].sending_rate > utility_info[2*i+1].sending_rate
            else:
                increase_i = utility_info[2*i].sending_rate < utility_info[2*i+1].sending_rate
            if i == 0:
                increase = increase_i

            # Cannot make decision if groups have inconsistent results.
            if increase_i != increase:
                return False
            i += 1

        return True

    def enter_decision_made(self):
        assert self.mode == PccSenderMode.PROBING

        # Change sending rate from central rate based on the probing rate with
        # higher utility.
        if self.direction == RateChangeDirection.INCREASE:
            self.pacing_rate = self.pacing_rate * (1 + kProbingStepSize) * \
                    (1 + kDecisionMadeStepSize)
        else:
            self.pacing_rate = self.pacing_rate * (1 - kProbingStepSize) * \
                    (1 - kDecisionMadeStepSize)

        self.mode = PccSenderMode.DECISION_MADE
        self.rounds = 1

    def schedule_send(self, first_pkt: bool = False, on_ack: bool = False):
        assert self.net, "network is not registered in sender."
        if first_pkt:
            next_send_time = 0
        else:
            next_send_time = self.get_cur_time() + BYTES_PER_PACKET / self.pacing_rate
        next_pkt = packet.Packet(next_send_time, self, 0)
        self.net.add_packet(next_pkt)


class VivaceLatency:
    cc_name = 'vivace_latency'

    def __init__(self, record_pkt_log: bool = False):
        self.record_pkt_log = record_pkt_log

    def test(self, trace: Trace, save_dir: str, plot_flag: bool = False) -> Tuple[float, float]:
        """Test a network trace and return rewards.

        The 1st return value is the reward in Monitor Interval(MI) level and
        the length of MI is 1 srtt. The 2nd return value is the reward in
        packet level. It is computed by using throughput, average rtt, and
        loss rate in each 500ms bin of the packet log. The 2nd value will be 0
        if record_pkt_log flag is False.

        Args:
            trace: network trace.
            save_dir: where a MI level log will be saved if save_dir is a
                valid path. A packet level log will be saved if record_pkt_log
                flag is True and save_dir is a valid path.
        """

        links = [Link(trace), Link(trace)]
        senders = [VivaceLatencySender(0, 0)]
        net = Network(senders, links, self.record_pkt_log)

        rewards = []
        start_rtt = trace.get_delay(0) * 2 / 1000
        run_dur = start_rtt
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            f_sim_log = open(os.path.join(save_dir, '{}_simulation_log.csv'.format(
                self.cc_name)), 'w', 1)
            writer = csv.writer(f_sim_log, lineterminator='\n')
            writer.writerow(['timestamp', "send_rate", 'recv_rate', 'latency',
                             'loss', 'reward', "action", "bytes_sent",
                             "bytes_acked", "bytes_lost", "send_start_time",
                             "send_end_time", 'recv_start_time',
                             'recv_end_time', 'latency_increase',
                             "packet_size", 'bandwidth', "queue_delay",
                             'packet_in_queue', 'queue_size', 'cwnd',
                             'ssthresh', "rto", "packets_in_flight"])
        else:
            f_sim_log = None
            writer = None

        while True:
            net.run(run_dur)
            mi = senders[0].get_run_data()

            throughput = mi.get("recv rate")  # bits/sec
            send_rate = mi.get("send rate")  # bits/sec
            latency = mi.get("avg latency")
            avg_queue_delay = mi.get("avg queue delay")
            loss = mi.get("loss ratio")

            reward = pcc_aurora_reward(
                throughput / BITS_PER_BYTE / BYTES_PER_PACKET, latency, loss,
                np.mean(trace.bandwidths) * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET)
            rewards.append(reward)
            try:
                ssthresh = senders[0].ssthresh
            except:
                ssthresh = 0
            action = 0

            if save_dir and writer:
                writer.writerow([
                    net.get_cur_time(), send_rate, throughput, latency, loss,
                    reward, action, mi.bytes_sent, mi.bytes_acked, mi.bytes_lost,
                    mi.send_start, mi.send_end, mi.recv_start, mi.recv_end,
                    mi.get('latency increase'), mi.packet_size,
                    links[0].get_bandwidth(
                        net.get_cur_time()) * BYTES_PER_PACKET * BITS_PER_BYTE,
                    avg_queue_delay, links[0].pkt_in_queue, links[0].queue_size,
                    senders[0].cwnd, ssthresh, senders[0].rto,
                    senders[0].bytes_in_flight / BYTES_PER_PACKET])
            if senders[0].srtt:
                run_dur = senders[0].srtt
            should_stop = trace.is_finished(net.get_cur_time())
            if should_stop:
                break
        if f_sim_log:
            f_sim_log.close()
        if self.record_pkt_log and save_dir:
            with open(os.path.join(
                    save_dir, "{}_packet_log.csv".format(self.cc_name)), 'w', 1) as f:
                pkt_logger = csv.writer(f, lineterminator='\n')
                pkt_logger.writerow(['timestamp', 'packet_event_id',
                                     'event_type', 'bytes', 'cur_latency',
                                     'queue_delay', 'packet_in_queue',
                                     'sending_rate', 'bandwidth'])
                pkt_logger.writerows(net.pkt_log)

        avg_sending_rate = senders[0].avg_sending_rate
        tput = senders[0].avg_throughput
        avg_lat = senders[0].avg_latency
        loss = senders[0].pkt_loss_rate
        pkt_level_reward = pcc_aurora_reward(tput, avg_lat,loss,
            avg_bw=trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET)
        pkt_level_original_reward = pcc_aurora_reward(tput, avg_lat, loss)
        if plot_flag and save_dir:
            plot_simulation_log(trace, os.path.join(save_dir, '{}_simulation_log.csv'.format(self.cc_name)), save_dir, self.cc_name)
            bin_tput_ts, bin_tput = senders[0].bin_tput
            bin_sending_rate_ts, bin_sending_rate = senders[0].bin_sending_rate
            lat_ts, lat = senders[0].latencies
            plot(trace, bin_tput_ts, bin_tput, bin_sending_rate_ts,
                 bin_sending_rate, tput * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                 avg_sending_rate * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                 lat_ts, lat, avg_lat * 1000, loss, pkt_level_original_reward,
                 pkt_level_reward, save_dir, self.cc_name)
        if save_dir:
            with open(os.path.join(save_dir, "{}_summary.csv".format(self.cc_name)), 'w', 1) as f:
                summary_writer = csv.writer(f, lineterminator='\n')
                summary_writer.writerow([
                    'trace_average_bandwidth', 'trace_average_latency',
                    'average_sending_rate', 'average_throughput',
                    'average_latency', 'loss_rate', 'mi_level_reward',
                    'pkt_level_reward'])
                summary_writer.writerow(
                    [trace.avg_bw, trace.avg_delay,
                     avg_sending_rate * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                     tput * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6, avg_lat,
                     loss, np.mean(rewards), pkt_level_reward])
        return np.mean(rewards), pkt_level_reward
