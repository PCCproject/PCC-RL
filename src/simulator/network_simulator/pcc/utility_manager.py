import numpy as np

from simulator.network_simulator.pcc.monitor_interval import MonitorInterval
from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET


class IntervalStats:
    def __init__(self) -> None:
        self.interval_duration = 0.0
        self.rtt_ratio = 0.0
        self.marked_lost_bytes = 0
        self.loss_rate = 0.0
        self.actual_sending_rate_mbps = 0.0
        self.ack_rate_mbps = 0.0

        self.avg_rtt = 0.0
        self.rtt_dev = 0.0
        self.min_rtt = -1.0
        self.max_rtt = -1.0
        self.approx_rtt_gradient = 0.0

        self.rtt_gradient = 0.0
        self.rtt_gradient_cut = 0.0
        self.rtt_gradient_error = 0.0

        self.trending_gradient = 0.0
        self.trending_gradient_cut = 0.0
        self.trending_gradient_error = 0.0

        self.trending_deviation = 0.0


class UtilityManager:
    kRttHistoryLen = 6
    # Exponent of sending rate contribution term in Vivace utility function.
    kSendingRateExponent = 0.9
    # Coefficient of loss penalty term in Vivace utility function.
    kVivaceLossCoefficient = 11.35
    # Coefficient of latency penalty term in Vivace utility function.
    kLatencyCoefficient = 900.0

    # The update rate for moving average variable.
    kAlpha = 0.1
    # The order of magnitude that distinguishes abnormal sample.
    kBeta = 100.0
    # Number of deviation above/below average trending gradient used for RTT
    # inflation tolerance for primary and scavenger senders.
    kInflationToleranceGainHigh = 2.0
    kInflationToleranceGainLow = 2.0
    # The threshold for ratio of monitor interval count, above which moving
    # average of trending RTT metrics (gradient and deviation) would be reset.
    kTrendingResetIntervalRatio = 0.95


    def __init__(self, utility_tag: str = "vivace") -> None:
        self.utility_tag = utility_tag
        self.interval_stats = IntervalStats()
        self.is_rtt_inflation_tolerable = True
        self.min_rtt = -1
        self.is_rtt_inflation_tolerable = True
        self.is_rtt_dev_tolerable = True
        self.mi_avg_rtt_history = []
        self.mi_rtt_dev_history = []
        self.ratio_inflated_mi = 0
        self.ratio_fluctuated_mi = 0

        self.min_trending_gradient = -1.0
        self.avg_trending_gradient = -1.0
        self.dev_trending_gradient = -1.0
        self.last_trending_gradient = -1.0

        self.avg_mi_rtt_dev = -1.0
        self.dev_mi_rtt_dev = -1.0

    def prepare_statistics(self, mi: MonitorInterval):
        self.preprocessing(mi)
        self.compute_simple_metrics(mi)
        self.compute_approx_rtt_gradient(mi)
        self.compute_rtt_gradient(mi)
        self.compute_rtt_deviation(mi)
        self.compute_rtt_gradient_error(mi)
        self.determine_tolerance_general()
        self.process_rtt_trend(mi)

    def preprocessing(self, mi: MonitorInterval):
        self.interval_stats.marked_lost_bytes = 0

    def compute_simple_metrics(self, mi: MonitorInterval):
        # Add the transfer time of the last packet in the monitor interval when
        # calculating monitor interval duration.
        self.interval_stats.interval_duration = mi.last_packet_sent_time - mi.first_packet_sent_time + BYTES_PER_PACKET / mi.sending_rate
        self.interval_stats.rtt_ratio = mi.rtt_on_monitor_start / mi.rtt_on_monitor_end
        self.interval_stats.loss_rate = (mi.bytes_lost - self.interval_stats.marked_lost_bytes) / mi.bytes_sent
        self.interval_stats.actual_sending_rate_mbps = mi.bytes_sent * BITS_PER_BYTE / self.interval_stats.interval_duration

        num_rtt_samples = len(mi.packet_rtt_samples)
        if num_rtt_samples > 1:
            ack_duration = mi.packet_rtt_samples[num_rtt_samples - 1].ack_timestamp - mi.packet_rtt_samples[0].ack_timestamp
            self.interval_stats.ack_rate_mbps = (mi.bytes_acked - BYTES_PER_PACKET) * BITS_PER_BYTE / ack_duration
        elif num_rtt_samples == 1:
            self.interval_stats.ack_rate_mbps = mi.bytes_acked / self.interval_stats.interval_duration
        else:
            self.interval_stats.ack_rate_mbps = 0.0

    def compute_approx_rtt_gradient(self, mi: MonitorInterval):
        # Separate all RTT samples in the interval into two halves, and
        # calculate an approximate RTT gradient.
        rtt_first_half = 0.0
        rtt_second_half = 0.0
        num_half_samples = int(len(mi.packet_rtt_samples) / 2)
        num_first_half_samples = 0
        num_second_half_samples = 0
        for i in range(num_half_samples):
            if mi.packet_rtt_samples[i].is_reliable_for_gradient_calculation:
                rtt_first_half = rtt_first_half + mi.packet_rtt_samples[i].sample_rtt
                num_first_half_samples+=1

            if mi.packet_rtt_samples[i + num_half_samples].is_reliable_for_gradient_calculation:
                rtt_second_half = rtt_second_half + mi.packet_rtt_samples[i + num_half_samples].sample_rtt
                num_second_half_samples+=1

        if num_first_half_samples == 0 or num_second_half_samples == 0:
          self.interval_stats.approx_rtt_gradient = 0.0
          return

        rtt_first_half = rtt_first_half * (1.0 / num_first_half_samples)
        rtt_second_half = rtt_second_half * (1.0 / num_second_half_samples)
        self.interval_stats.approx_rtt_gradient = 2.0 * (rtt_second_half - rtt_first_half) / (rtt_second_half + rtt_first_half)

    def compute_rtt_gradient(self, mi: MonitorInterval):
        if mi.num_reliable_rtt_for_gradient_calculation < 2:
          self.interval_stats.rtt_gradient = 0.0
          self.interval_stats.rtt_gradient_cut = 0.0
          return

        # Calculate RTT gradient using linear regression.
        gradient_x_avg = 0.0
        gradient_y_avg = 0.0
        gradient_x = 0.0
        gradient_y = 0.0
        for rtt_sample in mi.packet_rtt_samples:
            if not rtt_sample.is_reliable_for_gradient_calculation:
                continue

            gradient_x_avg += rtt_sample.packet_number
            gradient_y_avg += rtt_sample.sample_rtt

        gradient_x_avg /= mi.num_reliable_rtt_for_gradient_calculation
        gradient_y_avg /= mi.num_reliable_rtt_for_gradient_calculation
        for rtt_sample in mi.packet_rtt_samples:
            if not rtt_sample.is_reliable_for_gradient_calculation:
                continue

            delta_packet_number = rtt_sample.packet_number - gradient_x_avg
            delta_rtt_sample = rtt_sample.sample_rtt - gradient_y_avg
            gradient_x += delta_packet_number * delta_packet_number
            gradient_y += delta_packet_number * delta_rtt_sample

        self.interval_stats.rtt_gradient = gradient_y / gradient_x
        self.interval_stats.rtt_gradient /= (BYTES_PER_PACKET / mi.sending_rate)
        self.interval_stats.avg_rtt = gradient_y_avg
        self.interval_stats.rtt_gradient_cut = gradient_y_avg - self.interval_stats.rtt_gradient * gradient_x_avg

    def compute_rtt_deviation(self, mi: MonitorInterval):
        if mi.num_reliable_rtt < 2:
            self.interval_stats.rtt_dev = 0
            return

        # Calculate RTT deviation.
        self.interval_stats.rtt_dev = 0.0
        self.interval_stats.max_rtt = -1
        self.interval_stats.min_rtt = -1
        for rtt_sample in mi.packet_rtt_samples:
            if not rtt_sample.is_reliable:
              continue

            delta_rtt_sample = rtt_sample.sample_rtt - self.interval_stats.avg_rtt
            self.interval_stats.rtt_dev += delta_rtt_sample * delta_rtt_sample

            if self.min_rtt < 0 or rtt_sample.sample_rtt < self.min_rtt:
                self.min_rtt = rtt_sample.sample_rtt

            if self.interval_stats.min_rtt < 0 or rtt_sample.sample_rtt < self.interval_stats.min_rtt:
                self.interval_stats.min_rtt = rtt_sample.sample_rtt

            if self.interval_stats.max_rtt < 0 or rtt_sample.sample_rtt > self.interval_stats.max_rtt:
                self.interval_stats.max_rtt = rtt_sample.sample_rtt

        self.interval_stats.rtt_dev = np.sqrt(self.interval_stats.rtt_dev / mi.num_reliable_rtt)

    def compute_rtt_gradient_error(self, mi: MonitorInterval):
        self.interval_stats.rtt_gradient_error = 0.0
        if mi.num_reliable_rtt_for_gradient_calculation < 2:
            return

        for rtt_sample in mi.packet_rtt_samples:
            if not rtt_sample.is_reliable_for_gradient_calculation:
                continue

            regression_rtt = rtt_sample.packet_number * self.interval_stats.rtt_gradient + self.interval_stats.rtt_gradient_cut
            self.interval_stats.rtt_gradient_error += pow(rtt_sample.sample_rtt - regression_rtt, 2.0)

        self.interval_stats.rtt_gradient_error /= mi.num_reliable_rtt_for_gradient_calculation
        self.interval_stats.rtt_gradient_error = np.sqrt(self.interval_stats.rtt_gradient_error)
        self.interval_stats.rtt_gradient_error /= self.interval_stats.avg_rtt

    def determine_tolerance_general(self):
        if self.interval_stats.rtt_gradient_error < abs(self.interval_stats.rtt_gradient):
            self.is_rtt_inflation_tolerable = False
            self.is_rtt_dev_tolerable = False
        else:
            self.is_rtt_inflation_tolerable = True
            self.is_rtt_dev_tolerable = True

    def process_rtt_trend(self, mi: MonitorInterval):
        if mi.num_reliable_rtt < 2:
            return

        self.mi_avg_rtt_history.append(self.interval_stats.avg_rtt)
        self.mi_rtt_dev_history.append(self.interval_stats.rtt_dev)
        if len(self.mi_avg_rtt_history) > self.kRttHistoryLen:
          self.mi_avg_rtt_history.pop()

        if len(self.mi_rtt_dev_history) > self.kRttHistoryLen:
          self.mi_rtt_dev_history.pop()


        if len(self.mi_avg_rtt_history) >= self.kRttHistoryLen:
           self.compute_trending_gradient()
           self.compute_trending_gradient_error()
           self.determine_tolerance_inflation()

        if len(self.mi_rtt_dev_history) >= self.kRttHistoryLen:
            self.compute_trending_deviation()
            self.determine_tolerance_deviation()


    def compute_trending_gradient(self):
        # Calculate RTT gradient using linear regression.
        gradient_x_avg = 0.0
        gradient_y_avg = 0.0
        gradient_x = 0.0
        gradient_y = 0.0
        num_sample = len(self.mi_avg_rtt_history)
        for i in range(num_sample):
          gradient_x_avg += i
          gradient_y_avg += self.mi_avg_rtt_history[i]

        gradient_x_avg /= num_sample
        gradient_y_avg /= num_sample
        for i in range(num_sample):
            delta_x = i - gradient_x_avg
            delta_y = self.mi_avg_rtt_history[i] - gradient_y_avg
            gradient_x += delta_x * delta_x
            gradient_y += delta_x * delta_y

        self.interval_stats.trending_gradient = gradient_y / gradient_x
        self.interval_stats.trending_gradient_cut = gradient_y_avg - self.interval_stats.trending_gradient * gradient_x_avg

    def compute_trending_gradient_error(self):
        num_sample = len(self.mi_avg_rtt_history)
        self.interval_stats.trending_gradient_error = 0.0
        for i in range(num_sample):
            regression_rtt = i * self.interval_stats.trending_gradient + self.interval_stats.trending_gradient_cut
            self.interval_stats.trending_gradient_error += pow(self.mi_avg_rtt_history[i] - regression_rtt, 2.0)

        self.interval_stats.trending_gradient_error /= num_sample
        self.interval_stats.trending_gradient_error = np.sqrt(self.interval_stats.trending_gradient_error)

    def determine_tolerance_inflation(self):
        self.ratio_inflated_mi *= (1 - self.kAlpha)

        if self.utility_tag != "Scavenger" and len(self.mi_avg_rtt_history) < self.kRttHistoryLen:
            return

        if self.min_trending_gradient < 0.000001 or abs(self.interval_stats.trending_gradient) < self.min_trending_gradient / self.kBeta:
            self.avg_trending_gradient = 0.0
            self.min_trending_gradient = abs(self.interval_stats.trending_gradient)
            self.dev_trending_gradient = abs(self.interval_stats.trending_gradient)
            self.last_trending_gradient = self.interval_stats.trending_gradient
        else:
            dev_gain = self.kInflationToleranceGainLow if self.interval_stats.rtt_dev < 1000 else self.kInflationToleranceGainHigh
            tolerate_threshold_h = self.avg_trending_gradient + dev_gain * self.dev_trending_gradient
            tolerate_threshold_l = self.avg_trending_gradient - dev_gain * self.dev_trending_gradient
            if self.interval_stats.trending_gradient < tolerate_threshold_l or self.interval_stats.trending_gradient > tolerate_threshold_h:
                if interval_stats.trending_gradient > 0:
                    self.is_rtt_inflation_tolerable = False

                self.is_rtt_dev_tolerable = False
                self.ratio_inflated_mi += self.kAlpha
            else:
                self.dev_trending_gradient = self.dev_trending_gradient * (1 - self.kAlpha) + abs(self.interval_stats.trending_gradient - self.last_trending_gradient) * self.kAlpha
                self.avg_trending_gradient = self.avg_trending_gradient * (1 - self.kAlpha) + self.interval_stats.trending_gradient * self.kAlpha
            self.last_trending_gradient = self.interval_stats.trending_gradient

        self.min_trending_gradient = min(self.min_trending_gradient, abs(self.interval_stats.trending_gradient))

    def compute_trending_deviation(self):
        num_sample = len(self.mi_rtt_dev_history)
        avg_rtt_dev = 0.0
        for i in range(num_sample):
            avg_rtt_dev += self.mi_rtt_dev_history[i]

        avg_rtt_dev /= num_sample

        self.interval_stats.trending_deviation = 0.0
        for i in range(num_sample):
            delta_dev = avg_rtt_dev - self.mi_rtt_dev_history[i]
            self.interval_stats.trending_deviation += (delta_dev * delta_dev)

        self.interval_stats.trending_deviation /= num_sample
        self.interval_stats.trending_deviation = np.sqrt(self.interval_stats.trending_deviation)

    def determine_tolerance_deviation(self):
        self.ratio_fluctuated_mi *= (1 - self.kAlpha)

        if self.avg_mi_rtt_dev < 0.000001:
            self.avg_mi_rtt_dev = self.interval_stats.rtt_dev
            self.dev_mi_rtt_dev = 0.5 * self.interval_stats.rtt_dev
        else:
            if self.interval_stats.rtt_dev > self.avg_mi_rtt_dev + self.dev_mi_rtt_dev * 4.0 and interval_stats_.rtt_dev > 1:
                self.is_rtt_dev_tolerable = False
                self.ratio_fluctuated_mi += self.kAlpha
            else:
                self.dev_mi_rtt_dev = self.dev_mi_rtt_dev * (1 - self.kAlpha) + abs(self.interval_stats.rtt_dev - self.avg_mi_rtt_dev) * self.kAlpha
                self.avg_mi_rtt_dev = self.avg_mi_rtt_dev * (1 - self.kAlpha) + self.interval_stats.rtt_dev * self.kAlpha

        if self.ratio_fluctuated_mi > self.kTrendingResetIntervalRatio:
            self.avg_mi_rtt_dev = -1
            self.dev_mi_rtt_dev = -1
            self.ratio_fluctuated_mi = 0

    def calculate_utility(self, mi: MonitorInterval, event_time: float) -> float:
        # TODO: compute interval stats
        self.prepare_statistics(mi)
        utility = 0.0
        if self.utility_tag == "vivace":
            utility = self.calculate_utility_vivace(mi)
        else:
            raise RuntimeError
        return utility

    def calculate_utility_vivace(self, mi: MonitorInterval) -> float:
        return self.calculate_utility_proportional(
            mi, self.kLatencyCoefficient, self.kVivaceLossCoefficient)

    def calculate_utility_proportional(
        self, mi: MonitorInterval, latency_coefficient: float,
        loss_coefficient: float) -> float:
        sending_rate_contribution = pow(
            self.interval_stats.actual_sending_rate_mbps,
            self.kSendingRateExponent)

        rtt_gradient = 0.0 if self.is_rtt_inflation_tolerable else self.interval_stats.rtt_gradient
        if (mi.rtt_fluctuation_tolerance_ratio > 50.0 and
            abs(rtt_gradient) < 1000.0 / self.interval_stats.interval_duration):
            rtt_gradient = 0.0

        if rtt_gradient < 0:
            rtt_gradient = 0.0

        latency_penalty = latency_coefficient * rtt_gradient * self.interval_stats.actual_sending_rate_mbps

        loss_penalty = loss_coefficient * self.interval_stats.loss_rate * self.interval_stats.actual_sending_rate_mbps

        return sending_rate_contribution - latency_penalty - loss_penalty
