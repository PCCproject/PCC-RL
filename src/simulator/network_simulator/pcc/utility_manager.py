class IntervalStats:
    def __init__(self) -> None:
        self.interval_duration = 0.0
        self.rtt_ratio = 0.0
        self.marked_lost_bytes = 0
        self.loss_rate = 0
        self.actual_sending_rate_mbps = 0
        self.ack_rate_mbps = 0

        self.avg_rtt = 0.0
        self.rtt_dev = 0.0
        self.min_rtt = -1.0
        self.max_rtt = -1.0
        self.approx_rtt_gradient = 0

        self.rtt_gradient = 0
        self.rtt_gradient_cut = 0
        self.rtt_gradient_error = 0

        self.trending_gradient = 0
        self.trending_gradient_cut = 0
        self.trending_gradient_error = 0

        self.trending_deviation = 0

class UtilityManager:
    # Exponent of sending rate contribution term in Vivace utility function.
    kSendingRateExponent = 0.9
    # Coefficient of loss penalty term in Vivace utility function.
    kVivaceLossCoefficient = 11.35
    # Coefficient of latency penalty term in Vivace utility function.
    kLatencyCoefficient = 900.0

    def __init__(self, utility_tag: str = "vivace") -> None:
        self.utility_tag = utility_tag
        self.interval_stats = IntervalStats()

    def calculate_utility(self, mi, event_time: float) -> float:
        # TODO: compute interval stats
        utility = 0.0
        if self.utility_tag == "Vivace":
            utility = self.calculate_utility_vivace(mi)
        else:
            raise RuntimeError
        return utility

    def calculate_utility_vivace(self, mi) -> float:
        return self.calculate_utility_proportional(
            mi, self.kLatencyCoefficient, self.kVivaceLossCoefficient)

    def calculate_utility_proportional(self, mi, latency_coefficient: float,
                                       loss_coefficient: float) -> float:
        sending_rate_contribution = pow(self.interval_stats.actual_sending_rate_mbps, self.kSendingRateExponent)

        rtt_gradient = 0.0 if is_rtt_inflation_tolerable_ else self.interval_stats.rtt_gradient
        if (mi.rtt_fluctuation_tolerance_ratio > 50.0 and
            abs(rtt_gradient) < 1000.0 / self.interval_stats.interval_duration):
            rtt_gradient = 0.0

        if rtt_gradient < 0:
            rtt_gradient = 0.0

        latency_penalty = latency_coefficient * rtt_gradient * self.interval_stats.actual_sending_rate_mbps

        loss_penalty = loss_coefficient * self.interval_stats.loss_rate * self.interval_stats.actual_sending_rate_mbps

        return sending_rate_contribution - latency_penalty - loss_penalty
