import os

from simulator.pantheon_trace_parser.tunnel_graph import TunnelGraph


class Flow():
    def __init__(self, log_path, ms_per_bin=500):
        self.tunnel_graph = TunnelGraph(log_path, ms_per_bin=ms_per_bin)
        self.tunnel_graph.parse_tunnel_log()
        self.cc = str(os.path.basename(log_path).split("_")[0])
        self.ms_per_bin = ms_per_bin

    @property
    def link_capacity_timestamps(self):
        """Return througput timestamps in second."""
        return self.tunnel_graph.link_capacity_t

    @property
    def link_capacity(self):
        """Return throuhgput in Mbps."""
        return self.tunnel_graph.link_capacity

    @property
    def avg_link_capacity(self):
        return self.tunnel_graph.avg_capacity

    @property
    def throughput_timestamps(self):
        """Return througput timestamps in second."""
        return self.tunnel_graph.egress_t[1]

    @property
    def throughput(self):
        """Return throuhgput in Mbps."""
        return self.tunnel_graph.egress_tput[1]

    @property
    def avg_throughput(self):
        return self.tunnel_graph.avg_egress[1]

    @property
    def sending_rate_timestamps(self):
        """Return sending rate timestamps in second."""
        return self.tunnel_graph.ingress_t[1]

    @property
    def sending_rate(self):
        """Return sending rate in Mbps."""
        return self.tunnel_graph.ingress_tput[1]

    @property
    def avg_sending_rate(self):
        return self.tunnel_graph.avg_ingress[1]

    @property
    def one_way_delay_timestamps(self):
        """Return one-way delay timestamps in second."""
        return self.tunnel_graph.delays_t[1]

    @property
    def one_way_delay(self):
        """Return one-way delay in millisecond."""
        return self.tunnel_graph.delays[1]

    @property
    def loss_rate(self):
        return self.tunnel_graph.loss_rate[1]
