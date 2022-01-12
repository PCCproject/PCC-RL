import glob
import multiprocessing as mp
import os
from typing import List, Union

from simulator.trace import Trace

LINKS_ADDED_AFTER_NSDI = [
    '2019-12-11T14-49-Colombia-to-AWS-Brazil-2-5-runs-3-flows',
    "2020-02-17T21-29-AWS-California-2-to-Mexico-5-runs",
    "2020-02-17T21-29-AWS-California-1-to-Stanford-5-runs",
    "2020-02-18T01-17-AWS-Brazil-1-to-Brazil-5-runs-3-flows",
    "2020-02-18T01-17-AWS-India-1-to-India-5-runs-3-flows",


    "2019-07-11T21-09-AWS-California-1-to-Stanford-cellular-3-runs",
    "2019-06-27T17-42-AWS-California-1-to-Stanford-cellular-3-runs",
    "2019-06-27T00-52-AWS-California-1-to-Stanford-cellular-3-runs",
    "2019-05-29T18-09-AWS-California-1-to-Stanford-cellular-3-runs",
    "2019-02-05T00-46-AWS-India-1-to-India-cellular-3-runs",
    "2019-01-24T23-21-AWS-Brazil-2-to-Colombia-cellular-3-runs-3-flows",
    "2019-01-24T23-21-AWS-India-1-to-India-cellular-3-runs-3-flows",
    "2019-01-24T13-00-AWS-India-1-to-India-cellular-3-runs",
    "2019-01-22T00-44-India-cellular-to-AWS-India-1-3-runs-3-flows",
    "2019-01-21T19-18-India-cellular-to-AWS-India-1-3-runs",

    # didnt have before nsdi in simu
    "2018-12-10T20-36-AWS-India-1-to-India-cellular-3-runs",
    "2018-12-10T20-36-AWS-India-2-to-Stanford-cellular-3-runs",
    "2019-07-17T15-55-AWS-California-1-to-Stanford-cellular-3-runs",
    "2019-08-26T16-05-AWS-California-1-to-Stanford-cellular-3-runs"]


class PantheonDataset:

    def __init__(self, root: str, conn_type: str, post_nsdi: bool = False,
                 target_ccs: List[str] = ["bbr", "cubic", "vegas", "indigo",
                                          "ledbat", "quic"]):
        self.conn_type = conn_type
        self.link_dirs = sorted(glob.glob(os.path.join(root, conn_type, "*/")))
        self.trace_files = []
        self.link_names = []
        self.trace_names = [] # list of (link_name, run_name)
        for link_dir in self.link_dirs:
            link_name = link_dir.split('/')[-2]
            if not post_nsdi and link_name in LINKS_ADDED_AFTER_NSDI:
                continue
            self.link_names.append(link_name)
            for cc in target_ccs:
                trace_files = list(sorted(glob.glob(os.path.join(
                    link_dir, "{}_datalink_run[1-3].log".format(cc)))))
                self.trace_files += trace_files
                for trace_file in trace_files:
                    run_name = os.path.splitext(os.path.basename(trace_file))[0]
                    self.trace_names.append((link_name, run_name))

        self.traces = []

    def get_traces(self, loss: float, queue_size: Union[int, None] = None,
                   front_offset: float = 0.0, wrap: bool = False,
                   nproc: int = 8):
        if self.traces:
            return self.traces
        if not queue_size:
            queue_size = 10
        arguments = [(trace_file, loss, queue_size, 500, front_offset, wrap)
                     for trace_file in self.trace_files]
        with mp.Pool(processes=nproc) as pool:
            self.traces = pool.starmap(Trace.load_from_pantheon_file, arguments)
        if not queue_size:
            for trace in self.traces:
                trace.queue_size = max(2, int(trace.bdp))
        return self.traces

    def __len__(self):
        return len(self.trace_files)
