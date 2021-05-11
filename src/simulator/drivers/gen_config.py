import csv
from simulator.trace import Trace
from common.utils import write_json_file

for i in range(5):
    timestamps = []
    bandwidths = []
    delays = []
    queue = 2
    loss = 0
    delay_noise = 0
    with open('test_aws_new/run{}/delay_time_series.csv'.format(i), 'r') as f:
        reader = csv.reader(f)
        for cols in reader:
            timestamps.append(float(cols[0]))
            delays.append(float(cols[1]))
            bandwidths.append(0.6)
    tr = Trace(timestamps, bandwidths, delays, loss, queue, delay_noise)
    tr.dump('test_aws_new/run{}/trace.json'.format(i))
