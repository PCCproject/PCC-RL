import pandas as pd
import csv

for i in range(5):
    output = []

    # df = pd.read_csv('test_aws_new/run{}/aurora_emulation_log.csv'.format(i))
    df = pd.read_csv('test_aurora{}/aurora_simulation_log.csv'.format(i))
    output.append(df['timestamp'].tolist())
    output.append((df['latency_increase'] / (df['recv_end_time'] - df['recv_start_time'])).tolist())
    output.append((df['min_lat'] / df['latency']).tolist())
    output.append((df['send_rate'] / df['recv_rate']).tolist())
    output.append((df['latency_increase']).tolist())
    output.append((df['latency']).tolist())
    # output = output.transpose()
    # print(output))

    # output = pd.DataFrame(output, columns=).transpose()
    # print(output)
    with open('test_aurora{}/aurora_simulation_log_processed.csv'.format(i), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['send_latency_inflation',
                         'latency_ratio', 'send_ratio', 'latency_increase',
                         'latency'])
        writer.writerows(zip(*output))

