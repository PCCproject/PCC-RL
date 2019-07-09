# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


print("Importing module!")

def give_sample(flow_id, bytes_sent, bytes_acked, bytes_lost,
                send_start_time, send_end_time, recv_start_time,
                recv_end_time, rtt_samples, packet_size, utility):
    print("Got Sample:")
    print("\tflow_id: %d" % flow_id)
    print("\tbytes_sent: %d" % bytes_sent)
    print("\tbytes_acked: %d" % bytes_acked)
    print("\tbytes_lost: %d" % bytes_lost)
    print("\tsend_start_time: %f" % send_start_time)
    print("\tsend_end_time: %f" % send_end_time)
    print("\trecv_start_time: %f" % recv_start_time)
    print("\trecv_end_time: %f" % recv_end_time)
    print("\trtt_samples: %s" % rtt_samples)
    print("\tpacket_size: %d" % packet_size)
    print("\tutility: %f" % utility)
    
def reset(flow_id):
    pass

def get_rate(flow_id):
    return 3e6

def init(flow_id):
    pass
