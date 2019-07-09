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

import tensorflow as tf
import numpy as np
import io

class LoadedModel():

    def __init__(self, model_path):
        self.sess = tf.Session()
        self.model_path = model_path
        self.metagraph = tf.saved_model.loader.load(self.sess,
            [tf.saved_model.tag_constants.SERVING], self.model_path)
        sig = self.metagraph.signature_def["serving_default"]
        input_dict = dict(sig.inputs)
        output_dict = dict(sig.outputs)       
 
        self.input_obs_label = input_dict["ob"].name
        self.input_state_label = None
        self.initial_state = None
        self.state = None
        if "state" in input_dict.keys():
            self.input_state_label = input_dict["state"].name
            strfile = io.StringIO()
            print(input_dict["state"].tensor_shape, file=strfile)
            lines = strfile.getvalue().split("\n")
            dim_1 = int(lines[1].split(":")[1].strip(" "))
            dim_2 = int(lines[4].split(":")[1].strip(" "))
            self.initial_state = np.zeros((dim_1, dim_2), dtype=np.float32)
            self.state = np.zeros((dim_1, dim_2), dtype=np.float32)
 
        self.output_act_label = output_dict["act"].name
        self.output_stochastic_act_label = None
        if "stochastic_act" in output_dict.keys():
            self.output_stochastic_act_label = output_dict["stochastic_act"].name

        self.mask = None
        self.input_mask_label = None 
        if "mask" in input_dict.keys():
            self.input_mask_label = input_dict["mask"].name
            self.mask = np.ones((1, 1)).reshape((1,))

    def reset_state(self):      
        self.state = np.copy(self.initial_state)

    def reload(self):
        self.metagraph = tf.saved_model.loader.load(self.sess,
            [tf.saved_model.tag_constants.SERVING], self.model_path)
 
    def act(self, obs, stochastic=False):
        input_dict = {self.input_obs_label:obs}
        if self.state is not None:
            input_dict[self.input_state_label] = self.state

        if self.mask is not None:
            input_dict[self.input_mask_label] = self.mask

        sess_output = None
        if stochastic:
            sess_output = self.sess.run(self.output_stochastic_act_label, feed_dict=input_dict)
        else:
            sess_output = self.sess.run(self.output_act_label, feed_dict=input_dict)

        action = None
        if len(sess_output) > 1:
            action, self.state = sess_output
        else:
            action = sess_output

        return {"act":action}


class LoadedModelAgent():

    def __init__(self, model_path):
        self.model = LoadedModel(model_path)

    def reset(self):
        self.model.reset_state()

    def act(self, ob):

        act_dict = self.model.act(ob.reshape(1,-1), stochastic=False)

        ac = act_dict["act"]
        vpred = act_dict["vpred"] if "vpred" in act_dict.keys() else None
        state = act_dict["state"] if "state" in act_dict.keys() else None

        return ac[0][0]
