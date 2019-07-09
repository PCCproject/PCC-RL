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

import sys

_arg_dict = {}
for arg in sys.argv:
    eq_pos = arg.find('=')
    if eq_pos >= 0:
        _arg_dict[arg[:eq_pos]] = arg[eq_pos + 1:]
    else:
        _arg_dict[arg] = True

def arg_or_default(arg, default=None):
    if arg in _arg_dict.keys():
        result = _arg_dict[arg]
        if isinstance(default, int):
            return int(result)
        if isinstance(default, float):
            return float(result)
        return result
    else:
        return default

