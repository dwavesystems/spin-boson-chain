# Copyright 2021 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

r"""Contains functions for switching ``tensornetwork`` backend.
"""


#####################################
## Load libraries/packages/modules ##
#####################################

# For selecting different backends of the ``tensornetwork`` library.
import tensornetwork as tn



############################
## Authorship information ##
############################

__author__     = "D-Wave Systems Inc."
__copyright__  = "Copyright 2021"
__credits__    = ["Matthew Fitzpatrick"]
__maintainer__ = "D-Wave Systems Inc."
__email__      = "support@dwavesys.com"
__status__     = "Development"



####################################
## Define functions and constants ##
####################################

def tf_to_np(node):
    node.backend = tn.backends.backend_factory.get_backend("numpy")
    node.tensor = node.tensor.numpy()  # Converts to numpy array.

    return None



def np_to_tf(node):
    node.backend = tn.backends.backend_factory.get_backend("tensorflow")
    node.tensor = node.backend.convert_to_tensor(node.tensor)

    return None
