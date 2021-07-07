#!/usr/bin/env python
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

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Non-Production"



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
