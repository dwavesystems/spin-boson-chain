#!/usr/bin/env python
r"""For performing QR factorizations.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# For switching the ``tensornetwork`` backend.
import sbc._backend



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Non-Production"



##################################
## Define classes and functions ##
##################################

def split_node(node, left_edges, right_edges):
    # Switch to numpy backend (if numpy is not being used) so that SVD can be
    # performed on CPUs as it is currently faster than on GPUs.
    original_backend_name = node.backend.name
    if original_backend_name != "numpy":
        sbc._backend.tf_to_np(node)

    Q, R = tn.split_node_qr(node=node,
                            left_edges=left_edges,
                            right_edges=right_edges)

    Q[-1] | R[0]  # Break edge between the two nodes.

    # Switch back to original backend (if different from numpy).
    if original_backend_name != "numpy":
        sbc._backend.np_to_tf(Q)
        sbc._backend.np_to_tf(R)

    return Q, R
