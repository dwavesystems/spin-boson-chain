#!/usr/bin/env python
r"""For performing the application of a MPO to a MPS.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For creating tensor networks and performing contractions.
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



##################################
## Define classes and functions ##
##################################



def _apply_mpo_to_mps(mpo_nodes, mps_nodes):
    new_mps_nodes = []
    for mpo_node, mps_node in zip(mpo_nodes, mps_nodes):
        nodes_to_contract = [mpo_node, mps_node]
        network_struct = [(-1, -3, 1, -4), (-2, 1, -5)]
        new_mps_node = tn.ncon(nodes_to_contract, network_struct)

        tn.flatten_edges([new_mps_node[0], new_mps_node[1]])
        tn.flatten_edges([new_mps_node[1], new_mps_node[2]])
        new_mps_node.reorder_edges([new_mps_node[1],
                                    new_mps_node[0],
                                    new_mps_node[2]])

        new_mps_nodes.append(new_mps_node)

    return new_mps_nodes
