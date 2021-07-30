#!/usr/bin/env python
r"""For performing various TN operations involving QR factorization.
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

def left_to_right_sweep(nodes, is_infinite, normalize):
    num_nodes = len(nodes)
    imin = 0
    imax = imin + (num_nodes - 2 + int(is_infinite))

    kwargs = {"nodes": nodes, "current_orthogonal_center_idx": imin}
    
    for i in range(imin, imax+1):
        kwargs["current_orthogonal_center_idx"] = i
        shift_orthogonal_center_to_the_right(**kwargs)

    if (not is_infinite) and normalize:
        nodes[-1] /= tn.norm(nodes[-1])

    return None



def right_to_left_sweep(nodes, is_infinite, normalize):
    num_nodes = len(nodes)
    imax = num_nodes - 1 + int(is_infinite)
    imin = imax - (num_nodes - 1 + int(is_infinite)) + 1

    kwargs = {"nodes": nodes, "current_orthogonal_center_idx": imax}
    
    for i in range(imax, imin-1, -1):
        kwargs["current_orthogonal_center_idx"] = i
        shift_orthogonal_center_to_the_left(**kwargs)

    if (not is_infinite) and normalize:
        nodes[0] /= tn.norm(nodes[0])

    return None



def shift_orthogonal_center_to_the_right(nodes, current_orthogonal_center_idx):
    # Function does not check correctness of 'current_orthogonal_center_idx'.
    i = current_orthogonal_center_idx

    num_nodes = len(nodes)
    node_i = nodes[i%num_nodes]
    num_edges_per_node = len(node_i.edges)
    left_edges = tuple(node_i[idx] for idx in range(num_edges_per_node-1))
    right_edges = (node_i[num_edges_per_node-1],)
        
    Q, R = split_node(node_i, left_edges, right_edges)

    nodes[i%num_nodes] = Q

    node_iP1 = nodes[(i+1)%num_nodes]
    nodes_to_contract = (R, node_iP1)
    node_iP1_struct = ((1, -2, -3)
                       if num_edges_per_node == 3
                       else (1, -2, -3, -4))
    network_struct = [(-1, 1), node_iP1_struct]
    nodes[(i+1)%num_nodes] = tn.ncon(nodes_to_contract, network_struct)

    return Q, R



def shift_orthogonal_center_to_the_left(nodes, current_orthogonal_center_idx):
    # Function does not check correctness of 'current_orthogonal_center_idx'.
    i = current_orthogonal_center_idx
    
    num_nodes = len(nodes)
    conj_node_i = tn.conj(nodes[i%num_nodes])
    num_edges_per_node = len(conj_node_i.edges)
    left_edges = tuple(conj_node_i[idx] for idx in range(1, num_edges_per_node))
    right_edges = (conj_node_i[0],)

    Q, R = split_node(conj_node_i, left_edges, right_edges)

    new_edge_order = [Q[-1]] + [Q[idx] for idx in range(num_edges_per_node-1)]
    Q_dagger = tn.conj(Q.reorder_edges(new_edge_order))
    nodes[i%num_nodes] = Q_dagger

    new_edge_order = [R[1], R[0]]
    R_dagger = tn.conj(R.reorder_edges(new_edge_order))

    node_iM1 = nodes[(i-1)%num_nodes]
    nodes_to_contract = (node_iM1, R_dagger)
    node_iM1_struct = ((-1, -2, 1)
                       if num_edges_per_node == 3
                       else (-1, -2, -3, 1))
    network_struct = [node_iM1_struct, (1, -num_edges_per_node)]
    nodes[(i-1)%num_nodes] = tn.ncon(nodes_to_contract, network_struct)

    return R_dagger, Q_dagger



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
