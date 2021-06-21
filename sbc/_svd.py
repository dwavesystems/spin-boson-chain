#!/usr/bin/env python
r"""For performing various TN operations involving SVD.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For general array handling.
import numpy as np

# For performing SVDs as a backup if the default implementation fails.
import scipy.linalg

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

def left_to_right_svd_sweep(nodes, compress_params, is_infinite):
    truncated_schmidt_spectrum = []
        
    num_nodes = len(nodes)
    imax = num_nodes - 2 + int(is_infinite)
    
    for i in range(imax+1):
        current_orthogonal_center_idx = i
        S = shift_orthogonal_center_to_the_right(nodes,
                                                 current_orthogonal_center_idx,
                                                 compress_params)        
        truncated_schmidt_spectrum.append(S)

    return truncated_schmidt_spectrum



def right_to_left_svd_sweep(nodes, compress_params, is_infinite):
    truncated_schmidt_spectrum = []

    num_nodes = len(nodes)
    imax = num_nodes - 1 + int(is_infinite)
    
    for i in range(imax, 0, -1):
        current_orthogonal_center_idx = i
        S = shift_orthogonal_center_to_the_left(nodes,
                                                current_orthogonal_center_idx,
                                                compress_params)
        truncated_schmidt_spectrum.insert(0, S)

    return truncated_schmidt_spectrum



def shift_orthogonal_center_to_the_right(nodes,
                                         current_orthogonal_center_idx,
                                         compress_params):
    # Function does not check correctness of 'current_orthogonal_center_idx'.
    i = current_orthogonal_center_idx

    num_nodes = len(nodes)
    node_i = nodes[i%num_nodes]
    num_edges_per_node = len(node_i.edges)
    left_edges = tuple(node_i[idx] for idx in range(num_edges_per_node-1))
    right_edges = (node_i[num_edges_per_node-1],)
        
    U, S, V_dagger = split_node_full_svd(node_i,
                                         left_edges,
                                         right_edges,
                                         compress_params)

    nodes[i%num_nodes] = U

    node_iP1 = nodes[(i+1)%num_nodes]
    nodes_to_contract = (S, V_dagger, node_iP1)
    node_iP1_struct = ((2, -2, -3)
                       if num_edges_per_node == 3
                       else (2, -2, -3, -4))
    network_struct = [(-1, 1), (1, 2), node_iP1_struct]
    nodes[(i+1)%num_nodes] = tn.ncon(nodes_to_contract, network_struct)

    return S



def shift_orthogonal_center_to_the_left(nodes,
                                        current_orthogonal_center_idx,
                                        compress_params):
    # Function does not check correctness of 'current_orthogonal_center_idx'.
    i = current_orthogonal_center_idx
    
    num_nodes = len(nodes)
    node_i = nodes[i%num_nodes]
    num_edges_per_node = len(node_i.edges)
    left_edges = (node_i[0],)
    right_edges = tuple(node_i[idx] for idx in range(1, num_edges_per_node))

    U, S, V_dagger = split_node_full_svd(node_i,
                                         left_edges,
                                         right_edges,
                                         compress_params)
        
    nodes[i%num_nodes] = V_dagger

    node_iM1 = nodes[i-1]
    nodes_to_contract = (node_iM1, U, S)
    node_iM1_struct = ((-1, -2, 2)
                       if num_edges_per_node == 3
                       else (-1, -2, -3, 2))
    network_struct = [node_iM1_struct, (2, 1), (1, -3)]
    nodes[i-1] = tn.ncon(nodes_to_contract, network_struct)

    return S



def split_node_full_svd(node, left_edges, right_edges, compress_params):
    # Switch to numpy backend (if numpy is not being used) so that SVD can
    # be performed on CPUs as it is currently faster than on GPUs.
    original_backend_name = node.backend.name
    if original_backend_name != "numpy":
        tf_to_np_backend(node)
        
    if compress_params == None:
        max_num_singular_values = None
        max_trunc_err = None
        svd_rel_tol = None
    else:
        max_num_singular_values = compress_params.max_num_singular_values
        max_trunc_err = compress_params.max_trunc_err
        svd_rel_tol = compress_params.svd_rel_tol

    kwargs = {"node": node,
              "left_edges": left_edges,
              "right_edges": right_edges,
              "max_singular_values": max_num_singular_values,
              "max_truncation_err": max_trunc_err}

    try:
        U, S, V_dagger, _ = tn.split_node_full_svd(**kwargs)
        U[-1] | S[0]  # Break edge between U and S nodes.
        S[-1] | V_dagger[0]  # Break edge between S and V_dagger.
    except np.linalg.LinAlgError:
        U, S, V_dagger = split_node_full_svd_backup(**kwargs)

    if svd_rel_tol is not None:
        singular_vals = np.diag(S.tensor)
        max_singular_val = singular_vals[0]  # Already ordered.
        cutoff_idx = \
            np.where(singular_vals > max_singular_val * svd_rel_tol)[0][-1] + 1

        U = tn.Node(U.tensor[..., :cutoff_idx])
        S = tn.Node(S.tensor[:cutoff_idx, :cutoff_idx])
        V_dagger = tn.Node(V_dagger.tensor[:cutoff_idx, ...])

    # Switch back to original backend (if different from numpy).
    if original_backend_name != "numpy":
        np_to_tf_backend(U)
        np_to_tf_backend(S)
        np_to_tf_backend(V_dagger)

    return U, S, V_dagger



def split_node_qr(node, left_edges, right_edges):
    # Switch to numpy backend (if numpy is not being used) so that SVD can be
    # performed on CPUs as it is currently faster than on GPUs.
    original_backend_name = node.backend.name
    if original_backend_name != "numpy":
        tf_to_np_backend(node)

    Q, R = tn.split_node_qr(node=node,
                            left_edges=left_edges,
                            right_edges=right_edges)

    Q[-1] | R[0]  # Break edge between the two nodes.

    # Switch back to original backend (if different from numpy).
    if original_backend_name != "numpy":
        np_to_tf_backend(Q)
        np_to_tf_backend(R)

    return Q, R



def split_node_full_svd_backup(node,
                               left_edges,
                               right_edges,
                               max_singular_values,
                               max_truncation_err):
    num_left_edges = len(left_edges)
    num_right_edges = len(right_edges)
    node.reorder_edges(left_edges+right_edges)
    U_S_V_dagger_shape = node.shape
    
    if num_left_edges > 1:
        tn.flatten_edges([node[idx] for idx in range(num_left_edges)])
        num_edges = len(node.shape)
        node.reorder_edges([node[-1]]+[node[idx] for idx in range(num_edges-1)])
    if num_right_edges > 1:
        tn.flatten_edges([node[idx+1] for idx in range(num_right_edges)])

    U, S, V_dagger = scipy.linalg.svd(a=node.tensor,
                                      overwrite_a=True,
                                      lapack_driver="gesvd")

    truncation_err_sq = 0
    num_singular_values_left = S.size
    if max_truncation_err is not None:
        while num_singular_values_left > 1:
            idx = num_singular_values_left - 1
            truncation_err_sq += S[idx] * S[idx]
            if np.sqrt(truncation_err_sq) > max_truncation_err:
                break
            num_singular_values_left -= 1

    max_singular_values = \
        np.inf if max_singular_values is None else max_singular_values
    if num_singular_values_left > max_singular_values:
        cutoff_idx = max_singular_values
    else:
        cutoff_idx = num_singular_values_left

    new_U_shape = U_S_V_dagger_shape[:num_left_edges] + (cutoff_idx,)
    new_V_dagger_shape = (cutoff_idx,) + U_S_V_dagger_shape[-num_right_edges:]
    
    U = tn.Node(U[:, :cutoff_idx].reshape(new_U_shape))
    S = tn.Node(np.diag(S[:cutoff_idx]))
    V_dagger = tn.Node(V_dagger[:cutoff_idx, :].reshape(new_V_dagger_shape))

    return U, S, V_dagger
        


def tf_to_np_backend(node):
    node.backend = tn.backends.backend_factory.get_backend("numpy")
    node.tensor = node.tensor.numpy()  # Converts to numpy array.

    return None



def np_to_tf_backend(node):
    node.backend = tn.backends.backend_factory.get_backend("tensorflow")
    node.tensor = node.backend.convert_to_tensor(node.tensor)

    return None
