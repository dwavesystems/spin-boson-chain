#!/usr/bin/env python
r"""For performing various TN operations involving SVD.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For deep copies of objects.
import copy



# For general array handling.
import numpy as np

# For performing SVDs as a backup if the default implementation fails.
import scipy.linalg

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

def left_to_right_sweep(nodes, compress_params, is_infinite, starting_node_idx):
    truncated_schmidt_spectra = []
        
    num_nodes = len(nodes)
    imin = 0 if starting_node_idx is None else starting_node_idx
    imax = imin + (num_nodes - 2 + int(is_infinite))

    kwargs = {"nodes": nodes,
              "current_orthogonal_center_idx": imin,
              "compress_params": compress_params}
    
    for i in range(imin, imax+1):
        kwargs["current_orthogonal_center_idx"] = i
        U, S, V_dagger = shift_orthogonal_center_to_the_right(**kwargs)
        truncated_schmidt_spectra.append(S)

    return truncated_schmidt_spectra



def right_to_left_sweep(nodes, compress_params, is_infinite, starting_node_idx):
    truncated_schmidt_spectra = []

    num_nodes = len(nodes)
    imax = (num_nodes - 1 + int(is_infinite)
            if starting_node_idx is None
            else starting_node_idx)
    imin = imax - (num_nodes - 1 + int(is_infinite)) + 1

    kwargs = {"nodes": nodes,
              "current_orthogonal_center_idx": imax,
              "compress_params": compress_params}
    
    for i in range(imax, imin-1, -1):
        current_orthogonal_center_idx = i
        U, S, V_dagger = shift_orthogonal_center_to_the_left(**kwargs)
        truncated_schmidt_spectra.insert(0, S)

    return truncated_schmidt_spectra



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
        
    U, S, V_dagger = split_node_full(node_i,
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

    return U, S, V_dagger



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

    U, S, V_dagger = split_node_full(node_i,
                                     left_edges,
                                     right_edges,
                                     compress_params)
        
    nodes[i%num_nodes] = V_dagger

    node_iM1 = nodes[(i-1)%num_nodes]
    nodes_to_contract = (node_iM1, U, S)
    node_iM1_struct = ((-1, -2, 2)
                       if num_edges_per_node == 3
                       else (-1, -2, -3, 2))
    network_struct = [node_iM1_struct, (2, 1), (1, -3)]
    nodes[(i-1)%num_nodes] = tn.ncon(nodes_to_contract, network_struct)

    return U, S, V_dagger



def split_node_full(node, left_edges, right_edges, compress_params):
    # Switch to numpy backend (if numpy is not being used) so that SVD can
    # be performed on CPUs as it is currently faster than on GPUs.
    original_backend_name = node.backend.name
    if original_backend_name != "numpy":
        sbc._backend.tf_to_np(node)
        
    if compress_params is None:
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
        U, S, V_dagger, discarded_singular_values = \
            tn.split_node_full_svd(**kwargs)
        U[-1] | S[0]  # Break edge between U and S nodes.
        S[-1] | V_dagger[0]  # Break edge between S and V_dagger.
    except np.linalg.LinAlgError:
        U, S, V_dagger, discarded_singular_values = \
            split_node_full_backup(**kwargs)

    if svd_rel_tol is not None:
        singular_vals = np.diag(S.tensor)
        max_singular_val = singular_vals[0]  # Already ordered.
        cutoff_idx = \
            np.where(singular_vals > max_singular_val * svd_rel_tol)[0][-1] + 1

        to_concatenate = (discarded_singular_values, singular_vals[cutoff_idx:])
        discarded_singular_values = np.concatenate(to_concatenate)

        U = tn.Node(U.tensor[..., :cutoff_idx])
        S = tn.Node(S.tensor[:cutoff_idx, :cutoff_idx])
        V_dagger = tn.Node(V_dagger.tensor[:cutoff_idx, ...])

    S = rescale_S(S, discarded_singular_values)

    # Switch back to original backend (if different from numpy).
    if original_backend_name != "numpy":
        sbc._backend.np_to_tf(U)
        sbc._backend.np_to_tf(S)
        sbc._backend.np_to_tf(V_dagger)

    return U, S, V_dagger



def split_node_full_backup(node,
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
    discarded_singular_values = S[cutoff_idx:]

    return U, S, V_dagger, discarded_singular_values



def rescale_S(S, discarded_singular_values):
    # S is the node containing the singular values that were not discarded.
    kept_singular_values = np.diag(S.tensor)
    to_concatenate = (kept_singular_values, discarded_singular_values)
    all_singular_values = np.concatenate(to_concatenate)

    largest_singular_value = np.amax(kept_singular_values)
    all_singular_values_rescaled = all_singular_values / largest_singular_value
        
    kept_singular_values_rescaled = (kept_singular_values
                                     / np.amax(kept_singular_values))
    kept_singular_values_rescaled /= \
        np.linalg.norm(kept_singular_values_rescaled)
    kept_singular_values_rescaled *= \
        np.linalg.norm(all_singular_values_rescaled)
    kept_singular_values_rescaled *= largest_singular_value

    S = tn.Node(np.diag(kept_singular_values_rescaled))

    return S



def vidal_form(mps_nodes, compress_params, is_infinite):
    L = len(mps_nodes)
    Gammas = [None] * L
    # Lambdas = [None] * (L - 1 + int(is_infinite))

    # imax = L - 2 + int(is_infinite)

    kwargs = {"nodes": mps_nodes,
              "compress_params": None,
              "is_infinite": is_infinite,
              "starting_node_idx": None}
    left_to_right_sweep(**kwargs)

    kwargs["compress_params"] = compress_params
    Lambdas = right_to_left_sweep(**kwargs)
    if is_infinite:
        Lambdas.insert(0, Lambdas.pop())

    for r in range(L):
        if (not is_infinite) and (r == L-1):
            Gammas[r] = mps_nodes[r]
        else:
            idx = (r + int(is_infinite)) % L
            Lambda_tensor = np.array(Lambdas[idx].tensor)
            Lambda_inv_tensor = np.diag(1 / np.diag(Lambda_tensor))
            Lambda_inv = tn.Node(Lambda_inv_tensor)

            nodes_to_contract = (mps_nodes[r], Lambda_inv)
            network_struct = [(-1, -2, 1), (1, -3)]
            Gammas[r] = tn.ncon(nodes_to_contract, network_struct)

    # kwargs = {"nodes": mps_nodes,
    #           "current_orthogonal_center_idx": 0,
    #           "compress_params": None}

    # for i in range(imax+1):
    #     kwargs["current_orthogonal_center_idx"] = i
    #     U, S, V_dagger = shift_orthogonal_center_to_the_right(**kwargs)
    #     Lambdas[(i+1)%L] = S

    # for i in range(imax+1):
    #     Lambda_i_tensor = Lambdas[i].tensor
    #     Lambda_i_inv_tensor = np.diag(1 / np.diag(np.array(Lambda_i_tensor)))
    #     Lambda_i_inv = tn.Node(Lambda_i_inv_tensor)

    #     nodes_to_contract = (Lambda_i_inv, mps_nodes[i])
    #     network_struct = [(-1, 1), (1, -2, -3)]
    #     Gammas[i] = tn.ncon(nodes_to_contract, network_struct)

    return Gammas, Lambdas



def Lambda_Theta_form(mps_nodes, starting_node_idx):
    # Used for infinite chains.
    left_to_right_sweep(nodes=mps_nodes,
                        compress_params=None,
                        is_infinite=False,  # Avoid doing the extra shift.
                        starting_node_idx=starting_node_idx)

    r0 = starting_node_idx
    L = len(mps_nodes)
    node = mps_nodes[(r0-1)%L]
    U, S, V_dagger = split_node_full(node=node,
                                     left_edges=(node[0], node[1]),
                                     right_edges=(node[2],),
                                     compress_params=None)

    Theta_nodes = []
    for r in range(r0, r0+L):
        Theta_nodes.append(mps_nodes[r%L])

    Theta_nodes[-1] = U
    Lambda = S
    Theta_nodes[0] = tn.ncon([V_dagger, mps_nodes[r0]], [(-1, 1), (1, -2, -3)])

    mps_nodes[(r0-1)%L] = U
    nodes_to_contract = [S, V_dagger, mps_nodes[r0]]
    network_struct = [(-1, 1), (1, 2), (2, -2, -3)]
    mps_nodes[r0] = tn.ncon(nodes_to_contract, network_struct)

    Lambda_Theta = [Lambda, Theta_nodes]

    return Lambda_Theta
