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



# Assign an alias to the ``spinbosonchain`` library.
import spinbosonchain as sbc

# For switching the ``tensornetwork`` backend.
import spinbosonchain._backend



############################
## Authorship information ##
############################

__author__     = "D-Wave Systems Inc."
__copyright__  = "Copyright 2021"
__credits__    = ["Matthew Fitzpatrick"]
__maintainer__ = "D-Wave Systems Inc."
__email__      = "support@dwavesys.com"
__status__     = "Development"



##################################
## Define classes and functions ##
##################################

def left_to_right_sweep(nodes, compress_params, normalize):
    # [1]: Annals of Physics 326 (2011) 96-192.
    # nodes represents a MPS or MPO.
    
    # This function brings a MPS or MPO [treated as a MPS] to left-canonical
    # form as described in Sec. 4.4.1 of [1]. In the case of a MPO, the two
    # physical edges are effectively combined into one during the procedure.
    # If toggled, the singular values at each bond will be truncated according
    # to the compression parameters given by compress_params.
    
    truncated_schmidt_spectra = []
        
    num_nodes = len(nodes)
    imin = 0
    imax = imin + (num_nodes - 2)

    # current_orthogonal_center_idx is the index of the node to which to apply
    # a SVD; normalize_schmidt_spectra [a bool] indicates whether to normalize
    # the resulting singular value spectra a.k.a. schmidt spectra such that
    # the sum of squares of said values equals unity.
    kwargs = {"nodes": nodes,
              "current_orthogonal_center_idx": imin,
              "compress_params": compress_params,
              "normalize_schmidt_spectra": normalize}
    
    for i in range(imin, imax+1):
        kwargs["current_orthogonal_center_idx"] = i
        U, S, V_dagger = shift_orthogonal_center_to_the_right(**kwargs)
        truncated_schmidt_spectra.append(S)

    if normalize:
        # Normalize MPS/MPO such that its 'norm' equal unity.
        nodes[-1] /= tn.norm(nodes[-1])

    return truncated_schmidt_spectra



def right_to_left_sweep(nodes, compress_params, normalize):
    # [1]: Annals of Physics 326 (2011) 96-192.
    # nodes represents a MPS or MPO.
    
    # This function brings a MPS or MPO [treated as a MPS] to right-canonical
    # form as described in Sec. 4.4.2 of [1]. In the case of a MPO, the two
    # physical edges are effectively combined into one during the procedure.
    # If toggled, the singular values at each bond will be truncated according
    # to the compression parameters given by compress_params.
    
    truncated_schmidt_spectra = []

    num_nodes = len(nodes)
    imax = num_nodes - 1
    imin = 1

    # current_orthogonal_center_idx is the index of the node to which to apply
    # a SVD; normalize_schmidt_spectra [a bool] indicates whether to normalize
    # the resulting singular value spectra a.k.a. schmidt spectra such that
    # the sum of squares of said values equals unity.
    kwargs = {"nodes": nodes,
              "current_orthogonal_center_idx": imax,
              "compress_params": compress_params,
              "normalize_schmidt_spectra": normalize}
    
    for i in range(imax, imin-1, -1):
        kwargs["current_orthogonal_center_idx"] = i
        U, S, V_dagger = shift_orthogonal_center_to_the_left(**kwargs)
        truncated_schmidt_spectra.insert(0, S)

    if normalize:
        # Normalize MPS/MPO such that its 'norm' equal unity.
        nodes[0] /= tn.norm(nodes[0])

    return truncated_schmidt_spectra



def shift_orthogonal_center_to_the_right(nodes,
                                         current_orthogonal_center_idx,
                                         compress_params,
                                         normalize_schmidt_spectra):
    # See comments in function left_to_right_sweep for more info/context.
    
    i = current_orthogonal_center_idx

    num_nodes = len(nodes)
    node_i = nodes[i%num_nodes]
    num_edges_per_node = len(node_i.edges)
    left_edges = tuple(node_i[idx] for idx in range(num_edges_per_node-1))
    right_edges = (node_i[num_edges_per_node-1],)

    # Perform SVD on node_i.
    U, S, V_dagger = split_node_full(node_i,
                                     left_edges,
                                     right_edges,
                                     compress_params)

    if normalize_schmidt_spectra:
        S /= tn.norm(S)

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
                                        compress_params,
                                        normalize_schmidt_spectra):
    # See comments in function right_to_left_sweep for more info/context.
    
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

    # Perform SVD on node_i.
    if normalize_schmidt_spectra:
        S /= tn.norm(S)
        
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
    # Perform a SVD on node. See documentation for
    # spinbosonchain.compress.Params for more info/context.
    
    # Switch to numpy backend [if numpy is not being used] so that SVD can
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
        # Call the slower but more stable SVD subroutine if the default fails.
        U, S, V_dagger, discarded_singular_values = \
            split_node_full_backup(**kwargs)

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
        sbc._backend.np_to_tf(U)
        sbc._backend.np_to_tf(S)
        sbc._backend.np_to_tf(V_dagger)

    return U, S, V_dagger



def split_node_full_backup(node,
                           left_edges,
                           right_edges,
                           max_singular_values,
                           max_truncation_err):
    # Perform a SVD on node. See documentation for
    # spinbosonchain.compress.Params for more info/context. This SVD routine is
    # more stable albeit slower than the implementation of Google's
    # tensornetwork library.
    
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

    discarded_singular_values = S[cutoff_idx:]
    U = tn.Node(U[:, :cutoff_idx].reshape(new_U_shape))
    S = tn.Node(np.diag(S[:cutoff_idx]))
    V_dagger = tn.Node(V_dagger[:cutoff_idx, :].reshape(new_V_dagger_shape))

    return U, S, V_dagger, discarded_singular_values



def Lambda_Theta_form(mps_nodes):
    # [2]: arXiv.0804.2509
    
    # This function is used to bring a set of MPS nodes of a unit cell,
    # representing an infinite system, to what we will refer to as the
    # 'Lambda-Theta' form. If the 'Vidal' form [see e.g. Eq. (7) of [2]] of the
    # unit cell can be expressed as:
    #
    #     Lambda_0 Gamma_0 Lambda_1 Gamma_1 ... Lambda_{L-1} Gamma_{L-1},
    #
    # where L is the unit cell size, then the corresponding 'Lambda-Theta' form
    # is:
    #
    #     Lambda Theta,
    #
    # where
    #
    #     Lambda = Lambda_0,
    #     Theta = Gamma_0 Lambda_1 Gamma_1 ... Lambda_{L-1} Gamma_{L-1}.
    
    # This SVD is not strictly necessary, however numerical tests suggests it
    # improves overall stability.
    left_to_right_sweep(nodes=mps_nodes, compress_params=None, normalize=False)

    L = len(mps_nodes)
    node = mps_nodes[-1]
    U, S, V_dagger = split_node_full(node=node,
                                     left_edges=(node[0], node[1]),
                                     right_edges=(node[2],),
                                     compress_params=None)

    Theta_nodes = []

    # Keep in mind that infinite chains with unit cells are effectively periodic
    # systems, hence why the MPS nodes can be moved around in a cyclic fashion.
    if L == 1:
        Lambda = S
        Theta_node = tn.ncon([V_dagger, U], [(-1, 1), (1, -2, -3)])
        Theta_nodes.append(Theta_node)
        mps_nodes[0] = tn.ncon([Lambda, Theta_node], [(-1, 1), (1, -2, -3)])
        Lambda_Theta = [Lambda, Theta_nodes]
        
        return Lambda_Theta

    for r in range(L):
        Theta_nodes.append(mps_nodes[r])

    Theta_nodes[-1] = U
    Lambda = S
    Theta_nodes[0] = tn.ncon([V_dagger, mps_nodes[0]], [(-1, 1), (1, -2, -3)])

    mps_nodes[-1] = U
    mps_nodes[0] = tn.ncon([S, Theta_nodes[0]], [(-1, 1), (1, -2, -3)])

    Lambda_Theta = [Lambda, Theta_nodes]

    return Lambda_Theta
