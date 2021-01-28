#!/usr/bin/env python
r"""For performing various TN operations involving SVD.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For general array handling.
import numpy as np

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



def _left_to_right_svd_sweep_across_mps(mps_nodes, trunc_params=None):
    truncated_schmidt_spectrum = []
        
    num_mps_nodes = len(mps_nodes)
    if num_mps_nodes == 1:
        return truncated_schmidt_spectrum
    
    for i in range(num_mps_nodes-1):
        node_i = mps_nodes[i]

        # Switch to numpy backend (if numpy is not being used) so that SVD can
        # be performed on CPUs as it is currently faster than on GPUs.
        original_backend_name = node_i.backend.name
        if original_backend_name != "numpy":
            _tf_to_np_backend(node_i)
        
        left_edges = (node_i[0], node_i[1])
        right_edges = (node_i[2],)

        U, S, V_dagger = _split_node_svd(node_i,
                                         left_edges,
                                         right_edges,
                                         trunc_params,
                                         original_backend_name)

        truncated_schmidt_spectrum.append(S)
        mps_nodes[i] = U

        node_iP1 = mps_nodes[i+1]
        nodes_to_contract = (S, V_dagger, node_iP1)
        network_struct = [(-1, 1), (1, 2), (2, -2, -3)]
        mps_nodes[i+1] = tn.ncon(nodes_to_contract, network_struct)

    return truncated_schmidt_spectrum



def _right_to_left_svd_sweep_across_mps(mps_nodes, trunc_params=None):
    truncated_schmidt_spectrum = []
    
    num_mps_nodes = len(mps_nodes)
    if num_mps_nodes == 1:
        return truncated_schmidt_spectrum
        
    for i in range(num_mps_nodes-1, 0, -1):
        node_i = mps_nodes[i]
        
        # Switch to numpy backend (if numpy is not being used) so that SVD can
        # be performed on CPUs as it is currently faster than on GPUs.
        original_backend_name = node_i.backend.name
        if original_backend_name != "numpy":
            _tf_to_np_backend(node_i)
        
        left_edges = (node_i[0],)
        right_edges = (node_i[1], node_i[2])

        U, S, V_dagger = _split_node_svd(node_i,
                                         left_edges,
                                         right_edges,
                                         trunc_params,
                                         original_backend_name)
        
        truncated_schmidt_spectrum.insert(0, S)
        mps_nodes[i] = V_dagger

        node_iM1 = mps_nodes[i-1]
        nodes_to_contract = (node_iM1, U, S)
        network_struct = [(-1, -2, 2), (2, 1), (1, -3)]
        mps_nodes[i-1] = tn.ncon(nodes_to_contract, network_struct)

    return truncated_schmidt_spectrum



def _split_node_svd(node,
                    left_edges,
                    right_edges,
                    trunc_params,
                    original_backend_name):
    if trunc_params == None:
        max_num_singular_values = None
        max_trunc_err = None
        rel_tol = None
    else:
        max_num_singular_values = trunc_params.max_num_singular_values
        max_trunc_err = trunc_params.max_trunc_err
        rel_tol = trunc_params.rel_tol

    U, S, V_dagger, _ = \
        tn.split_node_full_svd(node=node,
                               left_edges=left_edges,
                               right_edges=right_edges,
                               max_singular_values=max_num_singular_values,
                               max_truncation_err=max_trunc_err)

    U[-1] | S[0]  # Break edge between U and S nodes.
    S[-1] | V_dagger[0]  # Break edge between S and V_dagger.

    if rel_tol is not None:
        singular_vals = np.diag(S.tensor)
        max_singular_val = singular_vals[0]  # Already ordered.
        cutoff_idx = \
            np.where(singular_vals > max_singular_val * rel_tol)[0][-1] + 1

        U = tn.Node(U.tensor[..., :cutoff_idx])
        S = tn.Node(S.tensor[:cutoff_idx, :cutoff_idx])
        V_dagger = tn.Node(V_dagger.tensor[:cutoff_idx, ...])

    # Switch back to original backend (if different from numpy).
    if original_backend_name != "numpy":
        _np_to_tf_backend(U)
        _np_to_tf_backend(S)
        _np_to_tf_backend(V_dagger)

    return U, S, V_dagger



def _tf_to_np_backend(node):
    node.backend = tn.backends.backend_factory.get_backend("numpy")
    node.tensor = node.tensor.numpy()  # Converts to numpy array.

    return None



def _np_to_tf_backend(node):
    node.backend = tn.backends.backend_factory.get_backend("tensorflow")
    node.tensor = node.backend.convert_to_tensor(node.tensor)

    return None
