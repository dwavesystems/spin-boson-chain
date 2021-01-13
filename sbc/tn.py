#!/usr/bin/env python
r"""Defines classes and functions related to tensor networks.
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

# List of public objects in objects.
__all__ = ["TruncParams"]



class TruncParams():
    r"""Parameters specifying how to truncate Schmidt spectrum in a SVD.

    ``sbc`` represents the system state and the local path influence functionals
    (see documentation for the module :mod:`sbc.state` for further discussion of
    the local path influence functionals) by set of matrix product states 
    (MPS's). As time evolves, the bond dimensions of the MPS's representing the 
    aforementioned objects will generally increase, thus increasing memory 
    requirements and the simulation runtime. To counter this, one can perform 
    singular value decomposition (SVD) truncations to compress efficiently the 
    MPS's. In short, increasing and decreasing the parameters 
    ``max_num_singular_values`` and ``max_trunc_err`` respectively translate to 
    MPS's with larger bond dimensions :math:`\chi` which generally translates to
    a decrease in the numerical errors of the simulation. However, since the 
    most computationally intensive parts of the simulation scale like 
    :math:`\chi^3`, increasing and decreasing the parameters 
    ``max_num_singular_values`` and ``max_trunc_err`` respectively lead to 
    longer runtimes.

    After the Schmidt spectrum has been truncated according to the parameters
    ``max_num_singular_values`` and ``max_trunc_err``, one can perform an
    additional truncation step by setting the parameter ``rel_tol`` to a
    positive number. If ``rel_tol`` is set as such, then ``sbc`` will discard
    singular values smaller than ``rel_tol * s_max``, where ``s_max`` is the
    largest singular value after the initial truncation procedure has been
    performed.

    Parameters
    ----------
    max_num_singular_values : `None` | `int`, optional
        The maximum number of singular values to keep in a SVD.
    max_trunc_err : `None` | `float`, optional
        The maximum allowed truncation error in a SVD truncation. If 
        ``max_num_singular_values`` is set to an `int`, and the truncation error
        cannot go below ``max_trunc_err`` without keeping more than
        ``max_num_singular_values`` singular values, then the parameter
        ``max_trunc_err`` is ignored while performing the SVD truncation.
    rel_tol : `None` | `float`, optional
        The relative tolerance after the initial truncation procedure has been
        performed. If not set to `None`, ``rel_tol`` is expected to be positive.

    Attributes
    ----------
    max_num_singular_values : `None` | `int`, read-only
        The maximum number of singular values to keep in a SVD.
    max_trunc_err : `None` | `float`, read-only
        The maximum allowed truncation error in a SVD truncation. If 
        ``max_num_singular_values`` is set to an `int`, and the truncation error
        cannot go below ``max_trunc_err`` without keeping more than
        ``max_num_singular_values`` singular values, then the attribute
        ``max_trunc_err`` is ignored while performing the SVD truncation.
    rel_tol : `None` | `float`, read-only
        The relative tolerance after the initial truncation procedure has been
        performed. If not set to `None`, ``rel_tol`` is expected to be positive.
    """
    def __init__(self,
                 max_num_singular_values=None,
                 max_trunc_err=None,
                 rel_tol=None):
        if max_num_singular_values == None:
            self.max_num_singular_values = None
        else:
            if max_num_singular_values < 1:
                raise ValueError("The parameter `max_num_singular_values` must "
                                 "be a positive integer or set to type `None`.")

        if max_trunc_err == None:
            self.max_trunc_err = None
        else:
            if max_trunc_err < 0:
                raise ValueError("The parameter `max_trunc_err` must be a "
                                 "non-negative number or set to type `None`.")

        if rel_tol == None:
            self.rel_tol = None
        else:
            if rel_tol <= 0:
                raise ValueError("The parameter `rel_tol` must be a "
                                 "positive number or set to type `None`.")
        
        self.max_num_singular_values = max_num_singular_values
        self.max_trunc_err = max_trunc_err
        self.rel_tol = rel_tol

        return None



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



def _tf_to_np_backend(node):
    node.backend = tn.backends.backend_factory.get_backend("numpy")
    node.tensor = node.tensor.numpy()  # Converts to numpy array.

    return None



def _np_to_tf_backend(node):
    node.backend = tn.backends.backend_factory.get_backend("tensorflow")
    node.tensor = node.backend.convert_to_tensor(node.tensor)

    return None
