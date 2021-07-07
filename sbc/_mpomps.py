#!/usr/bin/env python
r"""Contains an implementation for applying MPO's to MPS's.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For deep copies of objects.
import copy



# For general array handling.
import numpy as np

# For calculating the eigenspectra of Hermitian matrices.
import scipy.linalg

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# For performing SVD truncation sweeps, shifting orthogonal centers, QR
# factorizations, and single-node SVD.
import sbc._svd

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

def apply_mpo_to_mps_and_compress(mpo_nodes,
                                  mps_nodes,
                                  compress_params,
                                  is_infinite):
    if is_infinite:
        # In this case, ``mps_nodes = (Gamma, S)``, where ``Gamma`` and ``S``
        # are the 'Gamma' and 'S' nodes representing the infinite MPS.
        apply_infinite_mpo_to_infinite_mps_and_compress(mpo_nodes,
                                                        mps_nodes,
                                                        compress_params)
    else:
        # In this case, ``mps_nodes`` is a sequence of left-normalized MPS
        # nodes.
        apply_finite_mpo_to_finite_mps_and_compress(mpo_nodes,
                                                    mps_nodes,
                                                    compress_params)
        
    return None



def apply_infinite_mpo_to_infinite_mps_and_compress(mpo_nodes,
                                                    mps_nodes,
                                                    compress_params):
    # apply_directly_infinite_mpo_to_infinite_mps(mpo_nodes, mps_nodes)
    apply_directly_mpo_to_mps(mpo_nodes, mps_nodes)
    # canonicalize_and_compress_infinite_mps(mps_nodes,
    #                                        compress_params)

    # mps_node = mps_nodes[0]
    # kwargs = {"node": mps_nodes[0],
    #           "left_edges": (mps_node[0], mps_node[1]),
    #           "right_edges": (mps_node[2],),
    #           "compress_params": compress_params}
    # U, S, V_dagger = sbc._svd.split_node_full_svd(**kwargs)
        
    # sqrt_S = tn.Node(np.sqrt(S.tensor))
    # nodes_to_contract = [sqrt_S, V_dagger, U, sqrt_S]
    # network_struct = [(-1, 1), (1, 2), (2, -2, 3), (3, -3)]
    # mps_nodes[0] = tn.ncon(nodes_to_contract, network_struct)

    sbc._svd.left_to_right_svd_sweep(mps_nodes,
                                     compress_params=None,
                                     is_infinite=True)
    sbc._svd.right_to_left_svd_sweep(mps_nodes,
                                     compress_params=compress_params,
                                     is_infinite=True)

    return None



def apply_finite_mpo_to_finite_mps_and_compress(mpo_nodes,
                                                mps_nodes,
                                                compress_params):
    # Put MPO in left-canonical form. MPS is assumed to be already in said form.
    sbc._svd.left_to_right_svd_sweep(mpo_nodes,
                                     compress_params=None,
                                     is_infinite=False)

    if compress_params.max_num_var_sweeps > 0:
        norm_of_mps_to_compress = \
            norm_of_mpo_mps_network_wo_compression(mpo_nodes, mps_nodes)
        initial_mps_nodes = copy.deepcopy(mps_nodes)
    else:
        initial_mps_nodes = None

    if compress_params.method == "direct":
        # apply_directly_finite_mpo_to_finite_mps(mpo_nodes, mps_nodes)
        apply_directly_mpo_to_mps(mpo_nodes, mps_nodes)
        sbc._svd.right_to_left_svd_sweep(mps_nodes,
                                         compress_params=None,
                                         is_infinite=False)
    else:
        zip_up(mpo_nodes, mps_nodes, compress_params)

    # Perform a return SVD truncation sweep. If the SVD truncation sweep is
    # followed by variational compression, then L-nodes are calculated and
    # cached while performing the SVD truncation sweep for efficiency. Note
    # that the 'L'- and 'R'-nodes here are those used in variational
    # compression. This is different from the 'L' and 'R' nodes used to
    # bring an iMPS to canonical form.
    kwargs = {"mpo_nodes": mpo_nodes,
              "mps_nodes": mps_nodes,
              "initial_mps_nodes": initial_mps_nodes,
              "compress_params": compress_params}
    L_cache = return_svd_sweep(**kwargs)

    if compress_params.max_num_var_sweeps > 0:
        kwargs["norm_of_mps_to_compress"] = norm_of_mps_to_compress
        kwargs["L_cache"] = L_cache
        variational_compression(**kwargs)

    return None



def norm_of_mpo_mps_network_wo_compression(mpo_nodes, mps_nodes):
    # Here, 'R' refers to an R-node used in a variational compression.
    imax = len(mps_nodes) - 1
    R = trivial_R(mpo_nodes, mps_nodes, num_legs=4)  # Right-most R-node.
    
    for i in range(imax, -1, -1):
        MWWR = contract_MWWR_network(mps_nodes[i], mpo_nodes[i], R)
        conj_mps_node = tn.conj(mps_nodes[i])
        conj_mps_node[1] ^ MWWR[3]
        conj_mps_node[2] ^ MWWR[4]
        output_edge_order = (MWWR[0], MWWR[1], MWWR[2], conj_mps_node[0])
        R = tn.contract_between(node1=conj_mps_node,
                                node2=MWWR,
                                output_edge_order=output_edge_order)

    nodes_to_contract = [R]
    network_struct = [(1, 2, 2, 1)]
    result = tn.ncon(nodes_to_contract, network_struct)
    result = float(np.sqrt(np.real(result.tensor)))

    return result



def trivial_R(mpo_nodes, mps_nodes, num_legs):
    # Here, 'R' refers to an R-node used in a variational compression.
    w_r = mpo_nodes[-1].shape[-1]  # w_r: right dangling mpo bond dimension.
    chi_r = mps_nodes[-1].shape[-1]  # chi_r: right dangling mps bond dimension.
    if num_legs == 3:
        tensor = np.zeros([chi_r, w_r, chi_r*w_r], dtype=np.complex128)
        for m2 in range(chi_r*w_r):
            m1 = 0 if chi_r == 1 else m2
            w = 0 if w_r == 1 else m2
            tensor[m1, w, m2] = 1
        R = tn.Node(tensor)
    else:
        tensor = np.zeros([chi_r, w_r, w_r, chi_r], dtype=np.complex128)
        for m in range(chi_r):
            for w in range(w_r):
                tensor[m, w, w, m] = 1
        R = tn.Node(tensor)

    return R



def trivial_L(mpo_nodes, mps_nodes):
    # Here, 'L' refers to an L-node used in a variational compression.
    if mps_nodes is None:
        L = None
        return L
    
    w_l = mpo_nodes[0].shape[0]  # w_l: left dangling mpo bond dimension.
    chi_l = mps_nodes[0].shape[0]  # chi_l: left dangling mps bond dimension.
    tensor = np.zeros([chi_l, w_l, chi_l*w_l], dtype=np.complex128)
    for m2 in range(chi_l*w_l):
        m1 = 0 if chi_l == 1 else m2
        w = 0 if w_l == 1 else m2
        tensor[m1, w, m2] = 1
    L = tn.Node(tensor)

    return L



def contract_MR_network(M, R, num_R_legs=3):
    # Here, 'R' refers to an R-node used in a variational compression.
    nodes_to_contract = [M, R]
    if num_R_legs == 3:
        network_struct = [(-1, -2, 1), (1, -3, -4)]
    else:
        network_struct = [(-1, -2, 1), (1, -3, -4, -5)]
    MR = tn.ncon(nodes_to_contract, network_struct)

    return MR



def contract_MWR_network(M, W, R, num_R_legs=3):
    # Here, 'R' refers to an R-node used in a variational compression.
    MR = contract_MR_network(M, R, num_R_legs)

    W[2] ^ MR[1]
    W[3] ^ MR[2]
    if num_R_legs == 3:
        output_edge_order = (MR[0], W[0], W[1], MR[3])
        
    else:
        output_edge_order = (MR[0], W[0], W[1], MR[3], MR[4])
    MWR = tn.contract_between(node1=W,
                              node2=MR,
                              output_edge_order=output_edge_order)
    
    return MWR



def contract_MWWR_network(M, W, R):
    # Here, 'R' refers to an R-node used in a variational compression.
    MWR = contract_MWR_network(M, W, R, num_R_legs=4)
    W = tn.conj(W)

    W[2] ^ MWR[2]
    W[3] ^ MWR[3]
    output_edge_order = (MWR[0], MWR[1], W[0], W[1], MWR[4])
    MWWR = tn.contract_between(node1=W,
                               node2=MWR,
                               output_edge_order=output_edge_order)
    
    return MWWR



def contract_LM_network(L, M):
    # Here, 'L' refers to an L-node used in a variational compression.
    nodes_to_contract = [L, M]
    network_struct = [(1, -3, -4), (1, -2, -1)]
    LM = tn.ncon(nodes_to_contract, network_struct)

    return LM



def contract_LMW_network(L, M, W):
    # Here, 'L' refers to an R-node used in a variational compression.
    LM = contract_LM_network(L, M)

    LM[1] ^ W[2]
    LM[2] ^ W[0]
    output_edge_order = (LM[0], W[3], W[1], LM[3])
    LMW = tn.contract_between(node1=LM,
                              node2=W,
                              output_edge_order=output_edge_order)
    
    return LMW



def zip_up(mpo_nodes, mps_nodes, compress_params):
    imax = len(mps_nodes) - 1
    U = None
    S = None

    compress_params = copy.deepcopy(compress_params)
    if compress_params.max_num_singular_values is not None:
        compress_params.max_num_singular_values *= 2
    if compress_params.max_trunc_err is not None:
        compress_params.max_trunc_err /= 10
    if compress_params.svd_rel_tol is not None:
        compress_params.svd_rel_tol /= 10

    for i in range(imax, -1, -1):
        node_idx = i
        U, S = update_mps_node_in_zip_up(node_idx,
                                         mpo_nodes,
                                         mps_nodes,
                                         U,
                                         S,
                                         compress_params)

    return None



def update_mps_node_in_zip_up(node_idx,
                              mpo_nodes,
                              mps_nodes,
                              U,
                              S,
                              compress_params):
    i = node_idx
    imax = len(mps_nodes) - 1    
    mpo_node = mpo_nodes[i]
    mps_node = mps_nodes[i]

    if i == imax:
        nodes_to_contract = (mpo_node, mps_node)
        network_struct = [(-1, -3, 1, -4), (-2, 1, -5)]
        temp_node_2 = tn.ncon(nodes_to_contract, network_struct)
        tn.flatten_edges([temp_node_2[3], temp_node_2[4]])
    elif 0 <= i < imax:
        nodes_to_contract = (mps_node, U, S)
        network_struct = [(-1, -2, 2), (-3, 2, 1), (1, -4)]
        temp_node_1 = tn.ncon(nodes_to_contract, network_struct)
        temp_node_1[1] ^ mpo_node[2]
        temp_node_1[2] ^ mpo_node[3]
        output_edge_order = \
            (mpo_node[0], temp_node_1[0], mpo_node[1], temp_node_1[3])
        temp_node_2 = tn.contract_between(node1=temp_node_1,
                                          node2=mpo_node,
                                          output_edge_order=output_edge_order)

    if 0 < i <= imax:
        left_edges = (temp_node_2[0], temp_node_2[1])
        right_edges = (temp_node_2[2], temp_node_2[3])
        U, S, V_dagger = sbc._svd.split_node_full_svd(temp_node_2,
                                                      left_edges,
                                                      right_edges,
                                                      compress_params)
        mps_nodes[i] = V_dagger
    elif i == 0:
        tn.flatten_edges([temp_node_2[0], temp_node_2[1]])
        temp_node_2.reorder_edges([temp_node_2[2],
                                   temp_node_2[0],
                                   temp_node_2[1]])
        mps_nodes[i] = temp_node_2

    return U, S



def return_svd_sweep(mpo_nodes, mps_nodes, initial_mps_nodes, compress_params):
    # Here, 'L' refers to L-nodes used in a variational compression.
    imin = 0
    imax = len(mps_nodes) - 2
    L = trivial_L(mpo_nodes, initial_mps_nodes)
    L_cache = [L]
    
    for i in range(imin, imax+1):
        kwargs = {"nodes": mps_nodes,
                  "current_orthogonal_center_idx": i,
                  "compress_params": compress_params}
        sbc._svd.shift_orthogonal_center_to_the_right(**kwargs)
        
        if compress_params.max_num_var_sweeps > 0:
            M = initial_mps_nodes[i]
            W = mpo_nodes[i]
            LMW = contract_LMW_network(L, M, W)
            conj_mps_node = tn.conj(mps_nodes[i])
            LMW[2] ^ conj_mps_node[1]
            LMW[3] ^ conj_mps_node[0]
            output_edge_order = (LMW[0], LMW[1], conj_mps_node[2])
            L = tn.contract_between(node1=LMW,
                                    node2=conj_mps_node,
                                    output_edge_order=output_edge_order)
            L_cache.append(L)

    return L_cache



def variational_compression(mpo_nodes,
                            mps_nodes,
                            initial_mps_nodes,
                            norm_of_mps_to_compress,
                            L_cache,
                            compress_params):
    # Here, 'R' and 'L' refers to R- and L-nodes used in a variational
    # compression.
    imax = len(mps_nodes) - 1
    R_cache = [trivial_R(mpo_nodes, initial_mps_nodes, num_legs=3)]

    # A full sweep goes right to left, then left to right.
    sweep_count = 0
    while sweep_count < compress_params.max_num_var_sweeps:
        kwargs = {"mpo_nodes": mpo_nodes,
                  "mps_nodes": mps_nodes,
                  "initial_mps_nodes": initial_mps_nodes,
                  "norm_of_mps_to_compress": norm_of_mps_to_compress,
                  "L_cache": L_cache,
                  "R_cache": R_cache,
                  "compress_params": compress_params}
        
        if variational_compression_has_converged(**kwargs):
            break

        kwargs = {"mpo_nodes": mpo_nodes,
                  "mps_nodes": mps_nodes,
                  "initial_mps_nodes": initial_mps_nodes,
                  "L_cache": L_cache,
                  "R_cache": R_cache}
        
        for i in range(imax, 0, -1):
            update_node_and_shift_left_in_variational_compression(**kwargs)
        for i in range(imax):
            update_node_and_shift_right_in_variational_compression(**kwargs)

        sweep_count += 1

    return None



def variational_compression_has_converged(mpo_nodes,
                                          mps_nodes,
                                          initial_mps_nodes,
                                          norm_of_mps_to_compress,
                                          L_cache,
                                          R_cache,
                                          compress_params):
    # Here, 'R' and 'L' refers to R- and L-nodes used in a variational
    # compression.    
    overlap = overlap_btwn_compressed_and_uncompressed_mps(L_cache,
                                                           initial_mps_nodes,
                                                           mpo_nodes,
                                                           mps_nodes,
                                                           R_cache)

    nodes_to_contract = (mps_nodes[-1], tn.conj(mps_nodes[-1]))
    network_struct = [(1, 2, 3), (1, 2, 3)]
    norm_sq_of_compressed_mps = tn.ncon(nodes_to_contract, network_struct)
    norm_of_compressed_mps = \
        float(np.sqrt(np.real(norm_sq_of_compressed_mps.tensor)))

    var_rel_err_sq = (np.abs(norm_of_mps_to_compress * norm_of_mps_to_compress
                             - 2 * np.real(overlap)
                             + norm_of_compressed_mps * norm_of_compressed_mps)
                      / norm_of_mps_to_compress)
    var_rel_err = np.sqrt(var_rel_err_sq)

    if var_rel_err < compress_params.var_rel_tol:
        result = True
    else:
        result = False

    return result



def overlap_btwn_compressed_and_uncompressed_mps(L_cache,
                                                 initial_mps_nodes,
                                                 mpo_nodes,
                                                 mps_nodes,
                                                 R_cache):
    # Here, 'R' and 'L' refers to R- and L-nodes used in a variational
    # compression.
    L = L_cache[-1]
    M = initial_mps_nodes[-1]
    W = mpo_nodes[-1]
    R = R_cache[0]

    MWR = contract_MWR_network(M, W, R)
    
    L[0] ^ MWR[0]
    L[1] ^ MWR[1]
    output_edge_order = (L[2], MWR[2], MWR[3])
    LMWR = tn.contract_between(node1=L,
                               node2=MWR,
                               output_edge_order=output_edge_order)
    
    conj_mps_node = tn.conj(mps_nodes[-1])

    for edge_idx in range(3):
        conj_mps_node[edge_idx] ^ LMWR[edge_idx]

    overlap = tn.contract_between(node1=conj_mps_node, node2=LMWR)
    overlap = complex(overlap.tensor)

    return overlap



def update_node_and_shift_left_in_variational_compression(mpo_nodes,
                                                          mps_nodes,
                                                          initial_mps_nodes,
                                                          L_cache,
                                                          R_cache):
    # Here, 'R' and 'L' refers to R- and L-nodes used in a variational
    # compression.
    orthogonal_center_idx = len(L_cache) - 1
    L = L_cache[-1]
    M = initial_mps_nodes[orthogonal_center_idx]
    W = mpo_nodes[orthogonal_center_idx]
    R = R_cache[0]

    MWR = contract_MWR_network(M, W, R)
        
    L[0] ^ MWR[0]
    L[1] ^ MWR[1]
    output_edge_order = (L[2], MWR[2], MWR[3])
    LMWR = tn.contract_between(node1=L,
                               node2=MWR,
                               output_edge_order=output_edge_order)
    
    L_cache.pop()

    QR = tn.conj(LMWR)
    left_node, right_node = sbc._svd.split_node_qr(node=QR,
                                                   left_edges=(QR[1], QR[2]),
                                                   right_edges=(QR[0],))
    new_edge_order = [left_node[2], left_node[0], left_node[1]]
    B = tn.conj(left_node.reorder_edges(new_edge_order))
    mps_nodes[orthogonal_center_idx] = B
    conj_B = tn.conj(B)
        
    conj_B[1] ^ MWR[2]
    conj_B[2] ^ MWR[3]
    output_edge_order = (MWR[0], MWR[1], conj_B[0])
    next_R = tn.contract_between(node1=conj_B,
                                 node2=MWR,
                                 output_edge_order=output_edge_order)
    R_cache.insert(0, next_R)

    return None



def update_node_and_shift_right_in_variational_compression(mpo_nodes,
                                                           mps_nodes,
                                                           initial_mps_nodes,
                                                           L_cache,
                                                           R_cache):
    # Here, 'R' and 'L' refers to R- and L-nodes used in a variational
    # compression.
    orthogonal_center_idx = len(L_cache) - 1
    L = L_cache[-1]
    M = initial_mps_nodes[orthogonal_center_idx]
    W = mpo_nodes[orthogonal_center_idx]
    R = R_cache[0]
    
    LMW = contract_LMW_network(L, M, W)

    LMW[0] ^ R[0]
    LMW[1] ^ R[1]
    output_edge_order = (LMW[3], LMW[2], R[2])
    LMWR = tn.contract_between(node1=LMW,
                               node2=R,
                               output_edge_order=output_edge_order)
    R_cache.pop(0)

    QR = LMWR
    left_node, right_node = sbc._svd.split_node_qr(node=QR,
                                                   left_edges=(QR[0], QR[1]),
                                                   right_edges=(QR[2],))
    A = left_node
    mps_nodes[orthogonal_center_idx] = A
    conj_A = tn.conj(A)

    if orthogonal_center_idx == len(mps_nodes) - 2:
        nodes_to_contract = [right_node, mps_nodes[-1]]
        network_struct = [(-1, 1), (1, -2, -3)]
        mps_nodes[-1] = tn.ncon(nodes_to_contract, network_struct)
        
    LMW[2] ^ conj_A[1]
    LMW[3] ^ conj_A[0]
    output_edge_order = (LMW[0], LMW[1], conj_A[2])
    next_L = tn.contract_between(node1=LMW,
                                 node2=conj_A,
                                 output_edge_order=output_edge_order)
    L_cache.append(next_L)

    return None



def apply_directly_mpo_to_mps(mpo_nodes, mps_nodes):
    num_mps_nodes = len(mps_nodes)
    new_mps_nodes = []
    for idx, (mpo_node, mps_node) in enumerate(zip(mpo_nodes, mps_nodes)):
        new_mps_node = apply_directly_mpo_node_to_mps_node(mpo_node, mps_node)
        mps_nodes[idx] = new_mps_node

    return None



def apply_directly_finite_mpo_to_finite_mps(mpo_nodes, mps_nodes):
    num_mps_nodes = len(mps_nodes)
    new_mps_nodes = []
    for idx, (mpo_node, mps_node) in enumerate(zip(mpo_nodes, mps_nodes)):
        new_mps_node = apply_directly_mpo_node_to_mps_node(mpo_node, mps_node)
        mps_nodes[idx] = new_mps_node

    return None



def apply_directly_infinite_mpo_to_infinite_mps(mpo_nodes, mps_nodes):
    mpo_node = mpo_nodes[0]  # ``sbc`` only handles single-site unit cells.
    Gamma, S = mps_nodes
    
    sqrt_S = tn.Node(np.sqrt(S.tensor))

    M = tn.ncon((sqrt_S, Gamma, sqrt_S), ((-1, 1), (1, -2, 2), (2, -3)))
    M = apply_directly_mpo_node_to_mps_node(mpo_node, M)
    kwargs = {"node": M,
              "left_edges": (M[0], M[1]),
              "right_edges": (M[2],),
              "compress_params": None}
    U, S, V_dagger = sbc._svd.split_node_full_svd(**kwargs)
    Gamma = tn.ncon((V_dagger, U), ((-1, 1), (1, -2, -3)))
    # Gamma = apply_directly_mpo_node_to_mps_node(mpo_node, Gamma)

    # w_r = mpo_node.shape[0]  # w_r: mpo bond dimension.
    # chi_r = S.shape[0]  # chi_r: mps bond dimension.
    
    # S_tensor = S.tensor.numpy() if S.backend.name != "numpy" else S.tensor
    # new_S_tensor = np.zeros([w_r, chi_r, w_r, chi_r], dtype=np.complex128)
    
    # for w in range(w_r):
    #     for m in range(chi_r):
    #         new_S_tensor[w, m, w, m] = S_tensor[m, m]

    # S = tn.Node(new_S_tensor)
            
    # tn.flatten_edges([S[0], S[1]])
    # tn.flatten_edges([S[0], S[1]])

    mps_nodes[0] = Gamma
    mps_nodes[1] = S

    return None



def apply_directly_mpo_node_to_mps_node(mpo_node, mps_node):
    nodes_to_contract = (mpo_node, mps_node)
    network_struct = [(-1, -3, 1, -4), (-2, 1, -5)]
    new_mps_node = tn.ncon(nodes_to_contract, network_struct)

    tn.flatten_edges([new_mps_node[0], new_mps_node[1]])
    tn.flatten_edges([new_mps_node[1], new_mps_node[2]])
    new_mps_node.reorder_edges([new_mps_node[1],
                                new_mps_node[0],
                                new_mps_node[2]])

    return new_mps_node



# def canonicalize_and_compress_infinite_mps(mps_nodes, compress_params):
#     # The procedure below is described in detail in PRB 91, 115137 (2015).
#     Gamma, S = mps_nodes
#     last_S = None

#     try:
#         while S_has_not_converged(last_S, S):
#             last_S = S
#             update_Gamma_and_S(mps_nodes, compress_params=None)
#             S = mps_nodes[1]
#         update_Gamma_and_S(mps_nodes, compress_params=compress_params)

#     except np.linalg.LinAlgError:
#         # If the algorithm fails to canonicalize the MPS and the spatial bond
#         # dimension has increased to its threshold, then the canonicalization
#         # step is simply skipped. Otherwise, the simulation terminates
#         # unsuccessfully.
#         print(S.shape[0], compress_params.max_num_singular_values)
#         if S.shape[0] > compress_params.max_num_singular_values:
#             msg = _orthogonalize_and_compress_infinite_mps_err_msg_1
#             raise np.linalg.LinAlgError(msg)

#     print("Successful call")

#     return None



# def update_Gamma_and_S(mps_nodes, compress_params):
#     Gamma, S = mps_nodes
    
#     V_R = calc_V_R(Gamma, S)
#     V_L = calc_V_L(Gamma, S)

#     X, X_inv = calc_X_and_X_inv(V_R)
#     Y_T, Y_T_inv = calc_Y_T_and_Y_T_inv(V_L)

#     node = tn.ncon((Y_T, S, X), ((-1, 1), (1, 2), (2, -2)))
#     kwargs = {"node": node,
#               "left_edges": (node[0],),
#               "right_edges": (node[1],),
#               "compress_params": compress_params}
#     U, S, V_dagger = sbc._svd.split_node_full_svd(**kwargs)
        
#     nodes_to_contract = (V_dagger, X_inv, Gamma, Y_T_inv, U)
#     network_struct = [(-1, 1), (1, 3), (3, -2, 4), (4, 2), (2, -3)]
#     Gamma = tn.ncon(nodes_to_contract, network_struct)

#     mps_nodes[0] = Gamma
#     mps_nodes[1] = S

#     return None



# def S_has_not_converged(last_S, S):
#     if last_S is None:
#         result = True
#     else:
#         epsilon_D = 1.0e-14
#         print("relative error =", tn.norm(S-last_S) / tn.norm(S))
#         if tn.norm(S-last_S) / tn.norm(S) < epsilon_D:
#             result = False
#         else:
#             result = True

#     return result



# def calc_V_R(Gamma, S):
#     M = tn.ncon((Gamma, S), ((-1, -2, 1), (1, -3)))
#     conj_M = tn.conj(M)
#     M[1] ^ conj_M[1]
#     M[2] ^ conj_M[2]
#     V_R = tn.contract_between(node1=M,
#                               node2=conj_M,
#                               output_edge_order=(M[0], conj_M[0]))

#     return V_R



# def calc_V_L(Gamma, S):
#     M = tn.ncon((S, Gamma), ((-1, 1), (1, -2, -3)))
#     conj_M = tn.conj(M)
#     M[0] ^ conj_M[0]
#     M[1] ^ conj_M[1]
#     V_L = tn.contract_between(node1=M,
#                               node2=conj_M,
#                               output_edge_order=(M[2], conj_M[2]))

#     return V_L



# def calc_X_and_X_inv(V_R):
#     if V_R.backend.name != "numpy":
#         sbc._backend.tf_to_np(V_R)
        
#     D, W = scipy.linalg.eigh(V_R.tensor)
#     X = W @ np.sqrt(np.diag(np.abs(D)))
#     X_inv = tn.Node(np.linalg.inv(X))
#     print("X =", X)
#     print("X_inv =", X_inv)
#     X = tn.Node(X)

#     return X, X_inv



# def calc_Y_T_and_Y_T_inv(V_L):
#     if V_L.backend.name != "numpy":
#         sbc._backend.tf_to_np(V_L)
        
#     D, W = scipy.linalg.eigh(V_L.tensor)
#     Y_T = np.transpose(W @ np.sqrt(np.diag(np.abs(D))))
#     Y_T_inv = tn.Node(np.linalg.inv(Y_T))
#     print("Y_T =", Y_T)
#     print("Y_T_inv =", Y_T_inv)
#     Y_T = tn.Node(Y_T)

#     return Y_T, Y_T_inv
    


def canonicalize_and_compress_infinite_mps(mps_nodes, compress_params):
    # The procedure below is described in detail in PRB 78, 155117 (2008).
    Gamma, S = mps_nodes
    original_backend_name = S.backend.name

    # Note that 'R' and 'L' take on the definitions given in PRB 78, 155117
    # (2008).
    # V_R_0 = system_state_pkl_part.V_R
    V_R_0 = None
    V_R = _right_dominant_eigvec_of_R(Gamma, S, V_R_0)
    # V_L_0 = system_state_pkl_part.V_L
    V_L_0 = None
    V_L = _left_dominant_eigvec_of_L(Gamma, S, V_L_0)

    if original_backend_name != "numpy":
        sbc._backend.tf_to_np(V_R)
        sbc._backend.tf_to_np(V_L)
            
    D, W = scipy.linalg.eigh(V_R.tensor)
    X = W @ np.sqrt(np.diag(np.abs(D)))
    # print("W =", W)
    # print("D =", D)
    # print("X =", X)
    X_inv = tn.Node(np.linalg.inv(X))
    X = tn.Node(X)

    D, W = scipy.linalg.eigh(V_L.tensor)
    Y_T = np.transpose(W @ np.sqrt(np.diag(np.abs(D))))
    # print("W =", W)
    # print("D =", D)
    # print("Y_T =", Y_T)
    Y_T_inv = tn.Node(np.linalg.inv(Y_T))
    Y_T = tn.Node(Y_T)

    node = tn.ncon((Y_T, S, X), ((-1, 1), (1, 2), (2, -2)))
    kwargs = {"node": node,
              "left_edges": (node[0],),
              "right_edges": (node[1],),
              "compress_params": compress_params}
    U, S, V_dagger = sbc._svd.split_node_full_svd(**kwargs)
        
    nodes_to_contract = (V_dagger, X_inv, Gamma, Y_T_inv, U)
    network_struct = [(-1, 1), (1, 3), (3, -2, 4), (4, 2), (2, -3)]
    Gamma = tn.ncon(nodes_to_contract, network_struct)

    mps_nodes[0] = Gamma
    mps_nodes[1] = S

    return None



def _right_dominant_eigvec_of_R(Gamma, S, V_R_0):
    # The dominant eigenvector is calculated using the power iteration method.
    # Note that 'R' here takes on the definition given in PRB 78, 155117 (2008).
    M = tn.ncon((Gamma, S), ((-1, -2, 1), (1, -3)))
    conj_M = tn.conj(M)

    if True:
        MM = tn.ncon((M, conj_M), ((-1, 1, -3), (-2, 1, -4)))
        tn.flatten_edges([MM[0], MM[1]])
        tn.flatten_edges([MM[0], MM[1]])
        mat = np.array(MM.tensor)
        if M.shape[0] > 10:
            W, V = scipy.sparse.linalg.eigs(mat, k=1)
            dominant_eigval_idx = 0
        else:
            W, V = scipy.linalg.eig(mat)
            dominant_eigval_idx = np.argmax(np.abs(W))
        V_R = tn.Node(V[:, dominant_eigval_idx].reshape(M.shape[0], M.shape[0]))
        print("mu from R =", W[dominant_eigval_idx])
        return V_R

    if (V_R_0 is None) or (V_R_0.shape[0] != S.shape[0]):
        b = _random_starting_node_for_power_iteration(S)
    else:
        b = V_R_0 / tn.norm(V_R_0)

    last_mu = 0
    mu = np.inf
    epsilon_D = 1.0e-14

    while (np.abs(mu-last_mu) > np.abs(mu)*epsilon_D) or (mu == np.inf):
        Mb = tn.ncon((M, b), ((-1, -2, 1), (1, -3)))

        Mb[1] ^ conj_M[1]
        Mb[2] ^ conj_M[2]
        MMb = tn.contract_between(node1=Mb,
                                  node2=conj_M,
                                  output_edge_order=(Mb[0], conj_M[0]))

        conj_last_b = tn.conj(b)
        b = MMb / tn.norm(MMb)

        conj_last_b[0] ^ MMb[0]
        conj_last_b[1] ^ MMb[1]
        bMMb = tn.contract_between(node1=conj_last_b, node2=MMb)

        norm_of_conj_last_b = tn.norm(conj_last_b)
        last_mu = mu
        mu = bMMb / norm_of_conj_last_b / norm_of_conj_last_b
        mu = complex(np.array(mu.tensor))

    print("mu from R =", mu)
    result = b

    return result



def _left_dominant_eigvec_of_L(Gamma, S, V_L_0):
    # The dominant eigenvector is calculated using the power iteration method.
    # Note that 'L' here takes on the definition given in PRB 78, 155117 (2008).
    M = tn.ncon((S, Gamma), ((-1, 1), (1, -2, -3)))
    conj_M = tn.conj(M)

    if True:
        MM = tn.ncon((M, conj_M), ((-1, 1, -3), (-2, 1, -4)))
        tn.flatten_edges([MM[0], MM[1]])
        tn.flatten_edges([MM[0], MM[1]])
        mat = np.transpose(np.array(MM.tensor))
        if M.shape[0] > 10:
            W, V = scipy.sparse.linalg.eigs(mat, k=1)
            dominant_eigval_idx = 0
        else:
            W, V = scipy.linalg.eig(mat)
            dominant_eigval_idx = np.argmax(np.abs(W))
        V_L = tn.Node(V[:, dominant_eigval_idx].reshape(M.shape[0], M.shape[0]))
        print("mu from L =", W[dominant_eigval_idx])
        return V_L

    if (V_L_0 is None) or (V_L_0.shape[0] != S.shape[0]):
        conj_b = _random_starting_node_for_power_iteration(S)
    else:
        conj_b = V_L_0 / tn.norm(V_L_0)

    last_mu = 0
    mu = np.inf
    epsilon_D = 1.0e-14

    while (np.abs(mu-last_mu) > np.abs(mu)*epsilon_D) or (mu == np.inf):
        bM = tn.ncon((conj_b, M), ((-1, 1), (1, -2, -3)))

        bM[0] ^ conj_M[0]
        bM[1] ^ conj_M[1]
        bMM = tn.contract_between(node1=bM,
                                  node2=conj_M,
                                  output_edge_order=(bM[2], conj_M[2]))

        last_b = tn.conj(conj_b)
        conj_b = bMM / tn.norm(bMM)

        bMM[0] ^ last_b[0]
        bMM[1] ^ last_b[1]
        bMMb = tn.contract_between(node1=bMM, node2=last_b)

        norm_of_last_b = tn.norm(last_b)
        last_mu = mu
        mu = bMMb / norm_of_last_b / norm_of_last_b
        mu = complex(np.array(mu.tensor))

    print("mu from L =", mu)
    result = conj_b

    return result



def _random_starting_node_for_power_iteration(S):
    chi_s = S.shape[0]  # Spatial bond dimension.
    b = tn.Node(np.random.rand(chi_s, chi_s)
                +1j*np.random.rand(chi_s, chi_s))
    b /= tn.norm(b)

    return b



_orthogonalize_and_compress_infinite_mps_err_msg_1 = \
    ("To orthogonalize infinite MPS's, the algorithm requires inverting "
     "certain matrices. In the current situation, one of these matrices was "
     "singular resulting in a failed simulation.")
