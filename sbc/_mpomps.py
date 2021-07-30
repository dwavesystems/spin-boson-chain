#!/usr/bin/env python
r"""Contains an implementation for applying MPO's to MPS's. It also contains an
implementation for applying two-site gates (represented by MPO nodes) to MPS's.
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



# For performing SVD truncation sweeps, shifting orthogonal centers,
# and single-node SVD.
import sbc._svd

# For performing QR sweeps, and single-node QR factorizations.
import sbc._qr

# For finding the dominant eigenvectors of specially structured tensor networks.
import sbc._arnoldi



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
    kwargs = {"mpo_nodes": mpo_nodes,
              "mps_nodes": mps_nodes,
              "compress_params": compress_params}
    if is_infinite:
        apply_infinite_mpo_to_infinite_mps_and_compress(**kwargs)
    else:
        apply_finite_mpo_to_finite_mps_and_compress(**kwargs)

    return None



def apply_infinite_mpo_to_infinite_mps_and_compress(mpo_nodes,
                                                    mps_nodes,
                                                    compress_params):
    apply_directly_mpo_to_mps(mpo_nodes, mps_nodes)

    L = len(mps_nodes)
    Lambda_Theta = sbc._svd.Lambda_Theta_form(mps_nodes)
    canonicalize_and_compress_infinite_mps(Lambda_Theta, compress_params)

    Lambda = Lambda_Theta[0]
    Theta_nodes = Lambda_Theta[1]

    L = len(mps_nodes)

    nodes_to_contract = [Lambda, Theta_nodes[0]]
    network_struct = [(-1, 1), (1, -2, -3)]
    mps_nodes[0] = tn.ncon(nodes_to_contract, network_struct)
        
    for r in range(1, L):
        mps_nodes[r] = Theta_nodes[r]

    return None



def apply_finite_mpo_to_finite_mps_and_compress(mpo_nodes,
                                                mps_nodes,
                                                compress_params):
    # Put MPO in left-canonical form. MPS is assumed to be already in said form.
    kwargs = {"nodes": mpo_nodes, "is_infinite": False, "normalize": False}
    sbc._qr.left_to_right_sweep(**kwargs)

    if compress_params.max_num_var_sweeps > 0:
        norm_of_mps_to_compress = \
            norm_of_mpo_mps_network_wo_compression(mpo_nodes, mps_nodes)
        initial_mps_nodes = copy.deepcopy(mps_nodes)
    else:
        initial_mps_nodes = None

    if compress_params.method == "direct":
        apply_directly_mpo_to_mps(mpo_nodes, mps_nodes)
        kwargs["nodes"] = mps_nodes
        sbc._qr.right_to_left_sweep(**kwargs)
    else:
        zip_up(mpo_nodes, mps_nodes, compress_params)

    # Perform a return SVD truncation sweep. If the SVD truncation sweep is
    # followed by variational compression, then L-nodes are calculated and
    # cached while performing the SVD truncation sweep for efficiency. 
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
    nodes_to_contract = [M, R]
    if num_R_legs == 3:
        network_struct = [(-1, -2, 1), (1, -3, -4)]
    else:
        network_struct = [(-1, -2, 1), (1, -3, -4, -5)]
    MR = tn.ncon(nodes_to_contract, network_struct)

    return MR



def contract_MWR_network(M, W, R, num_R_legs=3):
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
    nodes_to_contract = [L, M]
    network_struct = [(1, -3, -4), (1, -2, -1)]
    LM = tn.ncon(nodes_to_contract, network_struct)

    return LM



def contract_LMW_network(L, M, W):
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
        U, S, V_dagger = sbc._svd.split_node_full(temp_node_2,
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
    imin = 0
    imax = len(mps_nodes) - 2
    L = trivial_L(mpo_nodes, initial_mps_nodes)
    L_cache = [L]

    # Normalize MPS now if no variational compression is to be performed.
    # Otherwise, normalization is performed after variational compression.
    normalize_schmidt_spectra = (compress_params.max_num_var_sweeps == 0)
    
    for i in range(imin, imax+1):
        kwargs = {"nodes": mps_nodes,
                  "current_orthogonal_center_idx": i,
                  "compress_params": compress_params,
                  "is_infinite": False,
                  "normalize_schmidt_spectra": normalize_schmidt_spectra}
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

    if compress_params.max_num_var_sweeps == 0:
        mps_nodes[-1] /= tn.norm(mps_nodes[-1])

    return L_cache



def variational_compression(mpo_nodes,
                            mps_nodes,
                            initial_mps_nodes,
                            norm_of_mps_to_compress,
                            L_cache,
                            compress_params):
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

    # Normalize MPS.
    mps_nodes[-1] /= tn.norm(mps_nodes[-1])

    return None



def variational_compression_has_converged(mpo_nodes,
                                          mps_nodes,
                                          initial_mps_nodes,
                                          norm_of_mps_to_compress,
                                          L_cache,
                                          R_cache,
                                          compress_params):
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
    left_node, right_node = sbc._qr.split_node(node=QR,
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
    left_node, right_node = sbc._qr.split_node(node=QR,
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
    for idx, (mpo_node, mps_node) in enumerate(zip(mpo_nodes, mps_nodes)):
        mps_nodes[idx] = apply_directly_mpo_node_to_mps_node(mpo_node, mps_node)

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



def canonicalize_and_compress_infinite_mps(Lambda_Theta, compress_params):
    V_L = calc_V_L(Lambda_Theta)
    V_R = calc_V_R(Lambda_Theta)
    X, X_inv = calc_X_and_X_inv(V_L)
    Y, Y_inv = calc_Y_and_Y_inv(V_R)

    old_Lambda = Lambda_Theta[0]
    old_Theta_nodes = Lambda_Theta[1]

    X_old_Lambda_Y = tn.ncon([X, old_Lambda, Y], [(-1, 1), (1, 2), (2, -2)])
    U, S, V_dagger = sbc._svd.split_node_full(node=X_old_Lambda_Y,
                                              left_edges=(X_old_Lambda_Y[0],),
                                              right_edges=(X_old_Lambda_Y[1],),
                                              compress_params=compress_params)

    new_Lambda = S / tn.norm(S)

    L = len(old_Theta_nodes)

    if L == 1:
        Z_L = tn.ncon([V_dagger, Y_inv], [(-1, 1), (1, -2)])
        Z_R = tn.ncon([X_inv, U], [(-1, 1), (1, -2)])
    else:
        new_Lambda_inv_tensor = np.diag(1 / np.diag(np.array(new_Lambda.tensor)))
        new_Lambda_inv = tn.Node(new_Lambda_inv_tensor)
        
        conj_U = tn.conj(U)
        V_T = tn.conj(V_dagger)

        nodes_to_contract = [new_Lambda_inv, conj_U, X, old_Lambda]
        network_struct = [(-1, 3), (1, 3), (1, 2), (2, -2)]
        Z_L = tn.ncon(nodes_to_contract, network_struct)

        nodes_to_contract = [old_Lambda, Y, V_T, new_Lambda_inv]
        network_struct = [(-1, 1), (1, 2), (3, 2), (3, -2)]
        Z_R = tn.ncon(nodes_to_contract, network_struct)

    new_Theta_nodes = old_Theta_nodes

    nodes_to_contract = [Z_L, new_Theta_nodes[0]]
    network_struct = [(-1, 1), (1, -2, -3)]
    new_Theta_nodes[0] = tn.ncon(nodes_to_contract, network_struct)

    nodes_to_contract = [new_Theta_nodes[-1], Z_R]
    network_struct = [(-1, -2, 1), (1, -3)]
    new_Theta_nodes[-1] = tn.ncon(nodes_to_contract, network_struct)

    compress_and_normalize_Theta(new_Lambda, new_Theta_nodes, compress_params)

    Lambda_Theta[0] = new_Lambda
    Lambda_Theta[1] = new_Theta_nodes

    return None



def compress_and_normalize_Theta(Lambda, Theta_nodes, compress_params):
    kwargs = {"nodes": Theta_nodes, "is_infinite": False, "normalize": False}
    sbc._qr.left_to_right_sweep(**kwargs)
    kwargs["compress_params"] = compress_params
    kwargs["normalize"] = True
    sbc._svd.right_to_left_sweep(**kwargs)

    M = tn.ncon([Lambda, Theta_nodes[0]], [(-1, 1), (1, -2, -3)])
    Theta_nodes[0] /= tn.norm(M)

    return None



def calc_V_L(Lambda_Theta):
    kwargs = {"Lambda_Theta": Lambda_Theta,
              "unit_cell_type": "left",
              "krylov_dim": 10,
              "epsilon_D": 1.0e-14}

    mu_L, V_L = sbc._arnoldi.dominant_eigpair_of_transfer_matrix(**kwargs)

    return V_L



def calc_V_R(Lambda_Theta):
    kwargs = {"Lambda_Theta": Lambda_Theta,
              "unit_cell_type": "right",
              "krylov_dim": 10,
              "epsilon_D": 1.0e-14}

    mu_R, V_R = sbc._arnoldi.dominant_eigpair_of_transfer_matrix(**kwargs)

    return V_R



def calc_X_and_X_inv(V_L):
    if V_L.backend.name != "numpy":
        sbc._backend.tf_to_np(V_L)

    D, W = scipy.linalg.eigh(V_L.tensor, driver='ev')
    D = np.abs(D)
    tol = 1.0e-14
    D[D<tol] = tol
    
    X_dagger = W @ np.diag(np.sqrt(D))
    X = np.conj(np.transpose(X_dagger))
    X = tn.Node(X)

    # Note that X_inv is really a pseudo-inverse of X.
    X_inv = W @ np.diag(1/np.sqrt(D))
    X_inv = tn.Node(X_inv)

    return X, X_inv



def calc_Y_and_Y_inv(V_R):
    if V_R.backend.name != "numpy":
        sbc._backend.tf_to_np(V_R)

    D, W = scipy.linalg.eigh(V_R.tensor, driver='ev')
    D = np.abs(D)
    tol = 1.0e-14
    D[D<tol] = tol
    
    Y = W @ np.diag(np.sqrt(D))
    Y = tn.Node(Y)  # Convert from numpy array to tensornetwork node.

    # Note that Y_inv is really a pseudo-inverse of Y.
    W_dagger = np.conj(np.transpose(W))
    Y_inv = np.diag(1/np.sqrt(D)) @ W_dagger
    Y_inv = tn.Node(Y_inv)

    return Y, Y_inv
