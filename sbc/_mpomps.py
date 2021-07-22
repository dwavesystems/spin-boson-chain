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

# For performing QR factorizations.
import sbc._qr

# For finding the dominant eigenvectors of specially structured tensor networks.
import sbc._arnoldi

# For debugging.
import sbc._testing_arnoldi



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
        # In this case ``mps_nodes=(Gammas, Lambdas)``, where ``Gammas`` and
        # ``Lambdas`` contain the Gamma and Lambda nodes of the MPS
        # respectively.
        apply_infinite_mpo_to_infinite_mps_and_compress(mpo_nodes,
                                                        mps_nodes,
                                                        compress_params)
    else:
        # In this case ``mps_nodes[i]`` is the MPS node at site ``i``.
        apply_finite_mpo_to_finite_mps_and_compress(mpo_nodes,
                                                    mps_nodes,
                                                    compress_params)
        
    return None



def apply_gates_to_mps_and_compress(gates,
                                    mps_nodes,
                                    Gammas,
                                    Lambdas,
                                    compress_params):
# def apply_gates_to_mps_and_compress(gates,
#                                     mps_nodes,
#                                     compress_params):
    # Used only for infinite chains. 
    L = len(mps_nodes)
    gate_idx = 0
    for rmin in range(0, 2):
        for r in range(rmin, L, 2):
            gate = gates[gate_idx]
            apply_directly_gate_to_mps(r, gate, Gammas, Lambdas)
            # apply_directly_gate_to_mps(r, gate, mps_nodes)

            kwargs = {"r": r+1,
                      "Gammas": Gammas,
                      "Lambdas": Lambdas,
                      "compress_params": compress_params}
            # truncate_bond_of_infinite_mps(**kwargs)

            # if True:
            #     kwargs = {"nodes": mps_nodes,
            #               "compress_params": None,
            #               "is_infinite": True,
            #               "starting_node_idx": r}
            #     schmidt_spectra = sbc._svd.left_to_right_sweep(**kwargs)
            #     Lambda_rP1 = schmidt_spectra[0]

            #     print("i =", r+1)
            #     # kwargs = {"i": r+1,
            #     #           "mps_nodes": mps_nodes,
            #     #           "Lambda_i": Lambda_rP1,
            #     #           "compress_params": compress_params}
            #     # truncate_bond_of_infinite_mps(**kwargs)

            #     kwargs = {"i": r+1,
            #               "mps_nodes": mps_nodes,
            #               "Lambda_i": Lambda_rP1,
            #               "compress_params": compress_params}
            #     if rmin == 0:
            #         kwargs["compress_params"] = None
            #     truncate_bond_of_infinite_mps(**kwargs)
            
            gate_idx += 1
        
    return None



# def apply_infinite_mpo_to_infinite_mps_and_compress(mpo_nodes,
#                                                     mps_nodes,
#                                                     Gammas,
#                                                     Lambdas,
#                                                     compress_params):
def apply_infinite_mpo_to_infinite_mps_and_compress(mpo_nodes,
                                                    mps_nodes,
                                                    # Lambda_Theta,
                                                    compress_params):
    # print("before")
    # for r in range(len(mps_nodes)):
        # print("mpo_nodes[{}].shape =".format(r), mpo_nodes[r].shape,
        #       "mps_nodes[{}].shape =".format(r), mps_nodes[r].shape)
    apply_directly_finite_mpo_to_finite_mps(mpo_nodes, mps_nodes)
    # print("compress_params:", compress_params.__dict__)
    kwargs = {"nodes": mps_nodes,
              "compress_params": None,
              "is_infinite": True,
              "starting_node_idx": None}
    sbc._svd.left_to_right_sweep(**kwargs)
    kwargs["compress_params"] = compress_params
    sbc._svd.right_to_left_sweep(**kwargs)
    # print("after")
    # for r in range(len(mps_nodes)):
        # print("mpo_nodes[{}].shape =".format(r), mpo_nodes[r].shape,
        #       "mps_nodes[{}].shape =".format(r), mps_nodes[r].shape)

    return None

    L = len(mps_nodes)
    for starting_node_idx in range(1):
        Lambda_Theta = sbc._svd.Lambda_Theta_form(mps_nodes, starting_node_idx)
        canonicalize_and_compress_infinite_mps(Lambda_Theta, compress_params)

        Lambda = Lambda_Theta[0]
        Theta_nodes = Lambda_Theta[1]

        # # print("Lambda =", np.diag(Lambda.tensor))
        # print("norm of Lambda =", tn.norm(Lambda))

        L = len(mps_nodes)
        r0 = starting_node_idx

        nodes_to_contract = [Lambda, Theta_nodes[0]]
        network_struct = [(-1, 1), (1, -2, -3)]
        mps_nodes[r0] = tn.ncon(nodes_to_contract, network_struct)
        
        for dr in range(1, L):
            mps_nodes[(r0+dr)%L] = Theta_nodes[dr]

    kwargs = {"nodes": mps_nodes,
              "compress_params": None,
              "is_infinite": True,
              "starting_node_idx": None}
    sbc._svd.left_to_right_sweep(**kwargs)
    sbc._svd.right_to_left_sweep(**kwargs)

    return None
    
    # new_Lambda, new_Theta_nodes = sbc._svd.Lambda_Theta_form(mps_nodes)
    # Lambda_Theta[0] = new_Lambda
    # Lambda_Theta[1] = new_Theta_nodes
    
    # apply_directly_mpo_to_mps(mpo_nodes, mps_nodes)
    # apply_directly_infinite_mpo_to_infinite_mps(mpo_nodes, Gammas, Lambdas)
    # apply_directly_infinite_mpo_to_infinite_mps(mpo_nodes, Lambda_Theta)
    # canonicalize_and_compress_infinite_mps(Gammas, Lambdas, compress_params)
    # canonicalize_and_compress_infinite_mps(Lambda_Theta, compress_params)

    Lambda = Lambda_Theta[0]
    Theta_nodes = Lambda_Theta[1]

    L = len(mps_nodes)
    mps_nodes[0] = tn.ncon([Lambda, Theta_nodes[0]], [(-1, 1), (1, -2, -3)])
    for r in range(1, L):
        mps_nodes[r] = Theta_nodes[r]

    # print("Lambda shape =", Lambda.shape)
    print("Lambda =", np.diag(Lambda.tensor))
    # for idx, Theta_node in enumerate(Theta_nodes):
    #     print("Theta node shape #{} =".format(idx), Theta_node.shape)

    # L = len(Gammas)
    # for r in range(L):
    #     print("Lambdas[{}] =".format(r), Lambdas[r].tensor)
    #     print("Gammas[{}] =".format(r), Gammas[r].tensor)
    #     mps_nodes[r] = tn.ncon([Lambdas[r], Gammas[r]], [(-1, 1), (1, -2, -3)])

    return None



def apply_finite_mpo_to_finite_mps_and_compress(mpo_nodes,
                                                mps_nodes,
                                                compress_params):
    # Put MPO in left-canonical form. MPS is assumed to be already in said form.
    sbc._svd.left_to_right_sweep(mpo_nodes,
                                 compress_params=None,
                                 is_infinite=False,
                                 starting_node_idx=None)

    if compress_params.max_num_var_sweeps > 0:
        norm_of_mps_to_compress = \
            norm_of_mpo_mps_network_wo_compression(mpo_nodes, mps_nodes)
        initial_mps_nodes = copy.deepcopy(mps_nodes)
    else:
        initial_mps_nodes = None

    if compress_params.method == "direct":
        apply_directly_finite_mpo_to_finite_mps(mpo_nodes, mps_nodes)
        # apply_directly_mpo_to_mps(mpo_nodes, mps_nodes)
        sbc._svd.right_to_left_sweep(mps_nodes,
                                     compress_params=None,
                                     is_infinite=False,
                                     starting_node_idx=None)
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



def apply_directly_finite_mpo_to_finite_mps(mpo_nodes, mps_nodes):
    for idx, (mpo_node, mps_node) in enumerate(zip(mpo_nodes, mps_nodes)):
        mps_nodes[idx] = apply_directly_mpo_node_to_mps_node(mpo_node, mps_node)

    return None



# def apply_directly_infinite_mpo_to_infinite_mps(mpo_nodes, Gammas, Lambdas):
def apply_directly_infinite_mpo_to_infinite_mps(mpo_nodes, Lambda_Theta):
    Lambda = Lambda_Theta[0]
    Theta_nodes = Lambda_Theta[1]
    
    # Treat ``Theta_nodes`` as a finite MPS.
    apply_directly_finite_mpo_to_finite_mps(mpo_nodes, Theta_nodes)

    old_Lambda_tensor = Lambda.tensor
    d = mpo_nodes[0].shape[0]
    chi = old_Lambda_tensor.shape[0]
    new_Lambda_tensor = np.zeros([chi, d, chi, d], dtype=np.complex)
        
    for m in range(chi):
        for w in range(d):
            new_Lambda_tensor[m, w, m, w] = old_Lambda_tensor[m, m]
    new_Lambda = tn.Node(new_Lambda_tensor)
        
    tn.flatten_edges([new_Lambda[0], new_Lambda[1]])
    tn.flatten_edges([new_Lambda[0], new_Lambda[1]])

    Lambda_Theta[0] = new_Lambda
    Lambda_Theta[1] = Theta_nodes
    
    # zip_obj = zip(mpo_nodes, Gammas, Lambdas)
    # for idx, (mpo_node, Gamma, Lambda) in enumerate(zip_obj):
    #     Gammas[idx] = apply_directly_mpo_node_to_mps_node(mpo_node, Gamma)
        
    #     d = mpo_node.shape[0]
    #     chi = Lambda.shape[0]
        
    #     old_Lambda_tensor = Lambda.tensor
    #     new_Lambda_tensor = np.zeros([chi, d, chi, d], dtype=np.complex)
        
    #     for m in range(chi):
    #         for w in range(d):
    #             new_Lambda_tensor[m, w, m, w] = old_Lambda_tensor[m, m]
    #     new_Lambda = tn.Node(new_Lambda_tensor)
        
    #     tn.flatten_edges([new_Lambda[0], new_Lambda[1]])
    #     tn.flatten_edges([new_Lambda[0], new_Lambda[1]])

    #     Lambdas[idx] = new_Lambda

    return None



# def apply_directly_infinite_mpo_to_infinite_mps(mpo_nodes, mps_nodes):
#     Gammas = mps_nodes[0]
#     Lambdas = mps_nodes[1]
#     L = len(Gammas)
#     imax = L - 1
    
#     for i in range(imax+1):
#         Gammas[i] = apply_directly_mpo_node_to_mps_node(mpo_nodes[i], Gammas[i])
#         chi_MPO = mpo_nodes[i].shape[0]  # MPO bond dimension at bond i.
#         chi_MPS = Lambdas[i].shape[0]  # MPS bond dimension at bond i.
    
#         Lambda_tensor = (Lambdas[i].tensor.numpy()
#                          if Lambdas[i].backend.name != "numpy"
#                          else Lambdas[i].tensor)
#         new_Lambda_tensor = np.zeros([chi_MPO, chi_MPS, chi_MPO, chi_MPS],
#                                      dtype=np.complex128)
    
#         for w in range(chi_MPO):
#             for m in range(chi_MPS):
#                 new_Lambda_tensor[w, m, w, m] = Lambda_tensor[m, m]

#         Lambda = tn.Node(new_Lambda_tensor)
            
#         tn.flatten_edges([Lambda[0], Lambda[1]])
#         tn.flatten_edges([Lambda[0], Lambda[1]])

#         Lambdas[i] = Lambda

#     mps_nodes[0] = Gammas
#     mps_nodes[1] = Lambdas

#     return None



# def apply_directly_finite_mpo_to_finite_mps(mpo_nodes, mps_nodes):
#     num_mps_nodes = len(mps_nodes)
#     new_mps_nodes = []
#     for idx, (mpo_node, mps_node) in enumerate(zip(mpo_nodes, mps_nodes)):
#         new_mps_node = apply_directly_mpo_node_to_mps_node(mpo_node, mps_node)
#         mps_nodes[idx] = new_mps_node

#     return None



def apply_directly_gate_to_mps(r, gate, Gammas, Lambdas):
# def apply_directly_gate_to_mps(r, gate, mps_nodes):
    # Used only for infinite chains.
    # L = len(mps_nodes)
    L = len(Gammas)

    mpo_node = gate[0]
    nodes_to_contract = [Lambdas[r], Gammas[r]]
    network_struct = [(-1, 1), (1, -2, -3)]
    mps_node = tn.ncon(nodes_to_contract, network_struct)
    node_1 = apply_directly_mpo_node_to_mps_node(mpo_node, mps_node)

    mpo_node = gate[1]
    nodes_to_contract = [Gammas[(r+1)%L], Lambdas[(r+2)%L]]
    network_struct = [(-1, -2, 1), (1, -3)]
    mps_node = tn.ncon(nodes_to_contract, network_struct)
    node_3 = apply_directly_mpo_node_to_mps_node(mpo_node, mps_node)

    d = gate[0].shape[-1]
    Lambda_rP1 = Lambdas[(r+1)%L]
    chi = Lambda_rP1.shape[0]
    Lambda_rP1_tensor = np.array(Lambda_rP1.tensor)
    tensor = np.zeros([chi, d, chi, d], dtype=np.complex)
    for m in range(chi):
        for w in range(d):
            tensor[m, w, m, w] = Lambda_rP1_tensor[m, m]
    node_2 = tn.Node(tensor)
    tn.flatten_edges([node_2[0], node_2[1]])
    tn.flatten_edges([node_2[0], node_2[1]])

    kwargs = {"node": node_1,
              "left_edges": (node_1[0], node_1[1]),
              "right_edges": (node_1[2],),
              "compress_params": None}
    U_1, S_1, V_1_dagger = sbc._svd.split_node_full(**kwargs)

    kwargs = {"node": node_3,
              "left_edges": (node_3[0],),
              "right_edges": (node_3[1], node_3[2]),
              "compress_params": None}
    U_3, S_3, V_3_dagger = sbc._svd.split_node_full(**kwargs)

    nodes_to_contract = [S_1, V_1_dagger, node_2, U_3, S_3]
    network_struct = [(-1, 1), (1, 2), (2, 3), (3, 4), (4, -2)]
    node_2 = tn.ncon(nodes_to_contract, network_struct)

    kwargs = {"node": node_2,
              "left_edges": (node_2[0],),
              "right_edges": (node_2[1],),
              "compress_params": None}
    U_2, S_2, V_2_dagger = sbc._svd.split_node_full(**kwargs)

    Lambda_r_tensor = np.array(Lambdas[r].tensor)
    print("Lamdas[r] =", Lambda_r_tensor)
    Lambda_r_inv_tensor = np.diag(1 / np.diag(Lambda_r_tensor))
    Lambda_r_inv = tn.Node(Lambda_r_inv_tensor)

    Lambda_rP2_tensor = np.array(Lambdas[(r+2)%L].tensor)
    print("Lamdas[r+2] =", Lambda_rP2_tensor)
    Lambda_rP2_inv_tensor = np.diag(1 / np.diag(Lambda_rP2_tensor))
    Lambda_rP2_inv = tn.Node(Lambda_rP2_inv_tensor)

    nodes_to_contract = [Lambda_r_inv, U_1, U_2]
    network_struct = [(-1, 1), (1, -2, 2), (2, -3)]
    Gammas[r] = tn.ncon(nodes_to_contract, network_struct)

    Lambdas[(r+1)%L] = S_2
    print("Lamdas[(r+1)%L] =", S_2.tensor)

    nodes_to_contract = [V_2_dagger, V_3_dagger, Lambda_rP2_inv]
    network_struct = [(-1, 1), (1, -2, 2), (2, -3)]
    Gammas[(r+1)%L] = tn.ncon(nodes_to_contract, network_struct)
    
    # mpo_node = gate[0]
    # print("left gate node shape =", mpo_node.shape)
    # mps_node = mps_nodes[r]
    # new_mps_node = apply_directly_mpo_node_to_mps_node(mpo_node, mps_node)
    # mps_nodes[r] = new_mps_node

    # mpo_node = gate[1]
    # print("right gate node shape =", mpo_node.shape)
    # mps_node = mps_nodes[(r+1)%L]
    # new_mps_node = apply_directly_mpo_node_to_mps_node(mpo_node, mps_node)
    # mps_nodes[(r+1)%L] = new_mps_node    

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
    mu_L, V_L = calc_mu_L_and_V_L(Lambda_Theta)
    mu_R, V_R = calc_mu_R_and_V_R(Lambda_Theta)
    abs_mu = 0.5 * abs(mu_L + mu_R)
    X = calc_X(V_L)
    Y = calc_Y(V_R)

    old_Lambda = Lambda_Theta[0]
    old_Theta_nodes = Lambda_Theta[1]

    X_old_Lambda_Y = tn.ncon([X, old_Lambda, Y], [(-1, 1), (1, 2), (2, -2)])
    U, S, V_dagger = sbc._svd.split_node_full(node=X_old_Lambda_Y,
                                              left_edges=(X_old_Lambda_Y[0],),
                                              right_edges=(X_old_Lambda_Y[1],),
                                              compress_params=compress_params)

    new_Lambda = S / tn.norm(S)
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

    for idx, Theta_node in enumerate(new_Theta_nodes):
        print("old Theta node shape #{} =".format(idx), Theta_node.shape)
    
    nodes_to_contract = [Z_L, new_Theta_nodes[0]]
    network_struct = [(-1, 1), (1, -2, -3)]
    new_Theta_nodes[0] = tn.ncon(nodes_to_contract, network_struct)
    new_Theta_nodes[0] /= (np.sqrt(abs_mu) * tn.norm(S))

    nodes_to_contract = [new_Theta_nodes[-1], Z_R]
    network_struct = [(-1, -2, 1), (1, -3)]
    new_Theta_nodes[-1] = tn.ncon(nodes_to_contract, network_struct)

    kwargs = {"nodes": new_Theta_nodes,
              "compress_params": None,
              "is_infinite": False,
              "starting_node_idx": None}
    sbc._svd.left_to_right_sweep(**kwargs)
    kwargs["compress_params"] = compress_params
    # for idx, new_Theta_node in enumerate(new_Theta_nodes):
    #     print("new Theta node max #{}:".format(idx), np.amax(np.abs(new_Theta_node.tensor)))
    sbc._svd.right_to_left_sweep(**kwargs)

    Lambda_Theta[0] = new_Lambda
    Lambda_Theta[1] = new_Theta_nodes

    # mu_L, V_L = calc_mu_L_and_V_L(Lambda_Theta)
    # mu_R, V_R = calc_mu_R_and_V_R(Lambda_Theta)
    # print("norm of S =", tn.norm(S))
    # print("abs_mu =", 0.5 * abs(mu_L + mu_R))

    # for idx, Theta_node in enumerate(new_Theta_nodes):
    #     print("new Theta node shape #{} =".format(idx), Theta_node.shape)

    return None



# def canonicalize_and_compress_infinite_mps(Gammas, Lambdas, compress_params):
#     L = len(Gammas)

#     V_L = calc_V_L(Gammas, Lambdas)
#     V_R = calc_V_R(Gammas, Lambdas)
#     X = calc_X(V_L)
#     Y = calc_Y(V_R)

#     Lambda_0 = tn.ncon([X, Lambdas[0], Y], [(-1, 1), (1, 2), (2, -2)])
#     U, S, V_dagger = sbc._svd.split_node_full(node=Lambda_0,
#                                               left_edges=(Lambda_0[0],),
#                                               right_edges=(Lambda_0[1],),
#                                               compress_params=compress_params)

#     conj_U = tn.conj(U)
#     V_T = tn.conj(V_dagger)

#     temp_mps_nodes = []

#     nodes_to_contract = [conj_U, X, Lambdas[0], Gammas[0]]
#     network_struct = [(1, -1), (1, 2), (2, 3), (3, -2, -3)]
#     temp_mps_node = tn.ncon(nodes_to_contract, network_struct)
#     temp_mps_nodes.append(temp_mps_node)

#     for r in range(1, L-1):
#         nodes_to_contract = [Lambdas[r], Gammas[r]]
#         network_struct = [(-1, 1), (1, -2, -3)]
#         temp_mps_node = tn.ncon(nodes_to_contract, network_struct)
#         temp_mps_nodes.append(temp_mps_node)

#     nodes_to_contract = [Lambdas[L-1], Gammas[L-1], Lambdas[0], Y, V_T]
#     network_struct = [(-1, 1), (1, -2, 2), (2, 3), (3, 4), (-3, 4)]
#     temp_mps_node = tn.ncon(nodes_to_contract, network_struct)
#     temp_mps_nodes.append(temp_mps_node)

#     kwargs = {"mps_nodes": temp_mps_nodes,
#               "compress_params": compress_params,
#               "is_infinite": False}
#     temp_Gammas, temp_Lambdas = sbc._svd.vidal_form(**kwargs)

#     Lambda_0_inv_tensor = np.diag(1 / np.diag(np.array(Lambda_0.tensor)))
#     Lambda_0_inv = tn.Node(Lambda_0_inv_tensor)

#     Lambdas[0] = Lambda_0
#     nodes_to_contract = [Lambda_0_inv, temp_Gammas[0]]
#     network_struct = [(-1, 1), (1, -2, -3)]
#     Gammas[0] = tn.ncon(nodes_to_contract, network_struct)

#     for r in range(1, L-1):
#         Lambdas[r] = temp_Lambdas[r-1]
#         Gammas[r] = temp_Gammas[r]

#     Lambdas[L-1] = temp_Lambdas[L-2]
#     nodes_to_contract = [temp_Gammas[L-1], Lambda_0_inv]
#     network_struct = [(-1, -2, 1), (1, -3)]
#     Gammas[L-1] = tn.ncon(nodes_to_contract, network_struct)

#     return None



# def canonicalize_and_compress_infinite_mps(mps_nodes, compress_params):
#     # The procedure below is described in detail in Annals. Phys. 326 (2011),
#     # pages 186-188.
#     # Gammas = mps_nodes[0]
#     # Lambdas = mps_nodes[1]
#     # L = len(Gammas)
#     # imax = L - 1
#     # i = 0
#     i = -1
#     kwargs = {"nodes": mps_nodes,
#               "compress_params": None,
#               "is_infinite": True,
#               "starting_node_idx": i}
#     schmidt_spectra = sbc._svd.left_to_right_sweep(**kwargs)
#     Lambda_iP1 = schmidt_spectra[0]
    
#     imin = i + 1
#     imax = len(mps_nodes)
#     # imax = imin

#     for i in range(imin, imax+1):
#         # truncate_bond_of_infinite_mps(idx_of_bond_to_truncate=i,
#         #                               Gammas=Gammas,
#         #                               Lambdas=Lambdas,
#         #                               compress_params=compress_params)
#         print("i =", i)
#         Lambda_i = Lambda_iP1
#         kwargs = {"i": i,
#                   "mps_nodes": mps_nodes,
#                   "Lambda_i": Lambda_i,
#                   "compress_params": compress_params}
#         Lambda_iP1 = truncate_bond_of_infinite_mps(**kwargs)

#     # mps_nodes[0] = Gammas
#     # mps_nodes[1] = Lambdas
    
#     return None



def truncate_bond_of_infinite_mps(r, Gammas, Lambdas, compress_params):
    L = len(Gammas)

    V_L = calc_V_L(r, Gammas, Lambdas)
    X = calc_X(V_L)

    V_R = calc_V_R(r, Gammas, Lambdas)
    Y = calc_Y(V_R)

    nodes_to_contract = [Gammas[(r-1)%L], Lambdas[r], Y]
    network_struct = [(-1, -2, 1), (1, 2), (2, -3)]
    B = tn.ncon(nodes_to_contract, network_struct)

    nodes_to_contract = [X, Lambdas[r], Gammas[r]]
    network_struct = [(-1, 1), (1, 2), (2, -2, -3)]
    A = tn.ncon(nodes_to_contract, network_struct)

    Lambda_r = tn.ncon([X, Lambdas[r], Y], [(-1, 1), (1, 2), (2, -2)])
    U, S, V_dagger = sbc._svd.split_node_full(node=Lambda_r,
                                              left_edges=(Lambda_r[0],),
                                              right_edges=(Lambda_r[1],),
                                              compress_params=compress_params)

    conj_U = tn.conj(U)
    V_T = tn.conj(V_dagger)    

    B = tn.ncon([B, V_T], [(-1, -2, 1), (-3, 1)])
    A = tn.ncon([conj_U, A], [(1, -1), (1, -2, -3)])

    Lambda_r_inv_tensor = np.diag(1 / np.diag(np.array(Lambda_r.tensor)))
    Lambda_r_inv = tn.Node(Lambda_r_inv_tensor)

    Gammas[(r-1)%L] = tn.ncon([B, Lambda_r_inv], [(-1, -2, 1), (1, -3)])
    Lambdas[r] = Lambda_r
    Gammas[r] = tn.ncon([Lambda_r_inv, A], [(-1, 1), (1, -2, -3)])

    return None



# def truncate_bond_of_infinite_mps(i,
#                                   mps_nodes,
#                                   Lambda_i,
#                                   compress_params):
#     L = len(mps_nodes)

#     # kwargs = {"nodes": mps_nodes,
#     #           "compress_params": None,
#     #           "is_infinite": True,
#     #           "starting_node_idx": i-1}
#     # schmidt_spectra = sbc._svd.left_to_right_sweep(**kwargs)
#     # Lambda_i = schmidt_spectra[0]

#     Lambda_i_inv_tensor = np.diag(1 / np.diag(np.array(Lambda_i.tensor)))
#     Lambda_i_inv = tn.Node(Lambda_i_inv_tensor)

#     A_iM1 = mps_nodes[(i-1)%L]
#     # M = tn.ncon([mps_nodes[(i-i)%L], Lambda_i], [(-1, -2, 1), (1, -3)])
#     try:
#         # print("i =", i)
#         # print("A_iM1 shape:", A_iM1.shape)
#         # print("Lambda_i shape:", Lambda_i.shape)
#         print("mps node shapes before:", [node.shape for node in mps_nodes])
#         # print("shapes #1 =", (A_iM1.shape, Lambda_i.shape))
#         # print("A_iM1 edge dim:", A_iM1.edges[-1].dimension)
#         # print("Lambda_i edge dim:", Lambda_i.edges[0].dimension)
#         M = tn.ncon([mps_nodes[(i-1)%L], Lambda_i], [(-1, -2, 1), (1, -3)])
#     except ValueError as err:
#         # print("A_iM1:", A_iM1)
#         # print("Lambda_i:", Lambda_i)
#         raise err

#     V_L = calc_V_L(mps_nodes, M, Lambda_i_inv, i)
#     X = calc_X(V_L)

#     V_R = calc_V_R(mps_nodes, M, Lambda_i_inv, i)
#     Y = calc_Y(V_R)

#     # print("Lambdas[{}] =".format(i), np.diag(Lambda_i.tensor))
#     Lambda_i = tn.ncon([X, Lambda_i, Y], [(-1, 1), (1, 2), (2, -2)])
#     # print("Lambdas'[{}] =".format(i), Lambda_i.tensor)
#     U, S, V_dagger = sbc._svd.split_node_full(node=Lambda_i,
#                                               left_edges=(Lambda_i[0],),
#                                               right_edges=(Lambda_i[1],),
#                                               compress_params=compress_params)
#     Lambda_i = S
#     # Lambdas[i] = S
#     # print("Lambdas''[{}] =".format(i), np.diag(S.tensor))
#     # print()

#     Lambda_i_inv_tensor = np.diag(1 / np.diag(np.array(Lambda_i.tensor)))
#     Lambda_i_inv = tn.Node(Lambda_i_inv_tensor)

#     conj_U = tn.conj(U)
#     V_T = tn.conj(V_dagger)
#     A_i = mps_nodes[i%L]
    
#     mps_nodes[i%L] = tn.ncon([conj_U, X, A_i], [(1, -1), (1, 2), (2, -2, -3)])
#     mps_nodes[(i-1)%L] = tn.ncon([M, Y, V_T, Lambda_i_inv],
#                                  [(-1, -2, 1), (1, 2), (3, 2), (3, -3)])

#     # Treat the unit cell as a finite MPS in this particular case to avoid the
#     # extra shift at the end.
#     # kwargs = {"nodes": mps_nodes,
#     #           "compress_params": None,
#     #           "is_infinite": False,
#     #           "starting_node_idx": i}
#     # schmidt_spectra = sbc._svd.left_to_right_sweep(**kwargs)
#     # Lambda_iP1 = schmidt_spectra[0]

#     # # print("shapes #2 =", (mps_nodes[i%L].shape, Lambda_iP1.shape))
#     print("mps node shapes after:", [node.shape for node in mps_nodes])

#     # return Lambda_iP1
#     return None



# def truncate_bond_of_infinite_mps(idx_of_bond_to_truncate,
#                                   Gammas,
#                                   Lambdas,
#                                   compress_params):
#     i = idx_of_bond_to_truncate
#     L = len(Gammas)
#     A = tn.ncon([Lambdas[i], Gammas[i]], [(-1, 1), (1, -2, -3)])
#     B = tn.ncon([Gammas[(L-1+i)%L], Lambdas[(L+i)%L]], [(-1, -2, 1), (1, -3)])

#     V_L = calc_V_L(A, Gammas, Lambdas, B, idx_of_bond_to_truncate)
#     V_R = calc_V_R(A, Gammas, Lambdas, B, idx_of_bond_to_truncate)

#     X = calc_X(V_L)
#     Y = calc_Y(V_R)

#     Lambda_i = Lambdas[i]
#     print("Lambdas[{}] =".format(i), np.diag(Lambda_i.tensor))
#     Lambda_i = tn.ncon([X, Lambda_i, Y], [(-1, 1), (1, 2), (2, -2)])
#     print("Lambdas'[{}] =".format(i), Lambda_i.tensor)
#     U, S, V_dagger = sbc._svd.split_node_full(node=Lambda_i,
#                                               left_edges=(Lambda_i[0],),
#                                               right_edges=(Lambda_i[1],),
#                                               compress_params=compress_params)
#     Lambdas[i] = S
#     print("Lambdas''[{}] =".format(i), np.diag(Lambdas[i].tensor))
#     print()

#     conj_U = tn.conj(U)
#     V_T = tn.conj(V_dagger)
#     A = tn.ncon([conj_U, X, A], [(1, -1), (1, 2), (2, -2, -3)])
#     B = tn.ncon([B, Y, V_T], [(-1, -2, 2), (2, 1), (-3, 1)])

#     Lambda_i_tensor = Lambdas[i].tensor
#     Lambda_i_inv_tensor = np.diag(1 / np.diag(np.array(Lambda_i_tensor)))
#     Lambda_i_inv = tn.Node(Lambda_i_inv_tensor)
    
#     Gammas[i] = tn.ncon([Lambda_i_inv, A], [(-1, 1), (1, -2, -3)])
#     Gammas[(L-1+i)%L] = tn.ncon([B, Lambda_i_inv], [(-1, -2, 1), (1, -3)])

#     return None



# def calc_V_L(mps_nodes, M, Lambda_i_inv, i):
# def calc_V_L(r, Gammas, Lambdas):
# def calc_V_L(Gammas, Lambdas):
def calc_mu_L_and_V_L(Lambda_Theta):
    # L = len(Gammas)
    
    # last_mu = 0
    # mu = np.inf
    # epsilon_D = 1.0e-14

    # kwargs = {"mps_nodes": mps_nodes,
    #           "M": M,
    #           "Lambda_i_inv": Lambda_i_inv,
    #           "i": i,
    #           "unit_cell_type": "left",
    #           "krylov_dim": 10,
    #           "starting_node": None,
    #           "epsilon_D": 1.0e-14}
    # kwargs = {"r": r,
    #          "Gammas": Gammas,
    # kwargs = {"Gammas": Gammas,
    #           "Lambdas": Lambdas,
    kwargs = {"Lambda_Theta": Lambda_Theta,
              "unit_cell_type": "left",
              "krylov_dim": 10,
              "starting_node": None,
              "epsilon_D": 1.0e-14}

    mu_L, V_L = sbc._arnoldi.dominant_eigpair_of_transfer_matrix(**kwargs)

    print("mu_L =", mu_L)

    return mu_L, V_L



# def calc_V_L(A, Gammas, Lambdas, B, idx_of_bond_to_truncate):
#     chi = A.shape[0]
#     conj_v = tn.Node(np.random.rand(chi, chi) + 1j*np.random.rand(chi, chi))
#     conj_v /= tn.norm(conj_v)

#     i = idx_of_bond_to_truncate
#     Lambda_i_tensor = Lambdas[i].tensor
#     Lambda_i_inv_tensor = np.diag(1 / np.diag(np.array(Lambda_i_tensor)))
#     Lambda_i_inv = tn.Node(Lambda_i_inv_tensor)
    
#     last_mu = 0
#     mu = np.inf
#     epsilon_D = 1.0e-14

#     # print("Start calc_V_L'")
#     # print("mu =", mu)
#     # print("conj_v =", conj_v.tensor)
#     # print("A =", A.tensor)
#     # L = len(Gammas)
#     # kmin = i + 1
#     # kmax = L - 2 + i
#     # for k in range(kmin, kmax+1):
#     #     print("Gammas[{}] =".format(k), Gammas[k].tensor)
#     #     print("Lambdas[{}] =".format(k), Lambdas[k].tensor)
#     # print("Lambdas[{}] =".format((kmax+1)%L), Lambdas[(kmax+1)%L].tensor)
#     # print("B =", B.tensor)
#     # print("Lambda_i_inv =", Lambda_i_inv.tensor)
#     # print()

#     while (np.abs(mu-last_mu) > np.abs(mu)*epsilon_D) or (mu == np.inf):
#         network = tn.ncon([conj_v, A], [(-1, 1), (1, -2, -3)])
#         conj_A = tn.conj(A)
#         network[0] ^ conj_A[0]
#         network[1] ^ conj_A[1]
#         output_edge_order=(conj_A[2], network[2])
#         network = tn.contract_between(node1=network,
#                                       node2=conj_A,
#                                       output_edge_order=output_edge_order)

#         L = len(Gammas)
#         kmin = i + 1
#         kmax = L - 2 + i
#         for k in range(kmin, kmax+1):
#             network = tn.ncon([network, Lambdas[k%L]], [(-1, 1), (1, -2)])
#             network = tn.ncon([Lambdas[k%L], network], [(-1, 1), (1, -2)])
#             network = tn.ncon([network, Gammas[k%L]], [(-1, 1), (1, -2, -3)])
#             conj_Gamma_k = tn.conj(Gammas[k%L])
#             network[0] ^ conj_Gamma_k[0]
#             network[1] ^ conj_Gamma_k[1]
#             output_edge_order = (conj_Gamma_k[2], network[2])
#             network = tn.contract_between(node1=network,
#                                           node2=conj_Gamma_k,
#                                           output_edge_order=output_edge_order)

#         network = tn.ncon([network, Lambdas[(kmax+1)%L]], [(-1, 1), (1, -2)])
#         network = tn.ncon([Lambdas[(kmax+1)%L], network], [(-1, 1), (1, -2)])

#         network = tn.ncon([network, B], [(-1, 1), (1, -2, -3)])
#         conj_B = tn.conj(B)
#         network[0] ^ conj_B[0]
#         network[1] ^ conj_B[1]
#         output_edge_order=(conj_B[2], network[2])
#         network = tn.contract_between(node1=network,
#                                       node2=conj_B,
#                                       output_edge_order=output_edge_order)

#         network = tn.ncon([network, Lambda_i_inv], [(-1, 1), (1, -2)])
#         network = tn.ncon([Lambda_i_inv, network], [(-1, 1), (1, -2)])

#         last_mu = mu
#         last_v = tn.conj(conj_v)
#         norm_of_last_v = tn.norm(last_v)
#         conj_v = network / tn.norm(network)
#         # print("tn.norm(network) =", tn.norm(network))

#         network[0] ^ last_v[0]
#         network[1] ^ last_v[1]
#         mu = (tn.contract_between(node1=network, node2=last_v)
#               / norm_of_last_v / norm_of_last_v)
#         mu = complex(np.array(mu.tensor))

#         # print("norm_of_last_v =", norm_of_last_v)
#         # print("mu =", mu)
#         # print("conj_v =", conj_v.tensor)
#         # print()

#     print("mu_L =", mu)
#     V_L = conj_v

#     return V_L



# def calc_V_R(mps_nodes, M, Lambda_i_inv, i):
# def calc_V_R(r, Gammas, Lambdas):
# def calc_V_R(Gammas, Lambdas):
def calc_mu_R_and_V_R(Lambda_Theta):
    # chi = M.shape[-1]
    # v = tn.Node(np.random.rand(chi, chi) + 1j*np.random.rand(chi, chi))
    # v /= tn.norm(v)

    # L = len(mps_nodes)
    # last_mu = 0
    # mu = np.inf
    # epsilon_D = 1.0e-14

    # if False:
    #     kwargs = {"mps_nodes": mps_nodes,
    #               "M": M,
    #               "Lambda_i_inv": Lambda_i_inv,
    #               "i": i,
    #               "unit_cell_type": "right",
    #               "krylov_dim": 10,
    #               "starting_node": None,
    #               "epsilon_D": 1.0e-14}
    #     mu_R_1, V_R_1 = sbc._testing_arnoldi.dominant_eigpair_of_transfer_matrix_using_scipy(**kwargs)
    #     print("mu_R #1 =", mu_R_1)

    # kwargs = {"mps_nodes": mps_nodes,
    #           "M": M,
    #           "Lambda_i_inv": Lambda_i_inv,
    #           "i": i,
    #           "unit_cell_type": "right",
    #           "krylov_dim": 10,
    #           "starting_node": None,
    #           "epsilon_D": 1.0e-14}
    # kwargs = {"r": r,
    #           "Gammas": Gammas,
    # kwargs = {"Gammas": Gammas,
    #           "Lambdas": Lambdas,
    kwargs = {"Lambda_Theta": Lambda_Theta,
              "unit_cell_type": "right",
              "krylov_dim": 10,
              "starting_node": None,
              "epsilon_D": 1.0e-14}

    mu_R_2, V_R_2 = sbc._arnoldi.dominant_eigpair_of_transfer_matrix(**kwargs)

    V_R = V_R_2
    mu_R = mu_R_2
    
    print("mu_R #2 =", mu_R_2)
    # print("mu_R =", mu)
    # print("V_R.shape =", V_R_1.shape)
    # V_R = v
    # V_R_2 = v
    # print("V_R_1 =", V_R_1.tensor)
    # print("V_R_2 =", V_R_2.tensor)

    # conj_V_R_2 = tn.conj(V_R_2)
    # overlap = float(np.abs(np.array(tn.ncon([conj_V_R_2, V_R_1], [(1, 2), (1, 2)]).tensor)))
    # print("overlap =", overlap)

    return mu_R, V_R



# def calc_V_R(A, Gammas, Lambdas, B, idx_of_bond_to_truncate):
#     chi = B.shape[-1]
#     v = tn.Node(np.random.rand(chi, chi) + 1j*np.random.rand(chi, chi))
#     v /= tn.norm(v)

#     i = idx_of_bond_to_truncate
#     Lambda_i_tensor = Lambdas[i].tensor
#     Lambda_i_inv_tensor = np.diag(1 / np.diag(np.array(Lambda_i_tensor)))
#     Lambda_i_inv = tn.Node(Lambda_i_inv_tensor)
    
#     last_mu = 0
#     mu = np.inf
#     epsilon_D = 1.0e-14

#     while (np.abs(mu-last_mu) > np.abs(mu)*epsilon_D) or (mu == np.inf):
#         network = tn.ncon([B, v], [(-1, -2, 1), (1, -3)])
#         conj_B = tn.conj(B)
#         conj_B[1] ^ network[1]
#         conj_B[2] ^ network[2]
#         output_edge_order=(network[0], conj_B[0])
#         network = tn.contract_between(node1=conj_B,
#                                       node2=network,
#                                       output_edge_order=output_edge_order)

#         L = len(Gammas)
#         kmin = i + 1
#         kmax = L - 2 + i

#         network = tn.ncon([Lambdas[(kmax+1)%L], network], [(-1, 1), (1, -2)])
#         network = tn.ncon([network, Lambdas[(kmax+1)%L]], [(-1, 1), (1, -2)])

#         for k in range(kmax, kmin-1, -1):
#             network = tn.ncon([Gammas[k%L], network], [(-1, -2, 1), (1, -3)])
#             conj_Gamma_k = tn.conj(Gammas[k%L])
#             conj_Gamma_k[1] ^ network[1]
#             conj_Gamma_k[2] ^ network[2]
#             output_edge_order = (network[0], conj_Gamma_k[0])
#             network = tn.contract_between(node1=conj_Gamma_k,
#                                           node2=network,
#                                           output_edge_order=output_edge_order)
#             network = tn.ncon([Lambdas[k%L], network], [(-1, 1), (1, -2)])
#             network = tn.ncon([network, Lambdas[k%L]], [(-1, 1), (1, -2)])

#         network = tn.ncon([A, network], [(-1, -2, 1), (1, -3)])
#         conj_A = tn.conj(A)
#         conj_A[1] ^ network[1]
#         conj_A[2] ^ network[2]
#         output_edge_order=(network[0], conj_A[0])
#         network = tn.contract_between(node1=conj_A,
#                                       node2=network,
#                                       output_edge_order=output_edge_order)

#         network = tn.ncon([Lambda_i_inv, network], [(-1, 1), (1, -2)])
#         network = tn.ncon([network, Lambda_i_inv], [(-1, 1), (1, -2)])

#         last_mu = mu
#         conj_last_v = tn.conj(v)
#         norm_of_last_v = tn.norm(conj_last_v)
#         v = network / tn.norm(network)

#         conj_last_v[0] ^ network[0]
#         conj_last_v[1] ^ network[1]
#         mu = (tn.contract_between(node1=conj_last_v, node2=network)
#               / norm_of_last_v / norm_of_last_v)
#         mu = complex(np.array(mu.tensor))

#     print("mu_R =", mu)
#     V_R = v

#     return V_R



def calc_X(V_L):
    if V_L.backend.name != "numpy":
        sbc._backend.tf_to_np(V_L)

    try:
        D, W = scipy.linalg.eigh(V_L.tensor)
        X_dagger = W @ np.sqrt(np.diag(np.abs(D)))
        X = np.conj(np.transpose(X_dagger))
        X = tn.Node(X)  # Convert from numpy array to tensornetwork node.
    except ValueError as err:
        print(V_L.tensor)
        raise err

    return X



def calc_Y(V_R):
    if V_R.backend.name != "numpy":
        sbc._backend.tf_to_np(V_R)
        
    D, W = scipy.linalg.eigh(V_R.tensor)
    Y = W @ np.sqrt(np.diag(np.abs(D)))
    Y = tn.Node(Y)  # Convert from numpy array to tensornetwork node.

    return Y
