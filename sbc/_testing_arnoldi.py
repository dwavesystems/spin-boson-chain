#!/usr/bin/env python

import numpy as np
import scipy.sparse.linalg
import tensornetwork as tn
import time
import sbc._svd
import sbc._arnoldi

def dominant_eigpair_of_transfer_matrix_using_scipy(mps_nodes,
                                                    M,
                                                    Lambda_i_inv,
                                                    i,
                                                    unit_cell_type,
                                                    krylov_dim,
                                                    starting_node,
                                                    epsilon_D):
    if unit_cell_type == "left":
        mat = construct_left_transfer_matrix(mps_nodes, M, Lambda_i_inv, i)
    else:
        mat = construct_right_transfer_matrix(mps_nodes, M, Lambda_i_inv, i)

    if M.shape[0] > 10:
        print("Using eigs")
        W, V = scipy.sparse.linalg.eigs(mat, k=1)
        dominant_eigval_idx = 0
    else:
        print("Using eig")
        W, V = scipy.linalg.eig(mat)
        # print("W =", W)
        dominant_eigval_idx = np.argmax(np.abs(W))

    dominant_eigval = W[dominant_eigval_idx]
    dominant_eigvec = V[:, dominant_eigval_idx]
    dominant_eigvec = tn.Node(dominant_eigvec.reshape(M.shape[-1], M.shape[-1]))

    if unit_cell_type == "left":
        dominant_eigvec = dominant_eigvec.reorder_axes([1, 0])

    return dominant_eigval, dominant_eigvec



def construct_left_transfer_matrix(mps_nodes, M, Lambda_i_inv, i, C=None):
    L = len(mps_nodes)
    kmin = i + 1
    kmax = (L - 2) + i

    A_i = mps_nodes[i%L]
    conj_A_i = tn.conj(A_i)
    network = tn.ncon([A_i, conj_A_i], [(-1, 1, -3), (-2, 1, -4)])
    
    for k in range(kmin, kmax+1):
        A_k = mps_nodes[k%L]
        conj_A_k = tn.conj(A_k)
        AA_k = tn.ncon([A_k, conj_A_k], [(-1, 1, -3), (-2, 1, -4)])
        network[2] ^ AA_k[0]
        network[3] ^ AA_k[1]
        output_edge_order = (network[0], network[1], AA_k[2], AA_k[3])
        network = tn.contract_between(node1=network,
                                      node2=AA_k,
                                      output_edge_order=output_edge_order)
        
    conj_M = tn.conj(M)
    MM = tn.ncon([M, conj_M], [(-1, 1, -3), (-2, 1, -4)])
    network[2] ^ MM[0]
    network[3] ^ MM[1]
    output_edge_order = (network[0], network[1], MM[2], MM[3])
    network = tn.contract_between(node1=network,
                                  node2=MM,
                                  output_edge_order=output_edge_order)

    network = tn.ncon([network, Lambda_i_inv], [(-1, -2, -3, 1), (1, -4)])
    network = tn.ncon([network, Lambda_i_inv], [(-1, -2, 1, -4), (1, -3)])

    if C is None:
        tn.flatten_edges([network[0], network[1]])
        tn.flatten_edges([network[0], network[1]])
        network.reorder_axes([1, 0])
        mat = np.array(network.tensor)
    else:
        mat = tn.ncon([C, network], [(2, 1), (1, 2, -2, -1)]).tensor

    return mat



def construct_right_transfer_matrix(mps_nodes, M, Lambda_i_inv, i, C=None):
    conj_M = tn.conj(M)
    network = tn.ncon([M, conj_M], [(-1, 1, -3), (-2, 1, -4)])

    L = len(mps_nodes)
    kmin = i
    kmax = (L - 2) + i
    
    for k in range(kmax, kmin-1, -1):
        A_k = mps_nodes[k%L]
        conj_A_k = tn.conj(A_k)
        AA_k = tn.ncon([A_k, conj_A_k], [(-1, 1, -3), (-2, 1, -4)])
        AA_k[2] ^ network[0]
        AA_k[3] ^ network[1]
        output_edge_order = (AA_k[0], AA_k[1], network[2], network[3])
        network = tn.contract_between(node1=AA_k,
                                      node2=network,
                                      output_edge_order=output_edge_order)
            
    network = tn.ncon([Lambda_i_inv, network], [(-1, 1), (1, -2, -3, -4)])
    network = tn.ncon([Lambda_i_inv, network], [(-2, 1), (-1, 1, -3, -4)])
        
    if C is None:
        tn.flatten_edges([network[0], network[1]])
        tn.flatten_edges([network[0], network[1]])
        mat = np.array(network.tensor)
    else:
        mat = tn.ncon([network, C], [(-1, -2, 1, 2), (1, 2)]).tensor

    return mat
        


def generate_random_example(chi, d, L, i):
    tensor = np.random.rand(chi, d, chi) + 1j*np.random.rand(chi, d, chi)
    node = tn.Node(tensor / np.linalg.norm(tensor))
    mps_nodes = [node]*L

    kwargs = {"nodes": mps_nodes,
              "compress_params": None,
              "is_infinite": True,
              "starting_node_idx": i-1}
    schmidt_spectra = sbc._svd.left_to_right_sweep(**kwargs)
    Lambda_i = schmidt_spectra[0]

    Lambda_i_inv_tensor = np.diag(1 / np.diag(np.array(Lambda_i.tensor)))
    Lambda_i_inv = tn.Node(Lambda_i_inv_tensor)

    M = tn.ncon([mps_nodes[(i-1)%L], Lambda_i], [(-1, -2, 1), (1, -3)])
    
    return mps_nodes, M, Lambda_i_inv, i

# chi = 32
# d = 4
# L = 2
# i = 0

# krylov_dim = 10
# starting_node = None
# epsilon_D = 1.0e-14

# mps_nodes, M, Lambda_i_inv, i = generate_random_example(chi, d, L, i)
# kwargs = {"mps_nodes": mps_nodes,
#           "M": M,
#           "Lambda_i_inv": Lambda_i_inv,
#           "i": i,
#           "unit_cell_type": "left",
#           "krylov_dim": krylov_dim,
#           "starting_node": starting_node,
#           "epsilon_D": epsilon_D}



# # print("Testing transfer matrix construction:")
# # print("Left transfer matrix:")
# # C = tn.Node(np.random.rand(chi, chi)+1j*np.random.rand(chi, chi))
# # result_1 = construct_left_transfer_matrix(mps_nodes, M, Lambda_i_inv, i, C=C)
# # result_2 = sbc._arnoldi.left_segment_ev_of_id(C, mps_nodes, M, Lambda_i_inv, i).tensor
# # rel_diff = np.linalg.norm((result_1-result_2)/np.linalg.norm(result_2))
# # # print("result 1 =", result_1)
# # # print("result 2 =", result_2)
# # print("relative difference =", rel_diff)
# # print()
# # print("Right transfer matrix:")
# # C = tn.Node(np.random.rand(chi, chi)+1j*np.random.rand(chi, chi))
# # result_1 = construct_right_transfer_matrix(mps_nodes, M, Lambda_i_inv, i, C=C)
# # result_2 = sbc._arnoldi.right_segment_ev_of_id(C, mps_nodes, M, Lambda_i_inv, i).tensor
# # rel_diff = np.linalg.norm((result_1-result_2)/np.linalg.norm(result_2))
# # # print("result 1 =", result_1)
# # # print("result 2 =", result_2)
# # print("relative difference =", rel_diff)
# # print()



# print("Left unit cell result:")
# kwargs["unit_cell_type"] = "left"

# start_time = time.time()
# dominant_eigval_1, dominant_eigvec_1 = \
#     dominant_eigpair_of_transfer_matrix_using_scipy(**kwargs)
# execution_time = time.time() - start_time
# print("scipy execution time:", execution_time)
# print("dominant_eigval_1 =", dominant_eigval_1)

# start_time = time.time()
# dominant_eigval_2, dominant_eigvec_2 = \
#     sbc._arnoldi.dominant_eigpair_of_transfer_matrix(**kwargs)
# execution_time = time.time() - start_time
# print("sbc execution time:", execution_time)
# print("dominant_eigval_2 =", dominant_eigval_2)

# rel_diff = abs((dominant_eigval_1-dominant_eigval_2) / dominant_eigval_1)

# conj_dominant_eigvec_2 = tn.conj(dominant_eigvec_2)
# overlap = tn.ncon([conj_dominant_eigvec_2, dominant_eigvec_1], [(1, 2), (1, 2)])
# overlap = float(np.abs(np.array(overlap.tensor)))

# print("relative difference =", rel_diff)
# print("overlap =", overlap)
# print()



# print("right unit cell result:")
# kwargs["unit_cell_type"] = "right"

# start_time = time.time()
# dominant_eigval_1, dominant_eigvec_1 = \
#     dominant_eigpair_of_transfer_matrix_using_scipy(**kwargs)
# execution_time = time.time() - start_time
# print("scipy execution time:", execution_time)
# print("dominant_eigval_1 =", dominant_eigval_1)

# start_time = time.time()
# dominant_eigval_2, dominant_eigvec_2 = \
#     sbc._arnoldi.dominant_eigpair_of_transfer_matrix(**kwargs)
# execution_time = time.time() - start_time
# print("sbc execution time:", execution_time)
# print("dominant_eigval_2 =", dominant_eigval_2)

# rel_diff = abs((dominant_eigval_1-dominant_eigval_2) / dominant_eigval_1)

# conj_dominant_eigvec_2 = tn.conj(dominant_eigvec_2)
# overlap = tn.ncon([conj_dominant_eigvec_2, dominant_eigvec_1], [(1, 2), (1, 2)])
# overlap = float(np.abs(np.array(overlap.tensor)))

# print("relative difference =", rel_diff)
# print("overlap =", overlap)
