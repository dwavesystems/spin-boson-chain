#!/usr/bin/env python

import numpy as np
import scipy.sparse.linalg
import tensornetwork as tn
import time

def right_dominant_eigvec_of_R(Gamma, S, starting_node=None):
    # The dominant eigenvector is calculated using the power iteration method.
    M = tn.ncon((Gamma, S), ((-1, -2, 1), (1, -3)))
    conj_M = tn.conj(M)

    if starting_node is None:
        b = random_starting_node_of_power_iteration(S)
    else:
        b = starting_node / tn.norm(starting_node)

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

    result = b

    return result



def left_dominant_eigvec_of_L(Gamma, S, starting_node=None):
    # The dominant eigenvector is calculated using the power iteration method.
    M = tn.ncon((S, Gamma), ((-1, 1), (1, -2, -3)))
    conj_M = tn.conj(M)

    if starting_node is None:
        conj_b = random_starting_node_of_power_iteration(S)
    else:
        conj_b = starting_node / tn.norm(starting_node)

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

    result = conj_b

    return result



def random_starting_node_of_power_iteration(S):
    chi_s = S.shape[-1]  # Spatial bond dimension.
    b = tn.Node(np.random.rand(chi_s, chi_s)
                +1j*np.random.rand(chi_s, chi_s))
    b /= tn.norm(b)

    return b



chi_t = 64
chi_s = 64

Gamma = tn.Node(np.random.rand(chi_s, chi_t, chi_s)
                +1j*np.random.rand(chi_s, chi_t, chi_s))
S = tn.Node(np.diag(np.sort(np.random.uniform(size=chi_s))[-1::-1]))

start_time = time.time()
M = tn.ncon((Gamma, S), ((-1, -2, 1), (1, -3)))
conj_M = tn.conj(M)
MM = tn.ncon((M, conj_M), ((-1, 1, -3), (-2, 1, -4)))
tn.flatten_edges([MM[0], MM[1]])
tn.flatten_edges([MM[0], MM[1]])
mat = MM.tensor
eigvals, eigvecs = scipy.sparse.linalg.eigs(mat, k=1)
scipy_eigvec = eigvecs[:, 0]
execution_time = time.time() - start_time
print("scipy execution time:", execution_time)

start_time = time.time()
sbc_eigvec = right_dominant_eigvec_of_R(Gamma, S)
tn.flatten_edges([sbc_eigvec[0], sbc_eigvec[1]])
sbc_eigvec = sbc_eigvec.tensor
execution_time = time.time() - start_time
print("scipy execution time:", execution_time)

overlap = np.vdot(sbc_eigvec, scipy_eigvec)
print("|overlap| =", np.abs(overlap))

print()

start_time = time.time()
M = tn.ncon((S, Gamma), ((-1, 1), (1, -2, -3)))
conj_M = tn.conj(M)
MM = tn.ncon((M, conj_M), ((-1, 1, -3), (-2, 1, -4)))
tn.flatten_edges([MM[0], MM[1]])
tn.flatten_edges([MM[0], MM[1]])
mat = np.transpose(MM.tensor)
eigvals, eigvecs = scipy.sparse.linalg.eigs(mat, k=1)
scipy_eigvec = eigvecs[:, 0]
execution_time = time.time() - start_time
print("scipy execution time:", execution_time)

start_time = time.time()
sbc_eigvec = left_dominant_eigvec_of_L(Gamma, S)
tn.flatten_edges([sbc_eigvec[0], sbc_eigvec[1]])
sbc_eigvec = sbc_eigvec.tensor
execution_time = time.time() - start_time
print("scipy execution time:", execution_time)

overlap = np.vdot(sbc_eigvec, scipy_eigvec)
print("|overlap| =", np.abs(overlap))
