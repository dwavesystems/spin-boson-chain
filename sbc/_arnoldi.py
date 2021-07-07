#!/usr/bin/env python
r"""Contains an implementation of the Arnoldi factorization with implicit 
restart. 
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For general array handling.
import numpy as np

# For calculating the eigenspectra of complex square matrices and performing
# QR factorizations.
import scipy.linalg

# For creating tensor networks and performing contractions.
import tensornetwork as tn

# For switching the ``tensornetwork`` backend.
from sbc import _backend



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

def dominant_eigvec_of_MM_network(M, starting_node=None, epsilon_D=1.0e-14):
    # The dominant eigenvector is calculated using the power iteration method.
    conj_M = tn.conj(M)

    if starting_node is None:
        chi_S = M.shape[-1]
        b = tn.Node(np.random.rand(chi_S, chi_S)
                    +1j*np.random.rand(chi_S, chi_S))
        b /= tn.norm(b)
    else:
        b = starting_node / tn.norm(starting_node)

    last_mu = 0
    mu = np.inf

    while (np.abs(mu-last_mu) > np.abs(mu)*epsilon_D) or (mu == np.inf):
        nodes_to_contract = (M, b)
        network_struct = [(-1, -2, 1), (1, -3)]
        Mb = tn.ncon(nodes_to_contract, network_struct)

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

        

class FactorizationAlg():
    r"""A class implementing the k-step Arnoldi factorization as outlined in
    http://www.netlib.org/utk/people/JackDongarra/etemplates/node221.html.
    """
    def __init__(self, M, krylov_dim, starting_node=None):
        self.M = M
        self.conj_M = tn.conj(M)
        self.step_count = 0
        self.H = np.zeros([krylov_dim, krylov_dim], dtype=np.complex)

        if starting_node is None:
            self.random_f()
        else:
            self.f = starting_node / tn.norm(starting_node)
            
        self.V = [tn.Node(np.zeros(self.f.shape)) for _ in range(krylov_dim)]

        self.beta = tn.norm(self.f)

        return None



    def random_f(self):
        chi_S = self.M.shape[-1]
        self.f = tn.Node(np.random.rand(chi_S, chi_S)
                         +1j*np.random.rand(chi_S, chi_S))
        self.f /= tn.norm(self.f)

        return None



    def apply_MM_network(self, a):
        step_count = self.step_count
        M = self.M
        conj_M = self.conj_M
        
        nodes_to_contract = (M, a)
        network_struct = [(-1, -2, 1), (1, -3)]
        Ma = tn.ncon(nodes_to_contract, network_struct)

        Ma[1] ^ conj_M[1]
        Ma[2] ^ conj_M[2]
        MMa = tn.contract_between(node1=Ma,
                                  node2=conj_M,
                                  output_edge_order=(Ma[0], conj_M[0]))

        return MMa



    def step(self):
        # print("Start of step")
        step_count = self.step_count

        self.V[step_count] = self.f / tn.norm(self.f)
        if step_count > 0:
            self.H[step_count, step_count-1] = self.beta

        # print("v")
        # print(self.V[step_count].tensor)
        w = self.apply_MM_network(a=self.V[step_count])
        h = np.array([vdot(v, w) for v in self.V[:step_count+1]])
        # print("w")
        # print(w.tensor)
        # print("h")
        # print(h)
        
        self.f = w
        for v, h_elem in zip(self.V[:step_count+1], h):
            self.f -= v * tn.Node(h_elem)

        eta = 1 / np.sqrt(2)
        if (tn.norm(self.f) < eta * np.linalg.norm(h)) and (step_count > 0):
            s = np.array([vdot(v, self.f) for v in self.V[:step_count+1]])
            for v, s_elem in zip(self.V[:step_count+1], s):
                self.f -= v * tn.Node(s_elem)
            h += s
            
            if tn.norm(self.f) < eta * np.linalg.norm(h):
                print("Check pt #1")
                self.terminated_early = True
                self.random_f()
                self.beta = 0
                s = np.array([vdot(v, self.f) for v in self.V[:step_count+1]])
                for v, s_elem in zip(self.V[:step_count+1], s):
                    self.f -= v * tn.Node(s_elem)
            else:
                print("Check pt #2")
                self.terminated_early = False
                self.beta = tn.norm(self.f)
        else:
            print("Check pt #3")
            self.terminated_early = False
            self.beta = tn.norm(self.f)

        # print("H contains NaNs:", np.isnan(self.H).any())
        self.H[:step_count+1, step_count] = h
        # print("H contains NaNs:", np.isnan(self.H).any())

        self.step_count += 1

        return None



    def update_V_H_f_beta_and_step_count(self, V, H, f, beta, step_count):
        print("Updated")
        self.V = V
        self.H = H
        self.f = f
        self.beta = beta
        self.step_count = step_count

        return None



    def ritz_eigenpairs(self):
        m = self.H.shape[0]

        H_eigvals, H_eigvecs = scipy.linalg.eig(self.H)

        ritz_eigvals = H_eigvals
        ritz_eigvecs = [tn.Node(np.zeros(self.f.shape)) for i in range(m)]
        for i in range(m):
            for l in range(m):
                ritz_eigvecs[i] += self.V[l] * tn.Node(H_eigvecs[l, i])

        return ritz_eigvals, ritz_eigvecs



class FactorizationWithRestartAlg():
    r"""A class implementing the implicitly restarted Arnoldi factorization as 
    outlined in 
    http://www.netlib.org/utk/people/JackDongarra/etemplates/node222.html.
    """
    def __init__(self, M, krylov_dim=10, starting_node=None, epsilon_D=1.0e-14):
        self.factorization_alg = FactorizationAlg(M, krylov_dim, starting_node)
        self.p = krylov_dim - 1
        self.epsilon_D = epsilon_D  # A kind of relative error tolerance.
        self.num_restarts = 0

        for _ in range(krylov_dim):
            self.factorization_alg.step()
            # if self.factorization_alg.terminated_early:
            #     break

        return None



    def step(self):
        k = 1
        p = self.p
        m = k + p

        ritz_eigvals, ritz_eigvecs = self.factorization_alg.ritz_eigenpairs()
        dominant_ritz_eigval_idx = np.argmax(np.abs(ritz_eigvals))
        self.dominant_ritz_eigval = ritz_eigvals[dominant_ritz_eigval_idx]
        self.dominant_ritz_eigvec = ritz_eigvecs[dominant_ritz_eigval_idx]
        
        shifts = np.delete(ritz_eigvals, dominant_ritz_eigval_idx)

        H = self.factorization_alg.H
        Id = np.eye(H.shape[0])
        Q = Id
        for j in range(p):
            mu = shifts[j]
            mat_to_QR_factorize = H - mu * Id
            Q_j, R_j = scipy.linalg.qr(mat_to_QR_factorize)
            Q_j_dagger = np.conj(np.transpose(Q_j))
            H = Q_j_dagger @ (H @ Q_j)
            Q = Q @ Q_j

        V = self.factorization_alg.V
        f = self.factorization_alg.f
        
        V_plus = [tn.Node(np.zeros(f.shape)) for _ in range(m)]
        for i in range(m):
            for l in range(m):
                V_plus[i] += V[l] * tn.Node(Q[l, i])
        V = V_plus

        beta = H[k, k-1]
        sigma = Q[m-1, k-1]
        f = V[k] * tn.Node(beta) + f * tn.Node(sigma)
        
        beta = tn.norm(f)
        step_count = k

        self.factorization_alg.update_V_H_f_beta_and_step_count(V,
                                                                H,
                                                                f,
                                                                beta,
                                                                step_count)
        for _ in range(p):
            self.factorization_alg.step()

        self.num_restarts += 1

        return None



    def has_converged(self):
        if self.num_restarts == 0:
            result = False
        else:
            theta = tn.Node(self.dominant_ritz_eigval)
            u = self.dominant_ritz_eigvec
            MMu = self.factorization_alg.apply_MM_network(a=u)
            residual_norm = tn.norm(MMu - theta * u) / tn.norm(theta)
            print(residual_norm)
            result = True if residual_norm < self.epsilon_D else False

        return result



# def dominant_eigvec(M, krylov_dim=10, starting_node=None, epsilon_D=1.0e-14):
#     kwargs = {"M": M,
#               "krylov_dim": krylov_dim,
#               "starting_node": starting_node,
#               "epsilon_D": epsilon_D}
#     factorization_with_restart_alg = FactorizationWithRestartAlg(**kwargs)

#     while not factorization_with_restart_alg.has_converged():
#         factorization_with_restart_alg.step()
#         print("eigval =", factorization_with_restart_alg.dominant_ritz_eigval)

#     print(factorization_with_restart_alg.dominant_ritz_eigval)
#     result = factorization_with_restart_alg.dominant_ritz_eigvec

#     return result
        


def vdot(a, b):
    conj_a = tn.conj(a)
    conj_a[0] ^ b[0]
    conj_a[1] ^ b[1]
    result = tn.contract_between(node1=conj_a, node2=b)
    result = complex(np.array(result.tensor))

    return result
