#!/usr/bin/env python
r"""Contains an implementation of the Arnoldi factorization with implicit 
restart for specially structured tensors. This module is used exclusively by the
``sbc._mpomps`` module.
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

# def left_segment_ev_of_id(C, mps_nodes, M, Lambda_i_inv, i):
#     network = C

#     L = len(mps_nodes)
#     kmin = i  # 'i' is the starting MPS node index from which to cycle through.
#     kmax = (L - 2) + i
    
#     for k in range(kmin, kmax+1):
#         A_k = mps_nodes[k%L]
#         network = tn.ncon([network, A_k], [(-1, 1), (1, -2, -3)])
#         conj_A_k = tn.conj(A_k)
#         network[0] ^ conj_A_k[0]
#         network[1] ^ conj_A_k[1]
#         output_edge_order = (conj_A_k[2], network[2])
#         network = tn.contract_between(node1=network,
#                                       node2=conj_A_k,
#                                       output_edge_order=output_edge_order)

#     network = tn.ncon([network, M], [(-1, 1), (1, -2, -3)])
#     conj_M = tn.conj(M)
#     network[0] ^ conj_M[0]
#     network[1] ^ conj_M[1]
#     output_edge_order=(conj_M[2], network[2])
#     network = tn.contract_between(node1=network,
#                                   node2=conj_M,
#                                   output_edge_order=output_edge_order)

#     network = tn.ncon([network, Lambda_i_inv], [(-1, 1), (1, -2)])
#     network = tn.ncon([Lambda_i_inv, network], [(-1, 1), (1, -2)])

#     result = network

#     return result



# def right_segment_ev_of_id(C, mps_nodes, M, Lambda_i_inv, i):
#     network = tn.ncon([M, C], [(-1, -2, 1), (1, -3)])
#     conj_M = tn.conj(M)
#     conj_M[1] ^ network[1]
#     conj_M[2] ^ network[2]
#     output_edge_order=(network[0], conj_M[0])
#     network = tn.contract_between(node1=conj_M,
#                                   node2=network,
#                                   output_edge_order=output_edge_order)

#     L = len(mps_nodes)
#     kmin = i  # 'i' is the starting MPS node index from which to cycle through.
#     kmax = (L - 2) + i
    
#     for k in range(kmax, kmin-1, -1):
#         A_k = mps_nodes[k%L]
#         network = tn.ncon([A_k, network], [(-1, -2, 1), (1, -3)])
#         conj_A_k = tn.conj(A_k)
#         conj_A_k[1] ^ network[1]
#         conj_A_k[2] ^ network[2]
#         output_edge_order = (network[0], conj_A_k[0])
#         network = tn.contract_between(node1=conj_A_k,
#                                       node2=network,
#                                       output_edge_order=output_edge_order)

#     network = tn.ncon([Lambda_i_inv, network], [(-1, 1), (1, -2)])
#     network = tn.ncon([network, Lambda_i_inv], [(-1, 1), (1, -2)])

#     result = network

#     return result



# def left_segment_ev_of_id(r, C, Gammas, Lambdas):
# def left_segment_ev_of_id(C, Gammas, Lambdas):
#     network = C

#     L = len(Gammas)
#     # kmin = r
#     # kmax = (L - 1) + r
#     kmin = 0
#     kmax = (L - 1)
    
#     for k in range(kmin, kmax+1):
#         M_k = tn.ncon([Lambdas[k%L], Gammas[k%L]], [(-1, 1), (1, -2, -3)])
#         network = tn.ncon([network, M_k], [(-1, 1), (1, -2, -3)])
#         conj_M_k = tn.conj(M_k)
#         network[0] ^ conj_M_k[0]
#         network[1] ^ conj_M_k[1]
#         output_edge_order = (conj_M_k[2], network[2])
#         network = tn.contract_between(node1=network,
#                                       node2=conj_M_k,
#                                       output_edge_order=output_edge_order)

#     result = network

#     return result



def left_segment_ev_of_id(C, Lambda_Theta):
    Lambda = Lambda_Theta[0]
    Theta_nodes = Lambda_Theta[1]

    network = tn.ncon([C, Lambda], [(-1, 1), (1, -2)])
    network = tn.ncon([Lambda, network], [(-1, 1), (1, -2)])

    # print("left_segment_ev_of_id check pts:")
    # print("current network abs max =", np.amax(np.abs(network.tensor)))

    for Theta_node in Theta_nodes:
        M = Theta_node
        network = tn.ncon([network, M], [(-1, 1), (1, -2, -3)])
        conj_M = tn.conj(M)
        network[0] ^ conj_M[0]
        network[1] ^ conj_M[1]
        output_edge_order = (conj_M[2], network[2])
        network = tn.contract_between(node1=network,
                                      node2=conj_M,
                                      output_edge_order=output_edge_order)
        # print("current network abs max =", np.amax(np.abs(network.tensor)))

    result = network

    return result



# def right_segment_ev_of_id(r, C, Gammas, Lambdas):
# def right_segment_ev_of_id(C, Gammas, Lambdas):
#     network = C
    
#     L = len(Gammas)
#     # kmin = r
#     # kmax = (L - 1) + r
#     kmin = 0
#     kmax = (L - 1)
    
#     for k in range(kmax, kmin-1, -1):
#         M_k = tn.ncon([Gammas[k%L], Lambdas[(k+1)%L]], [(-1, -2, 1), (1, -3)])
#         network = tn.ncon([M_k, network], [(-1, -2, 1), (1, -3)])
#         conj_M_k = tn.conj(M_k)
#         conj_M_k[1] ^ network[1]
#         conj_M_k[2] ^ network[2]
#         output_edge_order = (network[0], conj_M_k[0])
#         network = tn.contract_between(node1=conj_M_k,
#                                       node2=network,
#                                       output_edge_order=output_edge_order)

#     result = network

#     return result



def right_segment_ev_of_id(C, Lambda_Theta):
    Lambda = Lambda_Theta[0]
    Theta_nodes = Lambda_Theta[1]

    network = tn.ncon([Lambda, C], [(-1, 1), (1, -2)])
    network = tn.ncon([network, Lambda], [(-1, 1), (1, -2)])

    # print("right_segment_ev_of_id check pts:")
    # print("current network abs max =", np.amax(np.abs(network.tensor)))

    for Theta_node in Theta_nodes[::-1]:
        M = Theta_node
        network = tn.ncon([M, network], [(-1, -2, 1), (1, -3)])
        conj_M = tn.conj(M)
        conj_M[1] ^ network[1]
        conj_M[2] ^ network[2]
        output_edge_order = (network[0], conj_M[0])
        network = tn.contract_between(node1=conj_M,
                                      node2=network,
                                      output_edge_order=output_edge_order)
        # print("current network abs max =", np.amax(np.abs(network.tensor)))

    result = network

    return result



# def dominant_eigpair_of_transfer_matrix(mps_nodes,
#                                         M,
#                                         Lambda_i_inv,
#                                         i,
#                                         unit_cell_type,
#                                         krylov_dim,
#                                         starting_node,
#                                         epsilon_D):
# def dominant_eigpair_of_transfer_matrix(r,
#                                         Gammas,
# def dominant_eigpair_of_transfer_matrix(Gammas,
#                                         Lambdas,
def dominant_eigpair_of_transfer_matrix(Lambda_Theta,
                                        unit_cell_type,
                                        krylov_dim,
                                        starting_node,
                                        epsilon_D):
    # kwargs = {"mps_nodes": mps_nodes,
    #           "M": M,
    #           "Lambda_i_inv": Lambda_i_inv,
    #           "i": i,
    #           "unit_cell_type": unit_cell_type,
    #           "krylov_dim": krylov_dim,
    #           "starting_node": starting_node,
    #           "epsilon_D": epsilon_D}
    # kwargs = {"r": r,
    #           "Gammas": Gammas,
    # kwargs = {"Gammas": Gammas,
    #           "Lambdas": Lambdas,
    kwargs = {"Lambda_Theta": Lambda_Theta,
              "unit_cell_type": unit_cell_type,
              "krylov_dim": krylov_dim,
              "starting_node": starting_node,
              "epsilon_D": epsilon_D}
    factorization_with_restart_alg = FactorizationWithRestartAlg(**kwargs)

    while not factorization_with_restart_alg.has_converged():
        factorization_with_restart_alg.step()
        # print("eigval =", factorization_with_restart_alg.dominant_ritz_eigval)

    dominant_eigval = factorization_with_restart_alg.dominant_ritz_eigval
    dominant_eigvec = factorization_with_restart_alg.dominant_ritz_eigvec

    if unit_cell_type == "left":
        mu_L = dominant_eigval
        V_L = dominant_eigvec
        V_L_approx = left_segment_ev_of_id(V_L, Lambda_Theta) / tn.Node(mu_L)
        diff = tn.norm(V_L - V_L_approx)
    else:
        mu_R = dominant_eigval
        V_R = dominant_eigvec
        V_R_approx = right_segment_ev_of_id(V_R, Lambda_Theta) / tn.Node(mu_R)
        diff = tn.norm(V_R - V_R_approx)

    print("dominant eigvec err =", diff)
    # print()

    return dominant_eigval, dominant_eigvec

        

class FactorizationAlg():
    r"""A class implementing the k-step Arnoldi factorization as outlined in
    http://www.netlib.org/utk/people/JackDongarra/etemplates/node221.html.
    """
    # def __init__(self,
    #              mps_nodes,
    #              M,
    #              Lambda_i_inv,
    #              i,
    #              unit_cell_type,
    #              krylov_dim,
    #              starting_node):
    def __init__(self,
                 # r,
                 # Gammas,
                 # Lambdas,
                 Lambda_Theta,
                 unit_cell_type,
                 krylov_dim,
                 starting_node):
        # self.mps_nodes = mps_nodes
        # self.M = M
        # self.Lambda_i_inv = Lambda_i_inv
        # self.i = i
        # self.unit_cell_type = unit_cell_type
        # self.krylov_dim = krylov_dim

        # self.r = r
        # self.Gammas = Gammas
        # self.Lambdas = Lambdas
        self.Lambda_Theta = Lambda_Theta
        self.unit_cell_type = unit_cell_type
        self.krylov_dim = krylov_dim
        
        self.step_count = 0
        self.terminated_early = False
        self.H = np.zeros([krylov_dim, krylov_dim], dtype=np.complex)

        if starting_node is None:
            self.random_f()
        else:
            self.f = starting_node / tn.norm(starting_node)
            
        self.V = [tn.Node(np.zeros(self.f.shape)) for _ in range(krylov_dim)]

        self.beta = tn.norm(self.f)

        return None



    def random_f(self):
        # L = len(self.Gammas)
        chi = self.Lambda_Theta[0].shape[0]
        
        # if self.unit_cell_type == "left":
        #     # L = len(self.mps_nodes)
        #     # chi = self.mps_nodes[self.i%L].shape[0]
        #     # chi = self.Gammas[self.r%L].shape[0]
        #     chi = self.Gammas[0].shape[0]
        # else:
        #     # chi = self.M.shape[-1]
        #     chi = self.Gammas[(self.r-1)%L].shape[-1]

        # Ensure the random f is Hermitian and semidefinite.
        mat = np.random.rand(chi, chi) + 1j*np.random.rand(chi, chi)
        self.f = tn.Node(np.conj(np.transpose(mat)) @ mat)
        self.f /= tn.norm(self.f)

        return None



    def gen_new_krylov_vector(self, C):
        if self.unit_cell_type == "left":
            C = C.copy()
            C = C.reorder_axes([1, 0])
            # w_T = left_segment_ev_of_id(C,
            #                             self.mps_nodes,
            #                             self.M, self.
            #                             Lambda_i_inv,
            #                             self.i)
            # w_T = left_segment_ev_of_id(self.r, C, self.Gammas, self.Lambdas)
            # w_T = left_segment_ev_of_id(C, self.Gammas, self.Lambdas)
            w_T = left_segment_ev_of_id(C, self.Lambda_Theta)
            w = w_T.reorder_axes([1, 0])
        else:
            # w = right_segment_ev_of_id(C,
            #                            self.mps_nodes,
            #                            self.M, self.
            #                            Lambda_i_inv,
            #                            self.i)
            # w = right_segment_ev_of_id(self.r, C, self.Gammas, self.Lambdas)
            # w = right_segment_ev_of_id(C, self.Gammas, self.Lambdas)
            w = right_segment_ev_of_id(C, self.Lambda_Theta)
            
        return w



    def step(self):
        # print("Start of step")
        step_count = self.step_count

        # print("norm of f =", tn.norm(self.f))
        self.V[step_count] = self.f / tn.norm(self.f)
        if step_count > 0:
            self.H[step_count, step_count-1] = self.beta

        # print(self.V[step_count].tensor)
        w = self.gen_new_krylov_vector(C=self.V[step_count])
        h = np.array([vdot(v, w) for v in self.V[:step_count+1]])
        # print("w")
        # print(w.tensor)
        # print("h")
        # print(h)
        
        self.f = w
        for v, h_elem in zip(self.V[:step_count+1], h):
            self.f -= v * tn.Node(h_elem)

        # eta = 1e-2 / np.sqrt(2)
        eta = 1 / np.sqrt(2)
        if (tn.norm(self.f) < eta * np.linalg.norm(h)) and (step_count > 0):
            s = np.array([vdot(v, self.f) for v in self.V[:step_count+1]])
            for v, s_elem in zip(self.V[:step_count+1], s):
                self.f -= v * tn.Node(s_elem)
            h += s

            if False:
            # if tn.norm(self.f) < eta * np.linalg.norm(h):
                # print("Check pt #1; {}; {}".format(tn.norm(self.f), np.linalg.norm(h)))
                # self.terminated_early = True
                self.terminated_early = False
                self.random_f()
                self.beta = 0
                s = np.array([vdot(v, self.f) for v in self.V[:step_count+1]])
                for v, s_elem in zip(self.V[:step_count+1], s):
                    self.f -= v * tn.Node(s_elem)
            else:
                # print("Check pt #2")
                self.terminated_early = False
                self.beta = tn.norm(self.f)
        else:
            # print("Check pt #3")
            self.terminated_early = False
            self.beta = tn.norm(self.f)

        # print("H contains NaNs:", np.isnan(self.H).any())
        self.H[:step_count+1, step_count] = h
        # print("H contains NaNs:", np.isnan(self.H).any())
        # print("beta =", self.beta)

        self.step_count += 1

        tol = 1.0e-14
        if self.beta < tol:
            self.terminated_early = True

        return None



    def update_V_H_f_beta_and_step_count(self, V, H, f, beta, step_count):
        # print("Updated")
        self.V = V
        self.H = H
        self.f = f
        self.beta = beta
        self.step_count = step_count

        tol = 1.0e-14
        if self.beta < tol:
            self.terminated_early = True

        return None



    def ritz_eigenpairs(self):
        # print("H.shape =", self.H.shape)
        m = self.step_count
        # m = self.H.shape[0]

        H_eigvals, H_eigvecs = scipy.linalg.eig(self.H[:m, :m])

        ritz_eigvals = H_eigvals
        ritz_eigvecs = [tn.Node(np.zeros(self.f.shape)) for j in range(m)]
        for j in range(m):
            for l in range(m):
                ritz_eigvecs[j] += self.V[l] * tn.Node(H_eigvecs[l, j])

        return ritz_eigvals, ritz_eigvecs



class FactorizationWithRestartAlg():
    r"""A class implementing the implicitly restarted Arnoldi factorization as 
    outlined in 
    http://www.netlib.org/utk/people/JackDongarra/etemplates/node222.html.
    """
    # def __init__(self,
    #              mps_nodes,
    #              M,
    #              Lambda_i_inv,
    #              i,
    #              unit_cell_type,
    #              krylov_dim,
    #              starting_node,
    #              epsilon_D):
    def __init__(self,
                 # r,
                 # Gammas,
                 # Lambdas,
                 Lambda_Theta,
                 unit_cell_type,
                 krylov_dim,
                 starting_node,
                 epsilon_D):
        # self.factorization_alg = FactorizationAlg(mps_nodes,
        #                                           M,
        #                                           Lambda_i_inv,
        #                                           i,
        #                                           unit_cell_type,
        #                                           krylov_dim,
        #                                           starting_node)
        # self.factorization_alg = FactorizationAlg(r,
        #                                           Gammas,
        # self.factorization_alg = FactorizationAlg(Gammas,
        #                                           Lambdas,
        self.factorization_alg = FactorizationAlg(Lambda_Theta,
                                                  unit_cell_type,
                                                  krylov_dim,
                                                  starting_node)
        self.k = 1
        self.p = krylov_dim - self.k
        self.epsilon_D = epsilon_D  # A kind of relative error tolerance.
        self.num_restarts = 0

        for _ in range(krylov_dim):
            if self.factorization_alg.terminated_early:
                print("terminated early check pt #1")
                break
            self.factorization_alg.step()

        return None



    def step(self):
        k = self.k
        p = self.p
        m = k + p

        self.update_ritz_eigenpairs()
        
        shifts = np.delete(self.ritz_eigvals, self.ritz_eigval_order[:k])
        # print("# shifts =", len(shifts))

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
        for j in range(k):
            for l in range(m):
                V_plus[j] += V[l] * tn.Node(Q[l, j])
        V = V_plus

        H_plus = np.zeros([m, m], dtype=np.complex)
        for j in range(k):
            for l in range(k):
                H_plus[j, l] = H[j, l]
        H = H_plus

        beta = H[k, k-1]
        sigma = Q[m-1, k-1]
        # print("beta = {}; sigma = {}".format(beta, sigma))
        f = V[k] * tn.Node(beta) + f * tn.Node(sigma)
        
        beta = tn.norm(f)
        step_count = k

        self.factorization_alg.update_V_H_f_beta_and_step_count(V,
                                                                H,
                                                                f,
                                                                beta,
                                                                step_count)

        print("Restarted")
        for _ in range(p):
            if self.factorization_alg.terminated_early:
                print("terminated early check pt #2")
                break
            self.factorization_alg.step()

        self.num_restarts += 1

        return None



    def has_converged(self):
        if self.factorization_alg.terminated_early:
            print("terminated early check pt #3")
            self.update_ritz_eigenpairs()
            if self.factorization_alg.unit_cell_type == "left":
                self.dominant_ritz_eigvec = \
                    self.dominant_ritz_eigvec.reorder_axes([1, 0])
            result = True
        elif self.num_restarts == 0:
            result = False
        else:
            theta = tn.Node(self.dominant_ritz_eigval)
            u = self.dominant_ritz_eigvec
            wu = self.factorization_alg.gen_new_krylov_vector(C=u)
            # print("norm of theta =", tn.norm(theta))
            residual_norm = tn.norm(wu - theta * u) / tn.norm(theta)
            # print("residual_norm =", residual_norm)
            result = True if residual_norm < self.epsilon_D else False

            if self.factorization_alg.unit_cell_type == "left":
                self.dominant_ritz_eigvec = \
                    self.dominant_ritz_eigvec.reorder_axes([1, 0])


        return result



    def update_ritz_eigenpairs(self):
        self.ritz_eigvals, self.ritz_eigvecs = \
            self.factorization_alg.ritz_eigenpairs()
        
        self.dominant_ritz_eigval_idx = \
            np.argmax(np.abs(self.ritz_eigvals))
        self.ritz_eigval_order = np.argsort(np.abs(self.ritz_eigvals))[::-1]
        
        self.dominant_ritz_eigval = \
            self.ritz_eigvals[self.dominant_ritz_eigval_idx]
        self.dominant_ritz_eigvec = \
            self.ritz_eigvecs[self.dominant_ritz_eigval_idx]

        return None
        


def vdot(a, b):
    conj_a = tn.conj(a)
    conj_a[0] ^ b[0]
    conj_a[1] ^ b[1]
    result = tn.contract_between(node1=conj_a, node2=b)
    result = complex(np.array(result.tensor))

    return result
