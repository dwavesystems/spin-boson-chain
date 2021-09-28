# Copyright 2021 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

r"""Contains an implementation of the Arnoldi factorization with implicit 
restart for specially structured tensors. This module is used exclusively by the
``spinbosonchain._mpomps`` module.
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
from spinbosonchain import _backend



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

def left_segment_ev_of_id(C, Lambda_Theta):
    # [1]: Annals of Physics 326 (2011) 96-192.

    # This function evaluates the generalization of Eq. (351) of [1] to an
    # L-site unit cell, where L>0.

    # See comments in function spinbosonchain._svd.Lambda_Theta_form for a
    # description of the Lambda_Theta object.
    Lambda = Lambda_Theta[0]
    Theta_nodes = Lambda_Theta[1]

    network = tn.ncon([C, Lambda], [(-1, 1), (1, -2)])
    network = tn.ncon([Lambda, network], [(-1, 1), (1, -2)])

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

    result = network

    return result



def right_segment_ev_of_id(C, Lambda_Theta):
    # [1]: Annals of Physics 326 (2011) 96-192.

    # This function evaluates the generalization of Eq. (354) of [1] to an
    # L-site unit cell, where L>0.

    # See comments in function spinbosonchain._svd.Lambda_Theta_form for a
    # description of the Lambda_Theta object.
    Lambda = Lambda_Theta[0]
    Theta_nodes = Lambda_Theta[1]

    network = tn.ncon([Lambda, C], [(-1, 1), (1, -2)])
    network = tn.ncon([network, Lambda], [(-1, 1), (1, -2)])

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

    result = network

    return result



def vdot(a, b):
    # This function is a generalization of numpy's vdot function, where a and b
    # are both rank-2 tensors.
    conj_a = tn.conj(a)
    conj_a[0] ^ b[0]
    conj_a[1] ^ b[1]
    result = tn.contract_between(node1=conj_a, node2=b)
    result = complex(np.array(result.tensor))

    return result

        

class FactorizationAlg():
    r"""A class implementing the k-step Arnoldi factorization as outlined in
    http://www.netlib.org/utk/people/JackDongarra/etemplates/node221.html ([2]).
    """
    def __init__(self, Lambda_Theta, unit_cell_type, krylov_dim):
        # [1]: Annals of Physics 326 (2011) 96-192.
        
        # See comments in function spinbosonchain._svd.Lambda_Theta_form for a
        # description of the Lambda_Theta object.
        self.Lambda_Theta = Lambda_Theta
        
        # unit_cell_type indicates whether we are applying the Arnoldi
        # factorization to either the transfer matrix given by Eq. (351) or
        # (354) of [1]: if unit_cell_type="left" then its Eq. (351), else if
        # unit_cell_type="right" then its Eq. (354).
        self.unit_cell_type = unit_cell_type

        # krylov_dim is essentially k in [2].
        self.krylov_dim = krylov_dim
        
        self.step_count = 0

        # Indicates whether the Arnoldi procedure should be terminated early.
        self.terminated_early = False

        # The upper Hessenberg matrix in [2], H_k.
        self.H = np.zeros([krylov_dim, krylov_dim], dtype=np.complex)

        # self.f is f in [2].
        self.random_f()

        # self.V stores the Arnoldi vector discussed in [2].
        self.V = [tn.Node(np.zeros(self.f.shape)) for _ in range(krylov_dim)]

        # self.beta is beta in [2].
        self.beta = tn.norm(self.f)

        return None



    def random_f(self):
        # See comments in __init__ for descriptions of various quantities.
        chi = self.Lambda_Theta[0].shape[0]
        
        # Ensure the random f is Hermitian and semidefinite.
        mat = np.random.rand(chi, chi) + 1j*np.random.rand(chi, chi)
        self.f = tn.Node(np.conj(np.transpose(mat)) @ mat)
        self.f /= tn.norm(self.f)

        return None



    def gen_new_krylov_vector(self, C):
        # See comments in __init__ for descriptions of various quantities.
        if self.unit_cell_type == "left":
            C = C.copy()
            C = C.reorder_axes([1, 0])
            w_T = left_segment_ev_of_id(C, self.Lambda_Theta)
            w = w_T.reorder_axes([1, 0])
        else:
            w = right_segment_ev_of_id(C, self.Lambda_Theta)
            
        return w



    def step(self):
        # [2]: http://www.netlib.org/utk/people/JackDongarra/
        #      etemplates/node221.html
        # See comments in __init__ for descriptions of various quantities.
        # This method implements a single step of algorithm 7.35 of [2].
        step_count = self.step_count

        self.V[step_count] = self.f / tn.norm(self.f)
        if step_count > 0:
            self.H[step_count, step_count-1] = self.beta

        w = self.gen_new_krylov_vector(C=self.V[step_count])
        h = np.array([vdot(v, w) for v in self.V[:step_count+1]])
        
        self.f = w
        for v, h_elem in zip(self.V[:step_count+1], h):
            self.f -= v * tn.Node(h_elem)

        eta = 1 / np.sqrt(2)
        if (tn.norm(self.f) < eta * np.linalg.norm(h)) and (step_count > 0):
            s = np.array([vdot(v, self.f) for v in self.V[:step_count+1]])
            for v, s_elem in zip(self.V[:step_count+1], s):
                self.f -= v * tn.Node(s_elem)
            h += s
            self.beta = tn.norm(self.f)
        else:
            self.beta = tn.norm(self.f)

        self.H[:step_count+1, step_count] = h

        self.step_count += 1

        tol = 1.0e-14
        if self.beta < tol:
            self.terminated_early = True

        return None



    def update_V_H_f_beta_and_step_count(self, V, H, f, beta, step_count):
        # This method is called by the implicit restart Arnoldi scheme.
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
        # [3]: http://www.netlib.org/utk/people/JackDongarra/
        #      etemplates/node216.html
        # This method calculates the Ritz eigenpairs of our transfer matrix.
        # See [3] for a discussion on Ritz eigenvalues and eigenvectors.
        m = self.step_count

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
    http://www.netlib.org/utk/people/JackDongarra/etemplates/node222.html ([4]).
    """
    def __init__(self, Lambda_Theta, unit_cell_type, krylov_dim, epsilon_D):
        # [1]: Annals of Physics 326 (2011) 96-192.
        
        # See comments in function spinbosonchain._svd.Lambda_Theta_form for a
        # description of the Lambda_Theta object.

        # unit_cell_type indicates whether we are applying the Arnoldi procedure
        # to either the transfer matrix given by Eq. (351) or (354) of [1]: if
        # unit_cell_type="left" then its Eq. (351), else if
        # unit_cell_type="right" then its Eq. (354).

        # krylov_dim is essentially m in [4].

        # The implicit restart scheme requires the factorization scheme.
        self.factorization_alg = FactorizationAlg(Lambda_Theta,
                                                  unit_cell_type,
                                                  krylov_dim)

        # self.k and self.p are k and p in [4] respectively.
        self.k = 1  # We're only interested in the dominant eigenpair.
        self.p = krylov_dim - self.k
        self.epsilon_D = epsilon_D  # A kind of relative error tolerance.
        self.num_restarts = 0

        for _ in range(krylov_dim):
            if self.factorization_alg.terminated_early:
                break
            self.factorization_alg.step()

        return None



    def step(self):
        # [4]: http://www.netlib.org/utk/people/JackDongarra/
        #      etemplates/node222.html
        # See comments in __init__ for descriptions of various quantities.
        # This method implements a single step of algorithm 7.36 of [4].
        k = self.k
        p = self.p
        m = k + p

        self.update_ritz_eigenpairs()
        
        shifts = np.delete(self.ritz_eigvals, self.ritz_eigval_order[:k])

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
        f = V[k] * tn.Node(beta) + f * tn.Node(sigma)
        
        beta = tn.norm(f)
        step_count = k

        # Restart Arnoldi factorization.
        self.factorization_alg.update_V_H_f_beta_and_step_count(V,
                                                                H,
                                                                f,
                                                                beta,
                                                                step_count)

        for _ in range(p):
            if self.factorization_alg.terminated_early:
                break
            self.factorization_alg.step()

        self.num_restarts += 1

        return None



    def has_converged(self):
        # See comments in __init__ for descriptions of various quantities.
        # This method implements a single step of algorithm 7.36 of [4].

        # This method checks whether the Arnoldi procedure has converged.
        
        if self.factorization_alg.terminated_early:
            self.update_ritz_eigenpairs()
            if self.factorization_alg.unit_cell_type == "left":
                self.dominant_ritz_eigvec = \
                    self.dominant_ritz_eigvec.reorder_axes([1, 0])
            result = True
        elif self.num_restarts == 0:
            result = False
        else:
            eigval = tn.Node(self.dominant_ritz_eigval)
            u = self.dominant_ritz_eigvec
            wu = self.factorization_alg.gen_new_krylov_vector(C=u)
            residual_norm = tn.norm(wu - eigval * u) / tn.norm(eigval)
            result = True if residual_norm < self.epsilon_D else False

            if self.factorization_alg.unit_cell_type == "left":
                self.dominant_ritz_eigvec = \
                    self.dominant_ritz_eigvec.reorder_axes([1, 0])


        return result



    def update_ritz_eigenpairs(self):
        # [3]: http://www.netlib.org/utk/people/JackDongarra/
        #      etemplates/node216.html
        
        # This method updates attributes storing Ritz eigenpairs. See [3] for a
        # discussion on Ritz eigenvalues and eigenvectors.
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



def dominant_eigpair_of_transfer_matrix(Lambda_Theta,
                                        unit_cell_type,
                                        krylov_dim,
                                        epsilon_D):
    # [1]: Annals of Physics 326 (2011) 96-192.

    # See comments in function spinbosonchain._svd.Lambda_Theta_form for a
    # description of the Lambda_Theta object.

    # unit_cell_type indicates whether we are finding the dominant eigenpair
    # to either the transfer matrix given by Eq. (351) or (354) of [1]: if
    # unit_cell_type="left" then its Eq. (351), else if unit_cell_type="right"
    # then its Eq. (354).

    # See comments in constructor of FactorizationWithRestartAlg for
    # descriptions of krylov_dim and epsilon_D.
    kwargs = {"Lambda_Theta": Lambda_Theta,
              "unit_cell_type": unit_cell_type,
              "krylov_dim": krylov_dim,
              "epsilon_D": epsilon_D}
    factorization_with_restart_alg = FactorizationWithRestartAlg(**kwargs)

    while not factorization_with_restart_alg.has_converged():
        factorization_with_restart_alg.step()

    dominant_eigval = factorization_with_restart_alg.dominant_ritz_eigval
    dominant_eigvec = factorization_with_restart_alg.dominant_ritz_eigvec

    return dominant_eigval, dominant_eigvec
