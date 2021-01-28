#!/usr/bin/env python
r"""This module contains classes that can be used to construct certain tensors 
that encode the phase factor effects of z-fields and zz-couplers.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For creating multi-dimensional arrays to be used to construct tensor nodes
# and networks.
import numpy as np

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# For converting base-4 variables to Ising spin pairs.
from sbc._base4 import base_4_to_ising_pair



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

def calc_weights_and_k_prime_set(k, n):
    if k == 0:
        k_prime_set = (0,)
        weights = (0.5,)
    elif (k == 1) and (n == 1):
        k_prime_set = (0, 1)
        weights = (0.5, 0.5)
    elif (k == 1) and (n >= 2):
        k_prime_set = (0, 1)
        weights = (0.5, 1.0)
    elif (k == n) and (n >= 2):
        k_prime_set = (n-1, n)
        weights = (1.0, 0.5)
    elif k == n+1:
        k_prime_set = (n,)
        weights = (0.5,)
    elif 2 <= k <= n-1:
        k_prime_set = (k-1, k)
        weights = (1.0, 1.0)

    return weights, k_prime_set



class ZFieldPhaseFactorNodeRank3():
    def __init__(self, system_model, dt):
        self.z_fields = system_model.z_fields
        self.L = system_model.L
        self.dt = dt

        return None



    def calc_phase(self, r, j_r_m):
        dt = self.dt
        sigma_r_pos1_q, sigma_r_neg1_q = base_4_to_ising_pair(j_r_m)

        phase = 0.0
        for k_prime, weight in zip(self.k_prime_set, self.weights):
            hz = self.z_fields[r].eval(t=k_prime*dt)
            phase -= 0.5 * dt * weight * hz * (sigma_r_pos1_q - sigma_r_neg1_q)

        return phase
            


    def build(self, r, k, n):
        self.weights, self.k_prime_set = calc_weights_and_k_prime_set(k, n)
        L = self.L

        if (r == 0):
            tensor = np.zeros([1, 4, 4], dtype=np.complex128)
            for j_r_m in range(4):
                j_r_m_gt = j_r_m
                phase = self.calc_phase(r, j_r_m)
                tensor[0, j_r_m, j_r_m_gt] = np.exp(1.0j * phase)
        elif (1 <= r <= L-1):
            tensor = np.zeros([4, 4, 4], dtype=np.complex128)
            for j_r_m in range(4):
                j_r_m_lt = j_r_m
                j_r_m_gt = j_r_m
                phase = self.calc_phase(r, j_r_m)
                tensor[j_r_m_lt, j_r_m, j_r_m_gt] = np.exp(1.0j * phase)
        elif (r == L):
            tensor = np.zeros([4, 4, 1], dtype=np.complex128)
            for j_r_m in range(4):
                j_r_m_lt = j_r_m
                phase = self.calc_phase(r, j_r_m)
                tensor[j_r_m_lt, j_r_m, 0] = np.exp(1.0j * phase)
            
        node = tn.Node(tensor)

        return node



class ZZCouplerPhaseFactorNodeRank2():
    def __init__(self, system_model, dt):
        self.zz_couplers = system_model.zz_couplers
        self.dt = dt

        return None



    def calc_phase(self, r, j_r_m, j_rP1_m):
        dt = self.dt
        sigma_r_pos1_q, sigma_r_neg1_q = base_4_to_ising_pair(j_r_m)
        sigma_rP1_pos1_q, sigma_rP1_neg1_q = base_4_to_ising_pair(j_rP1_m)
        
        phase = 0.0
        for k_prime, weight in zip(self.k_prime_set, self.weights):
            Jzz = self.zz_couplers[r].eval(t=k_prime*dt)
            phase -= (0.5 * dt * weight * Jzz
                      * (sigma_r_pos1_q * sigma_rP1_pos1_q
                         - sigma_r_neg1_q * sigma_rP1_neg1_q))
        
        return phase
            


    def build(self, r, k, n):
        self.weights, self.k_prime_set = calc_weights_and_k_prime_set(k, n)

        tensor = np.zeros([4, 4], dtype=np.complex128)
        for j_r_m_gt in range(4):
            for j_rP1_m_lt in range(4):
                phase = self.calc_phase(r, j_r_m_gt, j_rP1_m_lt)
                tensor[j_r_m_gt, j_rP1_m_lt] = np.exp(1.0j * phase)
            
        node = tn.Node(tensor)

        return node



class ZFieldZZCouplerPhaseFactorMPS():
    def __init__(self, system_model, dt):
        self.z_field_phase_factor_node_rank_3_factory = \
            ZFieldPhaseFactorNodeRank3(system_model, dt)
        self.zz_coupler_phase_factor_node_rank_2_factory = \
            ZZCouplerPhaseFactorNodeRank2(system_model, dt)
        self.L = system_model.L

        return None



    def build(self, k, n):
        z_field_phase_factor_node_rank_3_factory = \
            self.z_field_phase_factor_node_rank_3_factory
        zz_coupler_phase_factor_node_rank_2_factory = \
            self.zz_coupler_phase_factor_node_rank_2_factory
        L = self.L
        
        mps_nodes = []
        for r in range(L):
            M_node = z_field_phase_factor_node_rank_3_factory.build(r, k, n)

            if r != 0:
                nodes_to_contract = [X_lt_node, M_node]
                network_struct = [(-1, 1), (1, -2, -3)]
                mps_node = tn.ncon(nodes_to_contract, network_struct)
            else:
                mps_node = M_node

            if r != L-1:
                X_node = \
                    zz_coupler_phase_factor_node_rank_2_factory.build(r, k, n)

                left_edges = (X_node[0],)
                right_edges = (X_node[1],)
                X_gt_node, X_lt_node, _ = \
                    tn.split_node(X_node, left_edges, right_edges)

                X_gt_node[-1] | X_lt_node[0]  # Break edge between nodes.

                nodes_to_contract = [mps_node, X_gt_node]
                network_struct = [(-1, -2, 1), (1, -3)]
                mps_node = tn.ncon(nodes_to_contract, network_struct)
            
            mps_nodes += [mps_node]

        return mps_nodes
