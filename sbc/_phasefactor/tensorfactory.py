#!/usr/bin/env python
r"""This module contains classes that can be used to construct certain tensors 
that encode the phase factor effects of z-fields and zz-couplers.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For creating multi-dimensional arrays to be used to construct tensor nodes
# and networks. Also to perform SVDs and matrix multiplication.
import numpy as np

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# For converting base-4 variables to Ising spin pairs.
import sbc._base4



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Development"



##################################
## Define classes and functions ##
##################################

def calc_weights_and_k_prime_set(k, n):
    # DM: Detailed manuscript.
    
    # For fixed k and n calc_weights_and_k_prime_set determines the set of
    # w_n_k_prime values that are required to evaluate Eq. (183) of DM. See
    # Eq. (58) of DM for the expression for the weights.
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



class ZFieldPhaseFactorNodeRank2():
    r"""This is a 'factory' class that builds instances of the node given by
    Eq. (188) of the detailed manuscript (DM). For context read Sec. 4.7 of 
    DM."""
    def __init__(self, system_model, dt):
        self.z_fields = system_model.z_fields
        self.dt = dt  # time step size.

        return None



    def calc_phase(self, r, j_r_m):
        # DM: Detailed manuscript.
        # This method calculates Eq. (183) of DM.
        
        dt = self.dt  # Time step size.

        # See Secs. 4.1 and 4.2 for context on base-4 variables.
        base_4_to_ising_pair = sbc._base4.base_4_to_ising_pair
        sigma_r_pos1_q, sigma_r_neg1_q = base_4_to_ising_pair(j_r_m)

        # self.k_prime_set is set in call to the 'build' method below.
        phase = 0.0
        for k_prime, weight in zip(self.k_prime_set, self.weights):
            hz = self.z_fields[r].eval(t=k_prime*dt)
            phase -= 0.5 * dt * weight * hz * (sigma_r_pos1_q - sigma_r_neg1_q)

        return phase
            


    def build(self, r, k, n):
        # DM: Detailed manuscript.
        # Construct node given by Eq. (188) of DM.
        
        self.weights, self.k_prime_set = calc_weights_and_k_prime_set(k, n)

        tensor = np.zeros([4, 4], dtype=np.complex128)
        for j_r_m in range(4):
            j_r_m_prime = j_r_m
            phase = self.calc_phase(r, j_r_m)
            tensor[j_r_m, j_r_m_prime] = np.exp(1.0j * phase)
            
        node = tn.Node(tensor)

        return node



class ZZCouplerPhaseFactorNodeRank2():
    r"""This is a 'factory' class that builds instances of the node given by
    Eq. (187) of the detailed manuscript (DM). For context read Sec. 4.7 of 
    DM."""
    def __init__(self, system_model, dt):
        self.zz_couplers = system_model.zz_couplers
        self.dt = dt  # Time step size.

        return None



    def calc_phase(self, r, j_r_m, j_rP1_m):
        # DM: Detailed manuscript.
        # This method calculates Eq. (184) of DM.
        
        dt = self.dt  # Time step size.

        # See Secs. 4.1 and 4.2 for context on base-4 variables.
        base_4_to_ising_pair = sbc._base4.base_4_to_ising_pair
        sigma_r_pos1_q, sigma_r_neg1_q = base_4_to_ising_pair(j_r_m)
        sigma_rP1_pos1_q, sigma_rP1_neg1_q = base_4_to_ising_pair(j_rP1_m)

        # self.k_prime_set is set in call to the 'build' method below.
        phase = 0.0
        for k_prime, weight in zip(self.k_prime_set, self.weights):
            Jzz = self.zz_couplers[r].eval(t=k_prime*dt)
            phase -= (0.5 * dt * weight * Jzz
                      * (sigma_r_pos1_q * sigma_rP1_pos1_q
                         - sigma_r_neg1_q * sigma_rP1_neg1_q))
        
        return phase
            


    def build(self, r, k, n):
        # DM: Detailed manuscript.
        # Construct node given by Eq. (187) of DM.
        
        self.weights, self.k_prime_set = calc_weights_and_k_prime_set(k, n)

        tensor = np.zeros([4, 4], dtype=np.complex128)
        for j_r_m in range(4):
            for j_rP1_m in range(4):
                phase = self.calc_phase(r, j_r_m, j_rP1_m)
                tensor[j_r_m, j_rP1_m] = np.exp(1.0j * phase)

        node = tn.Node(tensor)
        
        return node
