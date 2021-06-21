#!/usr/bin/env python
r"""This module contains classes that can be used to construct certain tensors 
that encode influence.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For creating multi-dimensional arrays to be used to construct tensor nodes
# and networks.
import numpy as np

# For creating tensor networks and performing contractions.
import tensornetwork as tn

# For putting MPO's in left-canonical form.
from sbc._svd import left_to_right_svd_sweep



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

class InfluenceNodeRank3():
    def __init__(self, total_two_pt_influence):
        self.total_two_pt_influence = total_two_pt_influence

        if total_two_pt_influence.alg == "yz-noise":
            self.max_m = lambda n: 3*n+3
        else:
            self.max_m = lambda n: n+1

        return None



    def build(self, m, n):
        self.total_two_pt_influence.set_m1_m2_n(m1=m, m2=m, n=n)

        shape = [4, 4, 1] if m < self.max_m(n) else [4, 4, 4]
        tensor = np.zeros(shape, dtype=np.complex128)

        for j_r_m in range(4):
            b_r_m = j_r_m
            b_r_mP1 = 0 if m < self.max_m(n) else j_r_m
            tensor[b_r_m, j_r_m, b_r_mP1] = \
                self.total_two_pt_influence.eval(j_r_m, j_r_m)

        node = tn.Node(tensor)

        return node



class InfluenceNodeRank4():
    def __init__(self, total_two_pt_influence):
        self.total_two_pt_influence = total_two_pt_influence
        K_tau = total_two_pt_influence.z_bath.pkl_part.K_tau

        if total_two_pt_influence.alg == "yz-noise":
            self.mu_m_tau = lambda m: max(0, m-3*K_tau+1)
        elif total_two_pt_influence.alg == "z-noise":
            self.mu_m_tau = lambda m: max(0, m-K_tau+1)

        return None



    def build(self, m1, m2, n):
        self.total_two_pt_influence.set_m1_m2_n(m1, m2, n)
        mu_m2_tau = self.mu_m_tau(m=m2)
        
        if m1 == mu_m2_tau:
            tensor = np.zeros([1, 4, 4, 4], dtype=np.complex128)
            for j_r_m1 in range(4):
                j_r_m1_prime = j_r_m1
                for b_r_m1P1 in range(4):
                    tensor[0, j_r_m1, j_r_m1_prime, b_r_m1P1] = \
                        self.total_two_pt_influence.eval(j_r_m1, b_r_m1P1)
        else:
            tensor = np.zeros([4, 4, 4, 4], dtype=np.complex128)
            for j_r_m1 in range(4):
                j_r_m1_prime = j_r_m1
                for b_r_m1 in range(4):
                    b_r_m1P1 = b_r_m1
                    tensor[b_r_m1, j_r_m1, j_r_m1_prime, b_r_m1P1] = \
                        self.total_two_pt_influence.eval(j_r_m1, b_r_m1P1)

        node = tn.Node(tensor)

        return node



class InfluenceMPO():
    def __init__(self, total_two_pt_influence):
        self.influence_node_rank_4_factory = \
            InfluenceNodeRank4(total_two_pt_influence)

        return None



    def build(self, m2, n):
        mu_m2_tau = self.influence_node_rank_4_factory.mu_m_tau(m=m2)
        mpo_nodes = []
        
        for m1 in range(mu_m2_tau, m2):
            mpo_node = self.influence_node_rank_4_factory.build(m1, m2, n)
            mpo_nodes.append(mpo_node)

        return mpo_nodes
