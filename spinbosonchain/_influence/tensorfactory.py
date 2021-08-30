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

class InfluenceNodeRank3():
    r"""This is a 'factory' class that builds instances of the node given by
    Eq. (118) of the detailed manuscript (DM). For context read Sec. 4.3 of 
    DM."""
    def __init__(self, total_two_pt_influence):
        # Object below represents quantity in Eq. (109) of DM.
        self.total_two_pt_influence = total_two_pt_influence

        # dm is given by Eq. (89) of DM.
        dm = 3 if total_two_pt_influence.alg == "yz-noise" else 1

        # Maximum possible m for a given time step index n.
        self.max_m = lambda n: (n+1)*dm

        return None



    def build(self, m, n):
        # DM: Detailed manuscript.
        # Construct node given by Eq. (118) of DM.

        # Object below represents quantity in Eq. (109) of DM.
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
    r"""This is a 'factory' class that builds instances of the nodes given by
    Eq. (120)-(123) of the detailed manuscript (DM). For context read Sec. 4.3 
    of DM."""
    def __init__(self, total_two_pt_influence):
        # Object below represents quantity in Eq. (109) of DM.
        self.total_two_pt_influence = total_two_pt_influence
        
        K_tau = total_two_pt_influence.z_bath.pkl_part.K_tau  # Eq. (87) of DM.

        # dm is given by Eq. (89) of DM.
        dm = 3 if total_two_pt_influence.alg == "yz-noise" else 1

        # mu_m_tau is given by Eq. (108) of DM.
        self.mu_m_tau = lambda m: max(0, m-K_tau*dm+1)

        # Maximum possible m2 for a given time step index n.
        self.max_m2 = lambda n: (n+1)*dm

        return None



    def build(self, m1, m2, n):
        # DM: Detailed manuscript.
        # Construct one of nodes given by Eqs. (120)-(123) of DM.

        # Object below represents quantity in Eq. (109) of DM.
        self.total_two_pt_influence.set_m1_m2_n(m1, m2, n)
        
        mu_m2_tau = self.mu_m_tau(m=m2)  # Eq. (108) of DM.
        
        if m1 == mu_m2_tau:  # Eq. (120) of DM.
            tensor = np.zeros([1, 4, 4, 4], dtype=np.complex128)
            for j_r_m1 in range(4):
                j_r_m1_prime = j_r_m1
                for b_r_m1P1 in range(4):
                    tensor[0, j_r_m1, j_r_m1_prime, b_r_m1P1] = \
                        self.total_two_pt_influence.eval(j_r_m1, b_r_m1P1)
        elif m1 == m2:  # Eqs. (122) or Eq. (123) of DM.
            shape = [4, 4, 4, 1] if m2 < self.max_m2(n) else [4, 4, 4, 4]
            tensor = np.zeros(shape, dtype=np.complex128)
            for j_r_m1 in range(4):
                j_r_m1_prime = j_r_m1
                b_r_m1 = j_r_m1
                b_r_m1P1 = 0 if m2 < self.max_m2(n) else j_r_m1
                tensor[b_r_m1, j_r_m1, j_r_m1_prime, b_r_m1P1] = \
                    self.total_two_pt_influence.eval(j_r_m1, j_r_m1)
        else:  # Eq. (121) of DM.
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
    r"""This is a 'factory' class that builds instances of the MPO given by
    Eq. (124) of the detailed manuscript (DM). For context read Sec. 4.3 
    of DM."""
    def __init__(self, total_two_pt_influence):
        # Builds the MPO nodes given by Eqs. (120)-(123) of DM.
        self.influence_node_rank_4_factory = \
            InfluenceNodeRank4(total_two_pt_influence)

        return None



    def build(self, m2, n):
        # DM: Detailed manuscript.
        # Construct MPO given by Eq. (124) of DM.

        # mu_m2_tau is given by Eq. (108) of Dm.
        mu_m2_tau = self.influence_node_rank_4_factory.mu_m_tau(m=m2)
        mpo_nodes = []

        # Build MPO nodes given by Eqs. (120)-(123) of DM.
        for m1 in range(mu_m2_tau, m2+1):
            mpo_node = self.influence_node_rank_4_factory.build(m1, m2, n)
            mpo_nodes.append(mpo_node)

        return mpo_nodes
