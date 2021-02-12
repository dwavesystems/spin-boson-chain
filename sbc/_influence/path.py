#!/usr/bin/env python
r"""This module contains a class that represents an influence path/functional.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For explicitly releasing memory.
import gc



# For calculating the total two-point influence function.
import sbc._influence.twopt

# For creating influence nodes and MPOs used to calculate the influence path.
from sbc._influence.tensorfactory import InfluenceNodeRank3
from sbc._influence.tensorfactory import InfluenceMPO

# For applying MPO's to MPS's.
from sbc._mpomps import _apply_mpo_to_mps

# For performing SVD truncation sweeps.
from sbc._svd import _left_to_right_svd_sweep_across_mps
from sbc._svd import _right_to_left_svd_sweep_across_mps



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

class Path():
    def __init__(self, r, system_model, bath_model, dt, trunc_params):
        total_two_point_influence = \
            sbc._influence.twopt.Total(r, system_model, bath_model, dt)
        self.influence_node_rank_3_factory = \
            InfluenceNodeRank3(total_two_point_influence)
        self.influence_mpo_factory = \
            InfluenceMPO(total_two_point_influence)
        self.trunc_params = trunc_params

        self.n = 0
        K_tau = total_two_point_influence.z_bath.K_tau

        if bath_model.y_spectral_densities != None:
            self.mu_m_tau = lambda m: max(0, m-3*K_tau+1)
            self.max_m2_in_first_iteration_procedure = lambda n: 3*n-2
            self.max_m2_in_second_iteration_procedure = lambda n: 3*n+2
        else:
            self.mu_m_tau = lambda m: max(0, m-K_tau+1)
            self.max_m2_in_first_iteration_procedure = lambda n: n-2
            self.max_m2_in_second_iteration_procedure = lambda n: n

        M_r_1_0_I = self.influence_node_rank_3_factory.build(0, 1)
        self.Xi_I_1_1_nodes = []
        self.Xi_I_1_2_nodes = [M_r_1_0_I]

        return None



    def evolve(self, num_n_steps):
        self.m2 = max(0, self.max_m2_in_first_iteration_procedure(self.n)+1)
        self.n += num_n_steps

        while self.m2 <= self.max_m2_in_first_iteration_procedure(self.n):
            self.m2_step()
            gc.collect()

        self.Xi_I_dashv_1_nodes = []
        self.Xi_I_dashv_2_nodes = self.Xi_I_1_2_nodes[:]  # Shallow copy.
        while self.m2 <= self.max_m2_in_second_iteration_procedure(self.n):
            self.m2_step()
            gc.collect()

        self.Xi_I_dashv_nodes = \
            self.Xi_I_dashv_1_nodes + self.Xi_I_dashv_2_nodes

        return None



    def m2_step(self):
        m2 = self.m2
        n = self.n
        
        if m2 <= self.max_m2_in_first_iteration_procedure(n):
            mps_nodes = self.Xi_I_1_2_nodes
        else:
            mps_nodes = self.Xi_I_dashv_2_nodes
        
        mpo_nodes = self.influence_mpo_factory.build(m2+1, n)
        mps_nodes = _apply_mpo_to_mps(mpo_nodes, mps_nodes)
        node = self.influence_node_rank_3_factory.build(m2+1, n)
        mps_nodes.append(node)

        _left_to_right_svd_sweep_across_mps(mps_nodes, self.trunc_params)
        _right_to_left_svd_sweep_across_mps(mps_nodes, self.trunc_params)

        if m2 <= self.max_m2_in_first_iteration_procedure(n):
            if self.mu_m_tau(m=m2+2) >= 1:
                self.Xi_I_1_1_nodes.append(mps_nodes.pop(0))
            self.Xi_I_1_2_nodes = mps_nodes
        else:
            if self.mu_m_tau(m=m2+2) >= 1:
                self.Xi_I_dashv_1_nodes.append(mps_nodes.pop(0))
            self.Xi_I_dashv_2_nodes = mps_nodes

        self.m2 += 1

        return None
        
