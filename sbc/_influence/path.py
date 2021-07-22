#!/usr/bin/env python
r"""This module contains a class that represents an influence path/functional.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For explicitly releasing memory.
import gc



# For general array handling.
import numpy as np

# For creating tensor networks.
import tensornetwork as tn



# For calculating the total two-point influence function.
import sbc._influence.twopt

# For creating influence nodes and MPOs used to calculate the influence path.
import sbc._influence.tensorfactory

# For applying MPO's to MPS's.
import sbc._mpomps

# For shifting orthogonal centers of MPS's.
import sbc._svd



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

class PathPklPart():
    def __init__(self, compress_params, alg):
        self.compress_params = compress_params
        self.alg = alg

        self.n = 0
        self.m2 = 0
        
        # For caching purposes.
        self.Xi_I_dashv_1_nodes = None
        self.Xi_I_dashv_2_nodes = None
        self.Xi_I_dashv_nodes = None
        self.Xi_I_1_1_nodes = None
        self.Xi_I_1_2_nodes = None

        return None



class Path():
    def __init__(self,
                 r,
                 system_model,
                 bath_model,
                 dt,
                 compress_params,
                 pkl_parts=None):
        total_two_point_influence = sbc._influence.twopt.Total(r,
                                                               system_model,
                                                               bath_model,
                                                               dt,
                                                               pkl_parts)

        InfluenceNodeRank3 = sbc._influence.tensorfactory.InfluenceNodeRank3
        InfluenceMPO = sbc._influence.tensorfactory.InfluenceMPO
        self.influence_node_rank_3_factory = \
            InfluenceNodeRank3(total_two_point_influence)
        self.influence_mpo_factory = \
            InfluenceMPO(total_two_point_influence)

        K_tau = total_two_point_influence.z_bath.pkl_part.K_tau

        if bath_model.y_spectral_densities != None:
            self.mu_m_tau = lambda m: max(0, m-3*K_tau+1)
            self.max_m2_in_first_iteration_procedure = lambda n: 3*n-2
            self.max_m2_in_second_iteration_procedure = lambda n: 3*n+2
        else:
            self.mu_m_tau = lambda m: max(0, m-K_tau+1)
            self.max_m2_in_first_iteration_procedure = lambda n: n-2
            self.max_m2_in_second_iteration_procedure = lambda n: n

        if pkl_parts is None:
            alg = total_two_point_influence.alg
            self.pkl_part = PathPklPart(compress_params, alg)
            M_r_1_0_I = self.influence_node_rank_3_factory.build(0, 1)
            self.pkl_part.Xi_I_1_1_nodes = []
            self.pkl_part.Xi_I_1_2_nodes = [M_r_1_0_I]
        else:
            self.pkl_part = pkl_parts["influence_path"]

        return None



    def reset_evolve_procedure(self, num_n_steps, k, forced_gc):
        n = self.pkl_part.n
        m2 = max(0, self.max_m2_in_first_iteration_procedure(n)+1)
        n += num_n_steps

        self.pkl_part.n = n
        self.pkl_part.m2 = m2

        while self.first_m2_step_seq_in_reset_evolve_procedure_not_finished(k):
            self.m2_step()
            if forced_gc:
                gc.collect()

        if self.pkl_part.m2 <= self.max_m2_in_first_iteration_procedure(n):
            return None

        self.pkl_part.Xi_I_dashv_1_nodes = []
        self.pkl_part.Xi_I_dashv_2_nodes = self.pkl_part.Xi_I_1_2_nodes[:]

        while self.second_m2_step_seq_in_reset_evolve_procedure_not_finished(k):
            self.m2_step()
            if forced_gc:
                gc.collect()

        return None



    def first_m2_step_seq_in_reset_evolve_procedure_not_finished(self, k):
        m2_limit = self.max_m2_in_first_iteration_procedure(self.pkl_part.n)
        
        if (k != -1) and (self.pkl_part.alg == "yz-noise"):
            target_num_Xi_I_1_1_nodes = 3
        else:
            target_num_Xi_I_1_1_nodes = 1

        num_Xi_I_1_1_nodes = len(self.pkl_part.Xi_I_1_1_nodes)

        condition_1 = self.pkl_part.m2 <= m2_limit
        condition_2 = num_Xi_I_1_1_nodes < target_num_Xi_I_1_1_nodes

        return condition_1 and condition_2



    def second_m2_step_seq_in_reset_evolve_procedure_not_finished(self, k):
        m2_limit = self.max_m2_in_second_iteration_procedure(self.pkl_part.n)
        
        if (k != -1) and (self.pkl_part.alg == "yz-noise"):
            target_num_Xi_I_dashv_1_nodes = 3
        else:
            target_num_Xi_I_dashv_1_nodes = 1

        num_Xi_I_dashv_1_nodes = len(self.pkl_part.Xi_I_dashv_1_nodes)

        condition_1 = self.pkl_part.m2 <= m2_limit
        condition_2 = num_Xi_I_dashv_1_nodes < target_num_Xi_I_dashv_1_nodes

        return condition_1 and condition_2


    def k_step(self, forced_gc):
        n = self.pkl_part.n
        max_m2_in_first_iteration_procedure_plus_1 = \
            self.max_m2_in_first_iteration_procedure(n)+1
        max_m2_in_second_iteration_procedure = \
            self.max_m2_in_second_iteration_procedure(n)

        if ((self.pkl_part.alg == "z-noise")
            or (self.pkl_part.m2 == max_m2_in_second_iteration_procedure)):
            num_m2_steps = 1
        else:
            num_m2_steps = 3

        for _ in range(num_m2_steps):
            self.m2_step()
            if self.pkl_part.m2 == max_m2_in_first_iteration_procedure_plus_1:
                self.pkl_part.Xi_I_dashv_1_nodes = []
                self.pkl_part.Xi_I_dashv_2_nodes = \
                    self.pkl_part.Xi_I_1_2_nodes[:]
            if forced_gc:
                gc.collect()

        return None


    
    def m2_step(self):
        m2 = self.pkl_part.m2
        n = self.pkl_part.n

        if m2 <= self.max_m2_in_first_iteration_procedure(n):
            mps_nodes = self.pkl_part.Xi_I_1_2_nodes
        else:
            mps_nodes = self.pkl_part.Xi_I_dashv_2_nodes

        node = tn.Node(np.ones([1, 4, 1]))
        mps_nodes.append(node)
        kwargs = {"nodes": mps_nodes,
                  "current_orthogonal_center_idx": len(mps_nodes) - 2,
                  "compress_params": None}
        sbc._svd.shift_orthogonal_center_to_the_right(**kwargs)

        kwargs = {"mpo_nodes": self.influence_mpo_factory.build(m2+1, n),
                  "mps_nodes": mps_nodes,
                  "compress_params": self.pkl_part.compress_params}
        sbc._mpomps.apply_finite_mpo_to_finite_mps_and_compress(**kwargs)

        if m2 <= self.max_m2_in_first_iteration_procedure(n):
            if self.mu_m_tau(m=m2+2) >= 1:
                self.pkl_part.Xi_I_1_1_nodes.append(mps_nodes.pop(0))
            self.pkl_part.Xi_I_1_2_nodes = mps_nodes
        else:
            if self.mu_m_tau(m=m2+2) >= 1:
                self.pkl_part.Xi_I_dashv_1_nodes.append(mps_nodes.pop(0))
            self.pkl_part.Xi_I_dashv_2_nodes = mps_nodes
            self.pkl_part.Xi_I_dashv_nodes = \
                (self.pkl_part.Xi_I_dashv_1_nodes
                 + self.pkl_part.Xi_I_dashv_2_nodes)

        self.pkl_part.m2 += 1

        return None
        
