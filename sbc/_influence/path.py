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
import sbc._qr



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

class PathPklPart():
    def __init__(self, compress_params, alg):
        # DM: Detailed manuscript.

        # 'Pickle parts' can be saved to file in case of a crash and then
        # subsequently recovered in a future run. See docs of method
        # sbc.state.recover_and_resume for background information on pickles and
        # simulation recovery.
        
        self.compress_params = compress_params
        self.alg = alg  # yz- or z-noise algorithm?

        self.n = 0  # Time step index.
        self.m2 = 0
        
        # For caching purposes.
        self.Xi_I_dashv_1_nodes = None  # Introduced in Eq. (143) of DM.
        self.Xi_I_dashv_2_nodes = None  # Introduced in Eq. (143) of DM.
        self.Xi_I_dashv_nodes = None  # Introduced in Eq. (143) of DM.
        self.Xi_I_1_1_nodes = None  # Introduced in Eq. (126) of DM.
        self.Xi_I_1_2_nodes = None  # Introduced in Eq. (126) of DM.

        return None



class Path():
    r"""This class represents a local influence path/functional, given by
    Eq. (103) of the detailed manuscript (DM). For context read Sec. 4.3 and 4.4
    of DM."""
    def __init__(self,
                 r,  # Site index.
                 system_model,
                 bath_model,
                 dt,  # Time step size.
                 compress_params,
                 pkl_parts=None):  # Used in loading/creating backups.
        # DM: Detailed manuscript.

        # 'Pickle parts' can be saved to file in case of a crash and then
        # subsequently recovered in a future run. See docs of method
        # sbc.state.recover_and_resume for background information on pickles and
        # simulation recovery.
        
        # total_two_point_influence represents the quantity in Eq. (109) of DM.
        total_two_point_influence = sbc._influence.twopt.Total(r,
                                                               system_model,
                                                               bath_model,
                                                               dt,
                                                               pkl_parts)

        # This class generates the M-nodes given by Eq. (118) of DM.
        InfluenceNodeRank3 = sbc._influence.tensorfactory.InfluenceNodeRank3

        # This class generates the W-nodes given by Eqs. (120)-(123) of DM.
        InfluenceMPO = sbc._influence.tensorfactory.InfluenceMPO
        
        influence_node_rank_3_factory = \
            InfluenceNodeRank3(total_two_point_influence)
        self.influence_mpo_factory = \
            InfluenceMPO(total_two_point_influence)

        # K_tau is given by Eq. (87) of DM.
        K_tau = total_two_point_influence.z_bath.pkl_part.K_tau

        # dm is given by Eq. (89) of DM.
        dm = 3 if bath_model.y_spectral_densities is not None else 1

        # mu_m_tau is given by Eq. (108) of DM.
        self.mu_m_tau = lambda m: max(0, m-K_tau*dm+1)

        # The 'first iteration procedure' involves executing Eqs. (130)-(137) of
        # DM, whereas the 'second iteration procedure' involves executing
        # Eqs. (146)-(154) of DM.
        self.max_m2_in_first_iteration_procedure = lambda n: n*dm-2
        self.max_m2_in_second_iteration_procedure = lambda n: (n+1)*dm-1

        if pkl_parts is None:  # Create pickle part from scratch.
            alg = total_two_point_influence.alg  # yz- or z-noise algorithm.
            self.pkl_part = PathPklPart(compress_params, alg)

            # M_r_1_0_I is given by Eq. (118) of DM with m2=0 and n=1.
            M_r_1_0_I = influence_node_rank_3_factory.build(m=0, n=1)

            # The quantities below are introduced in Eq. (126) of DM.
            self.pkl_part.Xi_I_1_1_nodes = []
            self.pkl_part.Xi_I_1_2_nodes = [M_r_1_0_I]
        else:  # Reload pickle part from backup.
            self.pkl_part = pkl_parts["influence_path"]

        return None



    def reset_evolve_procedure(self, num_n_steps, k, forced_gc):
        # DM: Detailed manuscript.

        # The 'evolve procedure' refers to the step-evolution procedure
        # implemented for the sbc.state.SystemState class, which represents
        # the system state. An 'evolution step' consists of a sequence of
        # 'k-steps', where in each k-step, a MPO is constructed which is applied
        # to the MPS representing the system state [given by one of
        # Eqs. (215)-(217) of DM depending on scenario]. Each of these MPO's
        # requires a set of influence nodes taken from the MPS's representing
        # the local influence paths. Details on the MPO construction procedure
        # are given in Sec. 4.8 of DM. Instances of k-steps are given by
        # Eqs. (219), (220), (221), (223), and (224) of DM. The current method
        # here essentially calculates the required set of influence nodes to
        # perform the first k-step of the current evolution step.

        # The 'first iteration procedure' involves executing Eqs. (130)-(137) of
        # DM, whereas the 'second iteration procedure' involves executing
        # Eqs. (146)-(154) of DM.
        n = self.pkl_part.n
        m2 = max(0, self.max_m2_in_first_iteration_procedure(n)+1)
        n += num_n_steps

        self.pkl_part.n = n
        self.pkl_part.m2 = m2

        while self.first_m2_step_seq_in_reset_evolve_procedure_not_finished(k):
            self.m2_step()
            if forced_gc:
                gc.collect()  # Enforce garbage collection.

        if self.pkl_part.m2 <= self.max_m2_in_first_iteration_procedure(n):
            return None

        # At this point the first iteration procedure as finished, so the
        # second procedure is initiated. The following code block is essentially
        # Eq. (146) of DM.
        self.pkl_part.Xi_I_dashv_1_nodes = []
        self.pkl_part.Xi_I_dashv_2_nodes = self.pkl_part.Xi_I_1_2_nodes[:]

        while self.second_m2_step_seq_in_reset_evolve_procedure_not_finished(k):
            self.m2_step()
            if forced_gc:
                gc.collect()  # Enforce garbage collection.

        return None



    def first_m2_step_seq_in_reset_evolve_procedure_not_finished(self, k):
        # See comments in method reset_evolve_procedure for context.

        m2_limit = self.max_m2_in_first_iteration_procedure(self.pkl_part.n)
        
        if (k != -1) and (self.pkl_part.alg == "yz-noise"):
            target_num_Xi_I_1_1_nodes = 3
        else:
            target_num_Xi_I_1_1_nodes = 1

        num_Xi_I_1_1_nodes = len(self.pkl_part.Xi_I_1_1_nodes)

        # If target_num_Xi_I_1_1_nodes of the Xi_I_1_1 nodes have been obtained
        # before iterating through all the m2 steps of the 'first procedure',
        # then the 'reset evolve procedure' is finished and we can proceed to
        # executing our first 'k-step'. Otherwise, we iterate through all the m2
        # steps of the first procedure.
        
        condition_1 = self.pkl_part.m2 <= m2_limit
        condition_2 = num_Xi_I_1_1_nodes < target_num_Xi_I_1_1_nodes

        return condition_1 and condition_2



    def second_m2_step_seq_in_reset_evolve_procedure_not_finished(self, k):
        # See comments in method reset_evolve_procedure for context.
        
        m2_limit = self.max_m2_in_second_iteration_procedure(self.pkl_part.n)
        
        if (k != -1) and (self.pkl_part.alg == "yz-noise"):
            target_num_Xi_I_dashv_1_nodes = 3
        else:
            target_num_Xi_I_dashv_1_nodes = 1

        num_Xi_I_dashv_1_nodes = len(self.pkl_part.Xi_I_dashv_1_nodes)

        # If target_num_Xi_I_dashv_1_nodes of the Xi_I_1_1 nodes have been
        # obtained before iterating through all the m2 steps of the 'second
        # procedure', then the 'reset evolve procedure' is finished and we can
        # proceed to executing our first 'k-step'. Otherwise, we iterate through
        # all the m2 steps of the secon procedure.

        condition_1 = self.pkl_part.m2 <= m2_limit
        condition_2 = num_Xi_I_dashv_1_nodes < target_num_Xi_I_dashv_1_nodes

        return condition_1 and condition_2


    def k_step(self, forced_gc):
        # DM: Detailed manuscript.

        # An 'evolution step', wherein the system state is evolved, consists of
        # a sequence of 'k-steps', where in each k-step, a MPO is constructed
        # which is applied to the MPS representing the system state [given by
        # one of Eqs. (215)-(217) of DM depending on scenario]. The role that
        # the influence paths play in a single k-step is that they construct the
        # set of influence nodes taken from the MPS's representing the local
        # influence paths that are required to construct the aforementioned MPO
        # for that k-step. Details on the MPO construction procedure are given
        # in Sec. 4.8 of DM. Instances of k-steps are given by Eqs. (219),
        # (220), (221), (223), and (224) of DM. 

        # The 'first iteration procedure' involves executing Eqs. (130)-(137) of
        # DM, whereas the 'second iteration procedure' involves executing
        # Eqs. (146)-(154) of DM.
        n = self.pkl_part.n
        max_m2_in_first_iteration_procedure_plus_1 = \
            self.max_m2_in_first_iteration_procedure(n)+1
        max_m2_in_second_iteration_procedure = \
            self.max_m2_in_second_iteration_procedure(n)

        if ((self.pkl_part.alg == "z-noise")
            or (self.pkl_part.m2 == max_m2_in_second_iteration_procedure)):
            num_m2_steps = 1  # The number of m2-steps taken in current k-step.
        else:
            num_m2_steps = 3  # The number of m2-steps taken in current k-step.

        for _ in range(num_m2_steps):
            self.m2_step()
            if self.pkl_part.m2 == max_m2_in_first_iteration_procedure_plus_1:
                # The following code block is essentially Eq. (146) of DM.
                self.pkl_part.Xi_I_dashv_1_nodes = []
                self.pkl_part.Xi_I_dashv_2_nodes = \
                    self.pkl_part.Xi_I_1_2_nodes[:]
            if forced_gc:
                gc.collect()  # Enforce garbage collection.

        return None


    
    def m2_step(self):
        # DM: Detailed manuscript.

        # This method either performs a single 'm2-step' in the 'first
        # procedure' which involves executing Eqs. (130)-(137) of DM, or the
        # 'second procedure' which involves executing Eqs. (146)-(154) of DM.
        
        m2 = self.pkl_part.m2
        n = self.pkl_part.n

        if m2 <= self.max_m2_in_first_iteration_procedure(n):
            mps_nodes = self.pkl_part.Xi_I_1_2_nodes
        else:
            mps_nodes = self.pkl_part.Xi_I_dashv_2_nodes

        # The following code block implements Eq. (131) if in first procedure,
        # and Eq. (147) if in second procedure.
        node = tn.Node(np.ones([1, 4, 1]))
        mps_nodes.append(node)
        kwargs = {"nodes": mps_nodes,
                  "current_orthogonal_center_idx": len(mps_nodes) - 2}
        sbc._qr.shift_orthogonal_center_to_the_right(**kwargs)
        mps_nodes[-1] /= tn.norm(mps_nodes[-1])

        # Perform MPS compression for computational efficiency.
        kwargs = {"mpo_nodes": self.influence_mpo_factory.build(m2+1, n),
                  "mps_nodes": mps_nodes,
                  "compress_params": self.pkl_part.compress_params}
        sbc._mpomps.apply_finite_mpo_to_finite_mps_and_compress(**kwargs)

        if m2 <= self.max_m2_in_first_iteration_procedure(n):
            if self.mu_m_tau(m=m2+2) >= 1:
                # This is essentially Eq. (137) of DM.
                self.pkl_part.Xi_I_1_1_nodes.append(mps_nodes.pop(0))
            # This is essentially Eq. (136) of DM.
            self.pkl_part.Xi_I_1_2_nodes = mps_nodes
        else:
            if self.mu_m_tau(m=m2+2) >= 1:
                # This is essentially Eq. (154) of DM.
                self.pkl_part.Xi_I_dashv_1_nodes.append(mps_nodes.pop(0))
            # This is essentially Eq. (153) of DM.
            self.pkl_part.Xi_I_dashv_2_nodes = mps_nodes
            # This is essentially Eq. (156) of DM.
            self.pkl_part.Xi_I_dashv_nodes = \
                (self.pkl_part.Xi_I_dashv_1_nodes
                 + self.pkl_part.Xi_I_dashv_2_nodes)

        self.pkl_part.m2 += 1

        return None
        
