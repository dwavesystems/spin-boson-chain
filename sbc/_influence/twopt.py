#!/usr/bin/env python
r"""This module contains classes representing two-point influence functions that
occur in our QUAPI-TN approach.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# Import a few math functions.
import math
import cmath



# Import class representing time-dependent scalar model parameters.
import sbc.scalar

# For evaluating eta-functions.
import sbc._influence.eta

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
__status__ = "Non-Production"



##################################
## Define classes and functions ##
##################################

class BathPklPart():
    def __init__(self, r, bath_model, dt, spin_basis):
        self.r = r
        self.dt = dt
        self.spin_basis = spin_basis

        tau = bath_model.memory
        self.K_tau = max(0, math.ceil((tau - 7.0*dt/4.0) / dt)) + 3

        eta = sbc._influence.eta.Eta(r, bath_model, dt, spin_basis)

        # For caching purposes.
        self.calc_eta_caches(eta)
        self.coupling_energy_scale_prod_cache = None
        self.eta_selection_cache = None
        self.q1 = None
        self.q2 = None
        self.n = None

        return None



    def calc_eta_caches(self, eta):
        K_tau = self.K_tau

        # Implementing Eq. (588) of the DM.
        uf = K_tau - 2
        self.eta_cache_1 = [0.0j]*(uf+1)
        for u in range(0, uf+1):
            l1 = 2*u + 2
            l2 = 0
            n = u + 1
            self.eta_cache_1[u] = eta.eval(l1, l2, n)

        # Implementing Eq. (589) of the DM.
        uf = K_tau - 2
        self.eta_cache_2 = [0.0j]*(uf+1)
        for u in range(0, uf+1):
            l1 = 2*u + 3
            l2 = 0
            n = u + 1
            self.eta_cache_2[u] = eta.eval(l1, l2, n)

        # Implementing Eq. (590) of the DM.
        uf = K_tau - 1
        self.eta_cache_3 = [0.0j]*(uf+1)
        for u in range(0, uf+1):
            l1 = 2*u + 2
            l2 = 1
            n = u + 1
            self.eta_cache_3[u] = eta.eval(l1, l2, n)

        # Implementing Eq. (591) of the DM.
        uf = 2*K_tau - 1
        self.eta_cache_4 = [0.0j]*(uf+1)
        for u in range(0, uf+1):
            l1 = 2*K_tau + 3
            l2 = u + 4
            n = K_tau + 2
            self.eta_cache_4[u] = eta.eval(l1, l2, n)

        # Implementing Eq. (592) of the DM.
        uf = 2*K_tau - 1
        self.eta_cache_5 = [0.0j]*(uf+1)
        for u in range(0, uf+1):
            l1 = 2*K_tau + 2
            l2 = u + 3
            n = K_tau + 1
            self.eta_cache_5[u] = eta.eval(l1, l2, n)

        # Implementing Eq. (593) of the DM.
        uf = 2*K_tau - 1
        self.eta_cache_6 = [0.0j]*(uf+1)
        for u in range(0, uf+1):
            l1 = 2*K_tau + 1
            l2 = u + 2
            n = K_tau
            self.eta_cache_6[u] = eta.eval(l1, l2, n)

        return None



class Bath():
    def __init__(self, r, bath_model, dt, spin_basis, pkl_part=None):
        if pkl_part is None:
            self.pkl_part = BathPklPart(r, bath_model, dt, spin_basis)
        else:
            self.pkl_part = pkl_part

        spin_basis = self.pkl_part.spin_basis
        self.set_coupling_energy_scales_and_l_idx_pairs_selector(bath_model,
                                                                 spin_basis)

        self.set_q1_q2_n(0, 0, 1)

        return None



    def set_coupling_energy_scales_and_l_idx_pairs_selector(self,
                                                            bath_model,
                                                            spin_basis):
        r = self.pkl_part.r
        
        if spin_basis == "y":
            y_coupling_energy_scales = bath_model.y_coupling_energy_scales
            if y_coupling_energy_scales == None:
                self.coupling_energy_scale = sbc.scalar.Scalar(0.0)
            else:
                self.coupling_energy_scale = y_coupling_energy_scales[r]
            self.l_idx_pairs_selector = self.l_idx_pairs_selector_for_y_noise
        elif spin_basis == "z":
            z_coupling_energy_scales = bath_model.z_coupling_energy_scales
            if z_coupling_energy_scales == None:
                self.coupling_energy_scale = sbc.scalar.Scalar(0.0)
            else:
                self.coupling_energy_scale = z_coupling_energy_scales[r]
            self.l_idx_pairs_selector = self.l_idx_pairs_selector_for_z_noise

        return None



    def l_idx_pairs_selector_for_y_noise(self, n, q1, q2):
        l_idx_pairs = ((q1, q2),)

        return l_idx_pairs



    def l_idx_pairs_selector_for_z_noise(self, n, q1, q2):
        if q1 == q2 == 0:
            l_idx_pairs = ((0, 0),)
        elif (q1 == 0) and (1 <= q2 <= n):
            l_idx_pairs = ((0, 2*q2-1), (0, 2*q2))
        elif (q1 == 0) and (q2 == n+1):
            l_idx_pairs = ((0, 2*n+1),)
        elif 1 <= q1 < q2 <= n:
            l_idx_pairs = ((2*q1-1, 2*q2-1), (2*q1, 2*q2-1),
                           (2*q1-1, 2*q2), (2*q1, 2*q2))
        elif 1 <= q1 == q2 <= n:
            l_idx_pairs = ((2*q1-1, 2*q2-1), (2*q1-1, 2*q2), (2*q1, 2*q2))
        elif (1 <= q1 <= n) and (q2 == n+1):
            l_idx_pairs = ((2*q1-1, 2*n+1), (2*q1, 2*n+1))
        elif q1 == q2 == n+1:
            l_idx_pairs = ((2*n+1, 2*n+1),)

        return l_idx_pairs



    def set_q1_q2_n(self, q1, q2, n):
        self.pkl_part.q1 = q1
        self.pkl_part.q2 = q2
        self.pkl_part.n = n

        self.pkl_part.coupling_energy_scale_prod_cache = []
        self.pkl_part.eta_selection_cache = []
        
        l1_l2_pairs = self.l_idx_pairs_selector(n, q1, q2)
        
        for l1, l2 in l1_l2_pairs:
            t1 = (l1//2) * self.pkl_part.dt
            t2 = (l2//2) * self.pkl_part.dt
            energy_scale_at_t1 = self.coupling_energy_scale.eval(t1)
            energy_scale_at_t2 = self.coupling_energy_scale.eval(t2)
            elem = energy_scale_at_t1 * energy_scale_at_t2
            self.pkl_part.coupling_energy_scale_prod_cache.append(elem)

            selected_eta = self.select_eta_from_cache(l1, l2, n)
            self.pkl_part.eta_selection_cache += [selected_eta]

        return None



    def select_eta_from_cache(self, l1, l2, n):
        K_tau = self.pkl_part.K_tau

        if (l1 == 0) and (0 <= l2 <= min(2*K_tau-1, 2*n-1)):
            selected_eta = self.pkl_part.eta_cache_6[2*K_tau-1-l2]
        elif (l1 == 0) and (l2 == 2*n):
            selected_eta = self.pkl_part.eta_cache_1[n-1]
        elif (l1 == 0) and (l2 == 2*n+1):
            selected_eta = self.pkl_part.eta_cache_2[n-1]
        elif (l1 == 1) and (1 <= l2 <= min(2*K_tau, 2*n-1)):
            selected_eta = self.pkl_part.eta_cache_5[2*K_tau-l2]
        elif (l1 == 1) and (l2 == 2*n):
            selected_eta = self.pkl_part.eta_cache_3[n-1]
        elif (l1 == 1) and (l2 == 2*n+1):
            selected_eta = self.pkl_part.eta_cache_1[n-1]
        elif (2 <= l2 <= 2*n-1) and (l2-l1 <= 2*K_tau-1):
            selected_eta = self.pkl_part.eta_cache_4[2*K_tau-1-l2+l1]
        elif (max(2, 2*n+1-2*K_tau) <= l1 <= 2*n) and (l2 == 2*n):
            selected_eta = self.pkl_part.eta_cache_5[2*K_tau-2*n-1+l1]
        elif (max(2, 2*n+2-2*K_tau) <= l1 <= 2*n+1) and (l2 == 2*n+1):
            selected_eta = self.pkl_part.eta_cache_6[2*K_tau-2*n-2+l1]

        return selected_eta



    def eval(self, j_r_m1, j_r_m2):
        result = 1.0

        base_4_to_ising_pair = sbc._base4.base_4_to_ising_pair
        sigma_r_pos1_q1, sigma_r_neg1_q1 = base_4_to_ising_pair(j_r_m1)
        sigma_r_pos1_q2, sigma_r_neg1_q2 = base_4_to_ising_pair(j_r_m2)

        # Retrieve cached terms set in the call to method
        # :meth:`sbc._influence.twopt.Bath.set_q1_q2_n`.
        zip_obj = zip(self.pkl_part.coupling_energy_scale_prod_cache,
                      self.pkl_part.eta_selection_cache)

        for coupling_energy_scale_prod, eta in zip_obj:
            gamma = (coupling_energy_scale_prod
                     * (sigma_r_pos1_q2-sigma_r_neg1_q2)
                     * ((sigma_r_pos1_q1-sigma_r_neg1_q1) * eta.real
                         + 1.0j * (sigma_r_pos1_q1+sigma_r_neg1_q1) * eta.imag))
            result *= cmath.exp(-gamma)

        return result



class TF():
    def __init__(self, r, system_model, dt, spin_basis):
        self.x_field = system_model.x_fields[r]
        self.dt = dt
        self.c = 1 if spin_basis=="y" else 2

        self.set_k_n(0, 1)

        return None



    def set_k_n(self, k, n):
        dt = self.dt
        w_n_k = 1.0 if 1 <= k <= n-1 else 0.5
        h_x_r_k = self.x_field.eval(t=k*dt)
        theta_r_n_k = 2 * dt * w_n_k * h_x_r_k

        self.cos_cache = math.cos(theta_r_n_k / 2)
        self.sin_cache = math.sin(theta_r_n_k / 2)

        return None



    def eval(self, j_r_m1, j_r_m2):
        # Retrieve cached terms set in the call to method
        # :meth:`sbc._influence.twopt.TF.set_k_n`.
        cos_cache = self.cos_cache
        sin_cache = self.sin_cache

        base_4_to_ising_pair = sbc._base4.base_4_to_ising_pair
        sigma_r_pos1_q1, sigma_r_neg1_q1 = base_4_to_ising_pair(j_r_m1)
        sigma_r_pos1_q2, sigma_r_neg1_q2 = base_4_to_ising_pair(j_r_m2)

        c = self.c
        result = (0.25 * (sigma_r_pos1_q2+sigma_r_pos1_q1)**2 * cos_cache
                  + (1.0j**(1+c) * (0.5 * (sigma_r_pos1_q2-sigma_r_pos1_q1))**c
                     * sin_cache))
        result *= (0.25 * (sigma_r_neg1_q1+sigma_r_neg1_q2)**2 * cos_cache
                   - (1.0j**(1+c) * (0.5 * (sigma_r_neg1_q1-sigma_r_neg1_q2))**c
                      * sin_cache))

        return result



class YZ():
    def __init__(self):
        return None

    

    def eval(self, j_r_m1, j_r_m2):
        base_4_to_ising_pair = sbc._base4.base_4_to_ising_pair
        sigma_r_pos1_q1, sigma_r_neg1_q1 = base_4_to_ising_pair(j_r_m1)
        sigma_r_pos1_q2, sigma_r_neg1_q2 = base_4_to_ising_pair(j_r_m2)

        result = ((1.0 + 1.0j*sigma_r_pos1_q1 + sigma_r_pos1_q2
                   - 1.0j*sigma_r_pos1_q1*sigma_r_pos1_q2)
                  * (1.0 - 1.0j*sigma_r_neg1_q1 + sigma_r_neg1_q2
                     + 1.0j*sigma_r_neg1_q1*sigma_r_neg1_q2)) / 8.0

        return result



class ZY():
    def __init__(self):
        return None

    

    def eval(self, j_r_m1, j_r_m2):
        base_4_to_ising_pair = sbc._base4.base_4_to_ising_pair
        sigma_r_pos1_q1, sigma_r_neg1_q1 = base_4_to_ising_pair(j_r_m1)
        sigma_r_pos1_q2, sigma_r_neg1_q2 = base_4_to_ising_pair(j_r_m2)

        result = ((1.0 - 1.0j*sigma_r_pos1_q2 + sigma_r_pos1_q1
                   + 1.0j*sigma_r_pos1_q2*sigma_r_pos1_q1)
                  * (1.0 + 1.0j*sigma_r_neg1_q2 + sigma_r_neg1_q1
                     - 1.0j*sigma_r_neg1_q2*sigma_r_neg1_q1)) / 8.0

        return result



class Total():
    def __init__(self, r, system_model, bath_model, dt, pkl_parts=None):
        if pkl_parts is None:
            pkl_parts = {"twopt_y_bath_influence": None,
                         "twopt_z_bath_influence": None}

        self.z_bath = Bath(r,
                           bath_model,
                           dt,
                           spin_basis="z",
                           pkl_part=pkl_parts["twopt_z_bath_influence"])

        if bath_model.y_spectral_densities != None:
            self.alg = "yz-noise"
            self.y_bath = Bath(r,
                               bath_model,
                               dt,
                               spin_basis="y",
                               pkl_part=pkl_parts["twopt_y_bath_influence"])
            self.tf = TF(r, system_model, dt, spin_basis="y")
            self.yz = YZ()
            self.zy = ZY()
        else:
            self.alg = "z-noise"
            self.tf = TF(r, system_model, dt, spin_basis="z")
        
        self.set_m1_m2_n(0, 0, 1)

        return None



    def set_m1_m2_n(self, m1, m2, n):
        self.two_pt_objs_to_eval = []
        
        if self.alg == "yz-noise":
            q1 = (2-((m1+2)%3)//2)*(m1//3) + (m1%3+1)//3
            q2 = (2-((m2+2)%3)//2)*(m2//3) + (m2%3+1)//3
            
            if (m1%3 == 0) and (m2%3 == 0):
                self.z_bath.set_q1_q2_n(q1, q2, n)
                self.two_pt_objs_to_eval += [self.z_bath]
            elif (m1 == m2-1) and (m2%3 == 0):
                self.two_pt_objs_to_eval += [self.yz]
            elif (m1 == m2-1) and (m2%3 == 1):
                self.two_pt_objs_to_eval += [self.zy]
            elif (m1%3 != 0) and (m2%3 == 1):
                self.y_bath.set_q1_q2_n(q1, q2, n)
                self.two_pt_objs_to_eval += [self.y_bath]
            elif (m1 == m2-1) and (m2%3 == 2):
                k = q1 // 2
                self.tf.set_k_n(k, n)
                self.two_pt_objs_to_eval += [self.tf]
                self.y_bath.set_q1_q2_n(q1, q2, n)
                self.two_pt_objs_to_eval += [self.y_bath]
            elif (m1 != m2-1) and (m1%3 != 0) and (m2%3 == 2):
                self.y_bath.set_q1_q2_n(q1, q2, n)
                self.two_pt_objs_to_eval += [self.y_bath]
        elif self.alg == "z-noise":
            q1 = m1
            q2 = m2

            self.z_bath.set_q1_q2_n(q1, q2, n)
            self.two_pt_objs_to_eval += [self.z_bath]

            if m1 == m2-1:
                k = m1
                self.tf.set_k_n(k, n)
                self.two_pt_objs_to_eval += [self.tf]

        return None



    def eval(self, j_r_m1, j_r_m2):
        result = 1.0
        for two_pt_obj in self.two_pt_objs_to_eval:
            result *= two_pt_obj.eval(j_r_m1, j_r_m2)

        return result
