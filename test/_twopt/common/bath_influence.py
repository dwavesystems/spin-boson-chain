#!/usr/bin/env python

#####################################
## Load libraries/packages/modules ##
#####################################

# For evaluating special math functions.
import numpy as np



# For specifying bath model components.
from sbc import bath

# To construct a sbc._twopt.common.BathInfluence object.
from sbc._twopt.common import Eta

# Import implementation of Eq. (70) of the detailed manuscript of our QUAPI-TN
# approach.
from sbc._twopt.common import eval_tilde_k_m

# Class to test.
from sbc._twopt.common import BathInfluence



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Non-Production"



#########################
## Main body of script ##
#########################

# _twopt.common.Eta test #1.
print("_twopt.common.Eta test #1")
print("=========================")

print("Constructing an instance of _twopt.common.Eta.\n")

# Need to construct a bath.SpectralDensityCmpnt object.

# A subcomponent of spectral density.
def A_v_0T_1_func_form(omega, lambda_v_1, omega_v_1_c):
    return lambda_v_1 * omega * np.exp(-omega / omega_v_1_c)

lambda_v_1 = 3.0 * 8.5 / 4.0
omega_v_1_c = 0.1

A_v_0T_1 = \
    bath.SpectralDensitySubcmpnt0T(func_form=A_v_0T_1_func_form,
                                   func_kwargs={"lambda_v_1": lambda_v_1,
                                                "omega_v_1_c": omega_v_1_c},
                                   hard_cutoff_freq=40*omega_v_1_c,
                                   zero_pt_derivative=lambda_v_1)

# A another subcomponent of spectral density
def A_v_0T_2_func_form(omega, lambda_v_2, omega_v_2_c):
    return lambda_v_2 * omega * np.exp(-omega / omega_v_2_c)

lambda_v_2 = 8.5 / 4.0
omega_v_2_c = 0.1

A_v_0T_2 = \
    bath.SpectralDensitySubcmpnt0T(func_form=A_v_0T_2_func_form,
                                   func_kwargs={"lambda_v_2": lambda_v_2,
                                                "omega_v_2_c": omega_v_2_c},
                                   hard_cutoff_freq=40*omega_v_2_c,
                                   zero_pt_derivative=lambda_v_2)

A_v_0T = bath.SpectralDensityCmpnt0T(subcmpnts=[A_v_0T_1, A_v_0T_2])
beta = 1.0
A_v_T = bath.SpectralDensityCmpnt(limit_0T=A_v_0T, beta=beta)

dt = 0.1
tilde_w_y_set = [0.25, 0.25, 0.25, 0.50]
tilde_w_z_set = [0.50, 0.25, 0.75, 1.00]
eval_k_y_n = lambda n: 2*n+1
eval_k_z_n = lambda n: n+1

eta_y = Eta(A_v_T, dt, tilde_w_y_set, eval_k_y_n)
eta_z = Eta(A_v_T, dt, tilde_w_z_set, eval_k_z_n)

print("Testing eta-function caching.")

for K_tau in range(3, 11):
    K_y_tau = 2*K_tau
    y_bath_influence = BathInfluence(eta_y, K_tau, K_y_tau)
    unformatted_msg_y = \
        "K_y_tau={}; n={}; tilde_k_m1={}; tilde_k_m2={}; y_diff={}"
    try:
        for n in range(1, 2*K_tau):
            for m2 in range(0, (3*n+3)+1):
                mu_m2_tau = max(0, m2-3*K_tau+1)
                for m1 in range(mu_m2_tau, m2):
                    if (m1%3 != 0) and (m2%3 != 0):
                        tilde_k_m1 = eval_tilde_k_m(m=m1)
                        tilde_k_m2 = eval_tilde_k_m(m=m2)
                        if m2 <= 3*n-1:
                            n1 = np.inf
                            n2 = n+2
                        else:
                            n1 = n
                            n2 = n
                        y_bath_influence.set_k1_k2_n(tilde_k_m1, tilde_k_m2, n1)
                        eta_y_cache = \
                            (y_bath_influence.selected_eta_cache_real_part
                             + 1j*y_bath_influence.selected_eta_cache_imag_part)
                        expected_eta_y = eta_y.eval(tilde_k_m2, tilde_k_m1, n2)
                        y_diff = abs(eta_y_cache-expected_eta_y)
                        if y_diff > 1.0e-13:
                            msg = unformatted_msg_y.format(K_y_tau, n,
                                                           tilde_k_m1,
                                                           tilde_k_m2,
                                                           y_diff)
                            print(msg)
                            
    except Exception as e:
        print(e)
        msg = unformatted_msg_y.format(K_y_tau, n, tilde_k_m1, tilde_k_m2, None)
        print(msg)
        raise

    K_z_tau = K_tau
    z_bath_influence = BathInfluence(eta_z, K_tau, K_z_tau)
    unformatted_msg_z = \
        "K_z_tau={}; n={}; tilde_k_m1={}; tilde_k_m2={}; z_diff={}"
    try:
        for n in range(1, 2*K_tau):
            for m2 in range(0, (3*n+3)+1):
                mu_m2_tau = max(0, m2-3*K_tau+1)
                for m1 in range(mu_m2_tau, m2):
                    if (m1%3 == 0) and (m2%3 == 0):
                        tilde_k_m1 = eval_tilde_k_m(m=m1)
                        tilde_k_m2 = eval_tilde_k_m(m=m2)
                        if m2 <= 3*n-1:
                            n1 = np.inf
                            n2 = n+2
                        else:
                            n1 = n
                            n2 = n
                        z_bath_influence.set_k1_k2_n(tilde_k_m1, tilde_k_m2, n1)
                        eta_z_cache = \
                            (z_bath_influence.selected_eta_cache_real_part
                             + 1j*z_bath_influence.selected_eta_cache_imag_part)
                        expected_eta_z = eta_z.eval(tilde_k_m2, tilde_k_m1, n2)
                        z_diff = abs(eta_z_cache-expected_eta_z)
                        if z_diff > 1.0e-13:
                            msg = unformatted_msg_z.format(K_z_tau, n,
                                                           tilde_k_m1,
                                                           tilde_k_m2,
                                                           z_diff)
                            print(msg)

    except Exception as e:
        print(e)
        msg = unformatted_msg_z.format(K_z_tau, n, tilde_k_m1, tilde_k_m2, None)
        print(msg)
        raise

print("Found no obvious bugs with eta-function caching.\n")

print("Testing evaluation method.")

for j_r_k1 in range(0, 4):
    for j_r_k2 in range(0, 4):
        y_bath_influence.eval(j_r_k1, j_r_k2)
        z_bath_influence.eval(j_r_k1, j_r_k2)

print("Found no obvious bugs with evaluation method.\n")
