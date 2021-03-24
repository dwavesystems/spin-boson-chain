#!/usr/bin/env python

#####################################
## Load libraries/packages/modules ##
#####################################

# For creating sequences of integers.
from itertools import product



# For creating arrays to be used to construct tensor nodes and networks. Also
# for evaluating special math functions.
import numpy as np

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# Import class representing time-dependent scalar model parameters.
from sbc.scalar import Scalar

# For specifying system model parameters.
from sbc import system

# For specifying bath model components.
from sbc import bath

# For calculating the total two-point influence function.
import sbc._influence.twopt

# For specifying how to truncate Schmidt spectra in MPS compression.
from sbc import trunc

# Import module containing class to test.
from sbc import _influence



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

def inefficient_way_of_calculating_path_z_noise(total_two_point_influence,
                                                initial_state,
                                                j_path):
    n = len(j_path) - 2
    K_tau = total_two_point_influence.z_bath.K_tau
    mu_m_tau = lambda m: max(0, m-K_tau+1)

    j_r_0 = j_path[0]
    result = initial_state[j_r_0]

    for m2 in range(0, n+2):
        mu_m2_tau = mu_m_tau(m=m2)
        for m1 in range(mu_m2_tau, m2+1):
            total_two_point_influence.set_m1_m2_n(m1, m2, n)
            j_r_m1 = j_path[m1]
            j_r_m2 = j_path[m2]
            result *= total_two_point_influence.eval(j_r_m1, j_r_m2)

    return result



def inefficient_way_for_calculating_rdm_z_noise(total_two_pt_influence,
                                                initial_state,
                                                n,
                                                j):
    result = 0.0j
    j_paths = product(range(0, 4), repeat=n+1)
    for j_path in j_paths:
        extended_j_path = list(j_path)+[j]
        result += \
            inefficient_way_of_calculating_path_z_noise(total_two_pt_influence,
                                                        initial_state,
                                                        extended_j_path)

    return result



def _get_eta(omega_c, beta, target_W, eta_upper_limit,
             A_z_0T_cmpnt_func_form, tol):
    eta_lower_limit = 0
    eta = eta_upper_limit

    A_z_0T_cmpnt = bath.SpectralDensityCmpnt0T(func_form=A_z_0T_cmpnt_func_form,
                                               func_kwargs={"eta": eta,
                                                            "omega_c": omega_c},
                                               hard_cutoff_freq=40*omega_c,
                                               zero_pt_derivative=eta)
    A_z_T_cmpnt = bath.SpectralDensityCmpnt(limit_0T=A_z_0T_cmpnt, beta=beta)
    W = bath.noise_strength(A_z_T_cmpnt)
    
    while abs(W - target_W) > tol * target_W:
        if W > target_W:
            eta_upper_limit = eta
            eta = (eta_lower_limit + eta) / 2
        else:
            eta_lower_limit = eta
            eta = (eta_upper_limit + eta) / 2
            
        A_z_0T_cmpnt = \
            bath.SpectralDensityCmpnt0T(func_form=A_z_0T_cmpnt_func_form,
                                        func_kwargs={"eta": eta,
                                                     "omega_c": omega_c},
                                        hard_cutoff_freq=40*omega_c,
                                        zero_pt_derivative=eta)
        A_z_T_cmpnt = bath.SpectralDensityCmpnt(limit_0T=A_z_0T_cmpnt,
                                                beta=beta)
        W = bath.noise_strength(A_z_T_cmpnt)        
        
    return eta



# Naive simulation test #1.
print("Naive simulation test #1")
print("============================")

# Need to construct a ``system.Model`` object, which specifies all the model,
# parameters. Since we are only dealing with influence, all we need are the
# x-fields.
print("Constructing an instance of ``system.Model``.\n")

system_model = system.Model(x_fields=[-0.05])

# Need to construct a ``bath.Model`` object. In order to do this, we need to a
# few more objects. Starting the the coupling energy scales. We'll assume zero
# y-noise here.
print("Constructing an instance of ``bath.Model`` with z-noise only.\n")

# Next we need to construct the spectral-densities of noise.
def A_z_0T_cmpnt_func_form(omega, eta, omega_c):
    return eta * omega * np.exp(-omega / omega_c)

omega_c = 0.1
W = 0.5
beta = 1

if W == 0.0:
    eta = 0.0
else:
    eta_upper_limit = 1000
    tol = 1.0e-6
    eta = _get_eta(omega_c, beta, W, eta_upper_limit,
                   A_z_0T_cmpnt_func_form, tol)

# The single Ohmic component of the zero-temperature spectral density.
A_z_0T_cmpnt = bath.SpectralDensityCmpnt0T(func_form=A_z_0T_cmpnt_func_form,
                                           func_kwargs={"eta": eta,
                                                        "omega_c": omega_c},
                                           hard_cutoff_freq=40*omega_c,
                                           zero_pt_derivative=eta)

# The zero-temperature spectral density (trivially contains one component).
A_z_0T = bath.SpectralDensity0T(cmpnts=[A_z_0T_cmpnt])
        
# Specify bath model components.
t_f = 4.5
tau = t_f
bath_model = bath.Model(L=1,
                        beta=beta,
                        memory=t_f,
                        z_coupling_energy_scales=[1.0],
                        z_spectral_densities_0T=[A_z_0T])

# Constructing total two-point influence.
print("Constructing an instance of ``_influence.twopt.Total``.")
r = 0
dt = 0.75
total_two_point_influence = \
    sbc._influence.twopt.Total(r, system_model, bath_model, dt)

# Run simulation and track decoherence in computational basis.
j = 2
initial_state = 0.5 * np.ones([4])
print("n={}; rho_A_in_CB_10={}".format(0, initial_state[j]))
n_final = 6
for n in range(1, n_final+1):
    rho_A_in_CB_10 = \
        inefficient_way_for_calculating_rdm_z_noise(total_two_point_influence,
                                                    initial_state,
                                                    n,
                                                    j)
    print("n={}; rho_A_in_CB_10={}".format(n, rho_A_in_CB_10))
