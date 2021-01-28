#!/usr/bin/env python

#####################################
## Load libraries/packages/modules ##
#####################################

# For evaluating special math functions.
import numpy as np



# Import class representing time-dependent scalar model parameters.
from sbc.scalar import Scalar

# For specifying system model parameters.
from sbc import system

# For specifying bath model components.
from sbc import bath

# Class to test.
from sbc._influence.twopt import Total



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

# _influence.twopt.Total test #1.
print("_influence.twopt.Total test #1")
print("==============================")

# Need to construct a ``system.Model`` object, which specifies all the model,
# parameters. Because we are testing two-point influence functions, all we need
# are the x-fields.
print("Constructing an instance of ``system.Model``.\n")

def quad_fn(t, a, b):
    return a * t * t + b

x_fields = [Scalar(quad_fn, {"a": 0.05, "b": -0.5}),
            Scalar(quad_fn, {"a": -0.05, "b": 0.5}),
            0.5]
system_model = system.Model(x_fields=x_fields)

# Need to construct a ``bath.Model`` object. In order to do this, we need to a
# few more objects. Starting the the coupling energy scales. We'll assume zero
# y-noise here.
print("Constructing an instance of ``bath.Model`` with z-noise only.\n")

def linear_fn(t, a, b):
    return a*t+b

z_coupling_energy_scales = [0.0,
                            Scalar(linear_fn, {"a": 0.2, "b": 0.1}),
                            0.0]

# Next we need to construct the spectral-densities of noise. 
def A_v_r_0T_s_func_form(omega, lambda_v_s, omega_v_s_c):
    return lambda_v_s * omega * np.exp(-omega / omega_v_s_c)

lambda_z_1 = 3 * 0.1025 / 8
omega_z_1_c = 3 * 10.0 / 8

A_z_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_v_r_0T_s_func_form,
                                func_kwargs={"lambda_v_s": lambda_z_1,
                                             "omega_v_s_c": omega_z_1_c},
                                hard_cutoff_freq=40*omega_z_1_c,
                                zero_pt_derivative=lambda_z_1)

lambda_z_2 = 0.1025 / 8
omega_z_2_c = 10.0 / 8

A_z_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_v_r_0T_s_func_form,
                                func_kwargs={"lambda_v_s": lambda_z_2,
                                             "omega_v_s_c": omega_z_2_c},
                                hard_cutoff_freq=40*omega_z_2_c,
                                zero_pt_derivative=lambda_z_2)

A_z_0_0T = 0
A_z_1_0T = bath.SpectralDensity0T(cmpnts=[A_z_r_0T_1, A_z_r_0T_2])
A_z_2_0T = 0

z_spectral_densities_0T = [A_z_0_0T, A_z_1_0T, A_z_2_0T]

bath_model = bath.Model(L=3,
                        beta=1.0,
                        memory=0.5,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        z_spectral_densities_0T=z_spectral_densities_0T)

print("Constructing various instances of ``_influence.twopt.Total`` to "
      "evaluate the 'total' two-point influence function for a given set of "
      "system and bath model components for various "
      "(r, m1, m2, n, j_r_m1, j_r_m2):\n")

dt = 0.1
unformatted_msg = ("    Evaluating for (r, m1, m2, n, j_r_m1, j_r_m2)="
                   "({}, {}, {}, {}, {}, {}): Result={}")
for r in range(bath_model.L):
    total_two_pt_influence = Total(r, system_model, bath_model, dt)
    K_tau = total_two_pt_influence.z_bath.K_tau
    for n in range(1, 11):
        for m2 in range(n+2):
            mu_m2_tau = max(0, m2-K_tau+1)
            for m1 in range(mu_m2_tau, m2+1):
                total_two_pt_influence.set_m1_m2_n(m1, m2, n)
                for j_r_m1 in range(4):
                    for j_r_m2 in range(4):
                        result = total_two_pt_influence.eval(j_r_m1, j_r_m2)
                        msg = unformatted_msg.format(r, m1, m2, n,
                                                     j_r_m1, j_r_m2, result)
                        print(msg)
print("\n\n")



# _influence.twopt.Total test #2.
print("_influence.twopt.Total test #2")
print("==============================")

# Need to construct a ``bath.Model`` object that encodes both y- and z-noise.
print("Constructing an instance of ``bath.Model`` with y- and z-noise.\n")

y_coupling_energy_scales = [Scalar(linear_fn, {"a": 0.4, "b": -0.2}),
                            Scalar(linear_fn, {"a": -0.2, "b": 0.1}),
                            0.25]

lambda_y_1 = 3 * 0.1025 / 4
omega_y_1_c = 3 * 10.0 / 4

A_y_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_v_r_0T_s_func_form,
                                func_kwargs={"lambda_v_s": lambda_y_1,
                                             "omega_v_s_c": omega_y_1_c},
                                hard_cutoff_freq=40*omega_y_1_c,
                                zero_pt_derivative=lambda_y_1)

lambda_y_2 = 0.1025 / 4
omega_y_2_c = 10.0 / 4

A_y_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_v_r_0T_s_func_form,
                                func_kwargs={"lambda_v_s": lambda_y_2,
                                             "omega_v_s_c": omega_y_2_c},
                                hard_cutoff_freq=40*omega_y_2_c,
                                zero_pt_derivative=lambda_y_2)


A_y_0_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1])
A_y_1_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1, A_y_r_0T_2])
A_y_2_0T = 0

y_spectral_densities_0T = [A_y_0_0T, A_y_1_0T, A_y_2_0T]

bath_model = bath.Model(L=3,
                        beta=1.0,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T,
                        z_spectral_densities_0T=z_spectral_densities_0T)

print("Constructing various instances of ``_influence.twopt.Total`` to "
      "evaluate the 'total' two-point influence function for a given set of "
      "system and bath model components for various "
      "(r, m1, m2, n, j_r_m1, j_r_m2):\n")

dt = 0.1
unformatted_msg = ("    Evaluating for (r, m1, m2, n, j_r_m1, j_r_m2)="
                   "({}, {}, {}, {}, {}, {}): Result={}")
for r in range(bath_model.L):
    total_two_pt_influence = Total(r, system_model, bath_model, dt)
    K_tau = total_two_pt_influence.z_bath.K_tau
    for n in range(1, 11):
        for m2 in range(0,3*n+4):
            mu_m2_tau = max(0, m2-3*K_tau+1)
            for m1 in range(mu_m2_tau, m2+1):
                total_two_pt_influence.set_m1_m2_n(m1, m2, n)
                for j_r_m1 in range(4):
                    for j_r_m2 in range(4):
                        result = total_two_pt_influence.eval(j_r_m1, j_r_m2)
                        msg = unformatted_msg.format(r, m1, m2, n,
                                                     j_r_m1, j_r_m2, result)
                        print(msg)
