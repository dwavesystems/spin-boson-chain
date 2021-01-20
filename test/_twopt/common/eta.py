#!/usr/bin/env python

#####################################
## Load libraries/packages/modules ##
#####################################

# For evaluating special math functions.
import numpy as np



# For specifying bath model components.
from sbc import bath

# Class to test.
from sbc._twopt.common import Eta



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
def A_z_0T_1_func_form(omega, lambda_z_1, omega_z_1_c):
    return lambda_z_1 * omega * np.exp(-omega / omega_z_1_c)

lambda_z_1 = 3.0 * 8.5 / 4.0
omega_z_1_c = 0.1

A_z_0T_1 = \
    bath.SpectralDensitySubcmpnt0T(func_form=A_z_0T_1_func_form,
                                   func_kwargs={"lambda_z_1": lambda_z_1,
                                                "omega_z_1_c": omega_z_1_c},
                                   hard_cutoff_freq=40*omega_z_1_c,
                                   zero_pt_derivative=lambda_z_1)

# Another subcomponent of spectral density
def A_z_0T_2_func_form(omega, lambda_z_2, omega_z_2_c):
    return lambda_z_2 * omega * np.exp(-omega / omega_z_2_c)

lambda_z_2 = 8.5 / 4.0
omega_z_2_c = 0.1

A_z_0T_2 = \
    bath.SpectralDensitySubcmpnt0T(func_form=A_z_0T_2_func_form,
                                   func_kwargs={"lambda_z_2": lambda_z_2,
                                                "omega_z_2_c": omega_z_2_c},
                                   hard_cutoff_freq=40*omega_z_2_c,
                                   zero_pt_derivative=lambda_z_2)

A_z_0T = bath.SpectralDensityCmpnt0T(subcmpnts=[A_z_0T_1, A_z_0T_2])
beta = 1.0
A_z_T = bath.SpectralDensityCmpnt(limit_0T=A_z_0T, beta=beta)

dt = 0.1
tilde_w_set = [1.0, 0.5, 1.0, 1.0]
eval_k_z_n = lambda n: n

eta = Eta(A_z_T, dt, tilde_w_set, eval_k_z_n)

print("Evaluating eta-function for various (k1, k2, n) triplets:")
unformatted_msg = "    Evaluating for (k1, k2, n)=({}, {}, {}): Result={}"

n = 10
k1 = n
for k2 in range(1, n+1):
    print(unformatted_msg.format(k1, k2, n, eta.eval(k1, k2, n)))
print()

n = 11
k1 = n-1
for k2 in range(1, k1):
    print(unformatted_msg.format(k1, k2, n, eta.eval(k1, k2, n)))
print()

k2 = 0
for k1 in range(1, 10):
    n = k1
    print(unformatted_msg.format(k1, k2, n, eta.eval(k1, k2, n)))
print()

k2 = 0
for k1 in range(1, 10):
    n = k1+1
    print(unformatted_msg.format(k1, k2, n, eta.eval(k1, k2, n)))
print()
