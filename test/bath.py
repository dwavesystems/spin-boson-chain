#!/usr/bin/env python
r"""This script runs several tests on the :mod:`sbc.bath` module."""



#####################################
## Load libraries/packages/modules ##
#####################################

# For evaluating special math functions.
import numpy as np



# Module to test.
from sbc import bath



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

# bath.SpectralDensitySubcmpnt0T test #1.
print("bath.SpectralDensitySubcmpnt0T test #1")
print("======================================")

print("Constructing an instance of bath.SpectralDensitySubcmpnt0T.\n")

# A low-frequency subcomponent.
def A_z_0T_LF_func_form(omega, lambda_z_LF, omega_z_LF_c):
    return lambda_z_LF * omega * np.exp(-omega / omega_z_LF_c)

lambda_z_LF = 8.5
omega_z_LF_c = 0.1

A_z_0T_LF = \
    bath.SpectralDensitySubcmpnt0T(func_form=A_z_0T_LF_func_form,
                                   func_kwargs={"lambda_z_LF": lambda_z_LF,
                                                "omega_z_LF_c": omega_z_LF_c},
                                   hard_cutoff_freq=40*omega_z_LF_c,
                                   zero_pt_derivative=lambda_z_LF)

print("Evaluating zero-temperature subcomponent of spectral density at various "
      "frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_z_0T_LF.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_z_0T_LF.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_z_0T_LF.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_z_0T_LF.eval(0.5)))
print(unformatted_msg.format(-0.5, A_z_0T_LF.eval(-0.5)))
print(unformatted_msg.format(5.0, A_z_0T_LF.eval(5.0)))
print(unformatted_msg.format(-5.0, A_z_0T_LF.eval(-5.0)))
print("\n\n")



# bath.SpectralDensitySubcmpnt0T test #2.
print("bath.SpectralDensitySubcmpnt0T test #2")
print("======================================")

print("Constructing an instance of bath.SpectralDensitySubcmpnt0T.\n")

# A high-frequency subcomponent.
def A_z_0T_HF_func_form(omega, lambda_z_HF, omega_z_HF_c):
    return lambda_z_HF * omega * np.exp(-omega / omega_z_HF_c)

lambda_z_HF = 0.1025
omega_z_HF_c = 10.0

A_z_0T_HF = \
    bath.SpectralDensitySubcmpnt0T(func_form=A_z_0T_HF_func_form,
                                   func_kwargs={"lambda_z_HF": lambda_z_HF,
                                                "omega_z_HF_c": omega_z_HF_c},
                                   hard_cutoff_freq=40*omega_z_HF_c,
                                   zero_pt_derivative=lambda_z_HF)

print("Evaluating zero-temperature subcomponent of spectral density at various "
      "frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_z_0T_HF.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_z_0T_HF.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_z_0T_HF.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_z_0T_HF.eval(0.5)))
print(unformatted_msg.format(-0.5, A_z_0T_HF.eval(-0.5)))
print(unformatted_msg.format(5.0, A_z_0T_HF.eval(5.0)))
print(unformatted_msg.format(-5.0, A_z_0T_HF.eval(-5.0)))
print("\n\n")



# bath.SpectralDensitySubcmpnt0T test #3.
print("bath.SpectralDensitySubcmpnt0T test #3")
print("======================================")

print("Constructing an instance of bath.SpectralDensitySubcmpnt0T.\n")

# A high-frequency subcomponent.
def A_y_0T_HF_func_form(omega, lambda_y_HF, omega_y_HF_c):
    return lambda_y_HF * omega * np.exp(-omega / omega_y_HF_c)

lambda_y_HF = 0.1025 / 2
omega_y_HF_c = 10.0 / 2

A_y_0T_HF = \
    bath.SpectralDensitySubcmpnt0T(func_form=A_y_0T_HF_func_form,
                                   func_kwargs={"lambda_y_HF": lambda_y_HF,
                                                "omega_y_HF_c": omega_y_HF_c},
                                   hard_cutoff_freq=40*omega_y_HF_c,
                                   zero_pt_derivative=lambda_y_HF)

print("Evaluating zero-temperature subcomponent of spectral density at various "
      "frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_y_0T_HF.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_y_0T_HF.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_y_0T_HF.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_y_0T_HF.eval(0.5)))
print(unformatted_msg.format(-0.5, A_y_0T_HF.eval(-0.5)))
print(unformatted_msg.format(5.0, A_y_0T_HF.eval(5.0)))
print(unformatted_msg.format(-5.0, A_y_0T_HF.eval(-5.0)))
print("\n\n")



# bath.SpectralDensityCmpnt0T test #1.
print("bath.SpectralDensityCmpnt0T test #1")
print("===================================")

print("Constructing an instance of bath.SpectralDensityCmpnt0T with two "
      "subcomponents.\n")

A_z_0T = bath.SpectralDensityCmpnt0T(subcmpnts=[A_z_0T_LF, A_z_0T_HF])

print("Evaluating zero-temperature spectral density of component of noise of "
      "interest at various frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_z_0T.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_z_0T.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_z_0T.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_z_0T.eval(0.5)))
print(unformatted_msg.format(-0.5, A_z_0T.eval(-0.5)))
print(unformatted_msg.format(5.0, A_z_0T.eval(5.0)))
print(unformatted_msg.format(-5.0, A_z_0T.eval(-5.0)))
print("\n\n")



# bath.SpectralDensityCmpnt0T test #2.
print("bath.SpectralDensityCmpnt0T test #2")
print("===================================")

print("Constructing an instance of bath.SpectralDensityCmpnt0T with two "
      "subcomponents.\n")

A_y_0T = bath.SpectralDensityCmpnt0T(subcmpnts=[A_y_0T_HF])

print("Evaluating zero-temperature spectral density of component of noise of "
      "interest at various frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_y_0T.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_y_0T.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_y_0T.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_y_0T.eval(0.5)))
print(unformatted_msg.format(-0.5, A_y_0T.eval(-0.5)))
print(unformatted_msg.format(5.0, A_y_0T.eval(5.0)))
print(unformatted_msg.format(-5.0, A_y_0T.eval(-5.0)))
print("\n\n")



# bath.SpectralDensityCmpnt test #1.
print("bath.SpectralDensityCmpnt test #1")
print("=================================")

print("Constructing an instance of bath.SpectralDensityCmpnt with two "
      "subcomponents.\n")

beta = 1.0
A_z = bath.SpectralDensityCmpnt(limit_0T=A_z_0T, beta=beta)

print("Evaluating finite-temperature spectral density of component of noise of "
      "interest at various frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_z.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_z.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_z.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_z.eval(0.5)))
print(unformatted_msg.format(-0.5, A_z.eval(-0.5)))
print(unformatted_msg.format(5.0, A_z.eval(5.0)))
print(unformatted_msg.format(-5.0, A_z.eval(-5.0)))
print("\n\n")



# bath.SpectralDensityCmpnt test #2.
print("bath.SpectralDensityCmpnt test #2")
print("=================================")

print("Constructing an instance of bath.SpectralDensityCmpnt with one "
      "subcomponent.\n")

beta = 1.0
A_y = bath.SpectralDensityCmpnt(limit_0T=A_y_0T, beta=beta)

print("Evaluating finite-temperature spectral density of component of noise of "
      "interest at various frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_y.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_y.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_y.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_y.eval(0.5)))
print(unformatted_msg.format(-0.5, A_y.eval(-0.5)))
print(unformatted_msg.format(5.0, A_y.eval(5.0)))
print(unformatted_msg.format(-5.0, A_y.eval(-5.0)))
print("\n\n")



# bath.Model test #1.
print("bath.Model test #1")
print("==================")

print("Constructing an instance of bath.Model.\n")

bath_model = bath.Model(spectral_density_y_cmpnt_0T=A_y_0T,
                        spectral_density_z_cmpnt_0T=A_z_0T,
                        beta=beta)

print("Evaluating finite-temperature spectral density of the yth component of "
      "noise at various frequencies:")
A_y = bath_model.spectral_density_y_cmpnt
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_y.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_y.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_y.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_y.eval(0.5)))
print(unformatted_msg.format(-0.5, A_y.eval(-0.5)))
print(unformatted_msg.format(5.0, A_y.eval(5.0)))
print(unformatted_msg.format(-5.0, A_y.eval(-5.0)))
print("\n\n")

print("Evaluating finite-temperature spectral density of the zth component of "
      "noise at various frequencies:")
A_z = bath_model.spectral_density_z_cmpnt
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_z.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_z.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_z.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_z.eval(0.5)))
print(unformatted_msg.format(-0.5, A_z.eval(-0.5)))
print(unformatted_msg.format(5.0, A_z.eval(5.0)))
print(unformatted_msg.format(-5.0, A_z.eval(-5.0)))
