#!/usr/bin/env python
r"""This script runs several tests on the :mod:`sbc.bath` module."""



#####################################
## Load libraries/packages/modules ##
#####################################

# For evaluating special math functions.
import numpy as np



# Import class representing time-dependent scalar model parameters.
from sbc.scalar import Scalar

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

# bath.SpectralDensityCmpnt0T test #1.
print("bath.SpectralDensityCmpnt0T test #1")
print("======================================")

print("Constructing an instance of bath.SpectralDensityCmpnt0T.\n")

# A low-frequency component.
def A_z_r_0T_LF_func_form(omega, lambda_z_LF, omega_z_LF_c):
    return lambda_z_LF * omega * np.exp(-omega / omega_z_LF_c)

lambda_z_LF = 8.5
omega_z_LF_c = 0.1

A_z_r_0T_LF = \
    bath.SpectralDensityCmpnt0T(func_form=A_z_r_0T_LF_func_form,
                                func_kwargs={"lambda_z_LF": lambda_z_LF,
                                             "omega_z_LF_c": omega_z_LF_c},
                                hard_cutoff_freq=40*omega_z_LF_c,
                                zero_pt_derivative=lambda_z_LF)

print("Evaluating zero-temperature component of spectral density at various "
      "frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_z_r_0T_LF.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_z_r_0T_LF.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_z_r_0T_LF.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_z_r_0T_LF.eval(0.5)))
print(unformatted_msg.format(-0.5, A_z_r_0T_LF.eval(-0.5)))
print(unformatted_msg.format(5.0, A_z_r_0T_LF.eval(5.0)))
print(unformatted_msg.format(-5.0, A_z_r_0T_LF.eval(-5.0)))
print("\n\n")



# bath.SpectralDensityCmpnt0T test #2.
print("bath.SpectralDensityCmpnt0T test #2")
print("======================================")

print("Constructing an instance of bath.SpectralDensityCmpnt0T.\n")

# A high-frequency component.
def A_z_r_0T_HF_func_form(omega, lambda_z_HF, omega_z_HF_c):
    return lambda_z_HF * omega * np.exp(-omega / omega_z_HF_c)

lambda_z_HF = 0.1025
omega_z_HF_c = 10.0

A_z_r_0T_HF = \
    bath.SpectralDensityCmpnt0T(func_form=A_z_r_0T_HF_func_form,
                                func_kwargs={"lambda_z_HF": lambda_z_HF,
                                             "omega_z_HF_c": omega_z_HF_c},
                                hard_cutoff_freq=40*omega_z_HF_c,
                                zero_pt_derivative=lambda_z_HF)

print("Evaluating zero-temperature component of spectral density at various "
      "frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_z_r_0T_HF.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_z_r_0T_HF.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_z_r_0T_HF.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_z_r_0T_HF.eval(0.5)))
print(unformatted_msg.format(-0.5, A_z_r_0T_HF.eval(-0.5)))
print(unformatted_msg.format(5.0, A_z_r_0T_HF.eval(5.0)))
print(unformatted_msg.format(-5.0, A_z_r_0T_HF.eval(-5.0)))
print("\n\n")



# bath.SpectralDensityCmpnt0T test #3.
print("bath.SpectralDensityCmpnt0T test #3")
print("======================================")

print("Constructing an instance of bath.SpectralDensityCmpnt0T.\n")

# A high-frequency component.
def A_y_r_0T_HF_func_form(omega, lambda_y_HF, omega_y_HF_c):
    return lambda_y_HF * omega * np.exp(-omega / omega_y_HF_c)

lambda_y_HF = 0.1025 / 2
omega_y_HF_c = 10.0 / 2

A_y_r_0T_HF = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_HF_func_form,
                                func_kwargs={"lambda_y_HF": lambda_y_HF,
                                             "omega_y_HF_c": omega_y_HF_c},
                                hard_cutoff_freq=40*omega_y_HF_c,
                                zero_pt_derivative=lambda_y_HF)

print("Evaluating zero-temperature component of spectral density at various "
      "frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_y_r_0T_HF.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_y_r_0T_HF.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_y_r_0T_HF.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_y_r_0T_HF.eval(0.5)))
print(unformatted_msg.format(-0.5, A_y_r_0T_HF.eval(-0.5)))
print(unformatted_msg.format(5.0, A_y_r_0T_HF.eval(5.0)))
print(unformatted_msg.format(-5.0, A_y_r_0T_HF.eval(-5.0)))
print("\n\n")



# bath.SpectralDensity0T test #1.
print("bath.SpectralDensity0T test #1")
print("===================================")

print("Constructing an instance of bath.SpectralDensity0T with two "
      "components.\n")

A_z_r_0T = bath.SpectralDensity0T(cmpnts=[A_z_r_0T_LF, A_z_r_0T_HF])

print("Evaluating zero-temperature spectral density of noise of interest at "
      "various frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_z_r_0T.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_z_r_0T.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_z_r_0T.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_z_r_0T.eval(0.5)))
print(unformatted_msg.format(-0.5, A_z_r_0T.eval(-0.5)))
print(unformatted_msg.format(5.0, A_z_r_0T.eval(5.0)))
print(unformatted_msg.format(-5.0, A_z_r_0T.eval(-5.0)))
print("\n\n")



# bath.SpectralDensity0T test #2.
print("bath.SpectralDensity0T test #2")
print("===================================")

print("Constructing an instance of bath.SpectralDensity0T with two "
      "components.\n")

A_y_r_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_HF])

print("Evaluating zero-temperature spectral density of noise of interest at "
      "various frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_y_r_0T.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_y_r_0T.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_y_r_0T.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_y_r_0T.eval(0.5)))
print(unformatted_msg.format(-0.5, A_y_r_0T.eval(-0.5)))
print(unformatted_msg.format(5.0, A_y_r_0T.eval(5.0)))
print(unformatted_msg.format(-5.0, A_y_r_0T.eval(-5.0)))
print("\n\n")



# bath.SpectralDensity test #1.
print("bath.SpectralDensity test #1")
print("=================================")

print("Constructing an instance of bath.SpectralDensity with two components.\n")

beta = 1.0
A_z_r_T = bath.SpectralDensity(limit_0T=A_z_r_0T, beta=beta)

print("Evaluating finite-temperature spectral density of noise of interest at "
      "various frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_z_r_T.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_z_r_T.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_z_r_T.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_z_r_T.eval(0.5)))
print(unformatted_msg.format(-0.5, A_z_r_T.eval(-0.5)))
print(unformatted_msg.format(5.0, A_z_r_T.eval(5.0)))
print(unformatted_msg.format(-5.0, A_z_r_T.eval(-5.0)))
print("\n\n")



# bath.SpectralDensity test #2.
print("bath.SpectralDensity test #2")
print("=================================")

print("Constructing an instance of bath.SpectralDensity with one component.\n")

beta = 1.0
A_y_r_T = bath.SpectralDensity(limit_0T=A_y_r_0T, beta=beta)

print("Evaluating finite-temperature spectral density of noise of interest at "
      "various frequencies:")
unformatted_msg = "    Evaluating at omega={}: Result={}"
print(unformatted_msg.format(0.0, A_y_r_T.eval(0.0)))
print(unformatted_msg.format(5.0e-4, A_y_r_T.eval(5.0e-4)))
print(unformatted_msg.format(-5.0e-4, A_y_r_T.eval(-5.0e-4)))
print(unformatted_msg.format(0.5, A_y_r_T.eval(0.5)))
print(unformatted_msg.format(-0.5, A_y_r_T.eval(-0.5)))
print(unformatted_msg.format(5.0, A_y_r_T.eval(5.0)))
print(unformatted_msg.format(-5.0, A_y_r_T.eval(-5.0)))
print("\n\n")



# bath.Model test #1.
print("bath.Model test #1")
print("==================")

print("Constructing an instance of bath.Model.\n")

bath_model = bath.Model(L=3, beta=beta, memory=0.5)

print("Print attributes of object:")
print("    L =", bath_model.L)
print("    memory =", bath_model.memory)
print("    y_coupling_energy_scales =", bath_model.y_coupling_energy_scales)
print("    z_coupling_energy_scales =", bath_model.z_coupling_energy_scales)
print("    y_spectral_densities =", bath_model.y_spectral_densities)
print("    z_spectral_densities =", bath_model.z_spectral_densities)
print()
print("\n\n")



# bath.Model test #2.
print("bath.Model test #2")
print("==================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}

y_coupling_energy_scale_1 = Scalar(linear_fn, func_kwargs_1)
y_coupling_energy_scale_2 = Scalar(linear_fn, func_kwargs_2)
y_coupling_energy_scale_3 = const_scalar

y_coupling_energy_scales = [y_coupling_energy_scale_1,
                            y_coupling_energy_scale_2,
                            y_coupling_energy_scale_3]

print("Constructing an instance of bath.Model.\n")

bath_model = bath.Model(L=3,
                        beta=beta,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales)

print("Print attributes of object:")
print("    L =", bath_model.L)
print("    memory =", bath_model.memory)
print("    y_coupling_energy_scales =", bath_model.y_coupling_energy_scales)
print("    z_coupling_energy_scales =", bath_model.z_coupling_energy_scales)
print("    y_spectral_densities =", bath_model.y_spectral_densities)
print("    z_spectral_densities =", bath_model.z_spectral_densities)
print()
print("\n\n")



# bath.Model test #3.
print("bath.Model test #3")
print("==================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}

y_coupling_energy_scale_1 = Scalar(linear_fn, func_kwargs_1)
y_coupling_energy_scale_2 = Scalar(linear_fn, func_kwargs_2)
y_coupling_energy_scale_3 = const_scalar

y_coupling_energy_scales = [y_coupling_energy_scale_1,
                            y_coupling_energy_scale_2,
                            y_coupling_energy_scale_3]

print("Constructing an instance of bath.Model.\n")

bath_model = bath.Model(L=4,
                        beta=beta,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales)

print("Print attributes of object:")
print("    L =", bath_model.L)
print("    memory =", bath_model.memory)
print("    y_coupling_energy_scales =", bath_model.y_coupling_energy_scales)
print("    z_coupling_energy_scales =", bath_model.z_coupling_energy_scales)
print("    y_spectral_densities =", bath_model.y_spectral_densities)
print("    z_spectral_densities =", bath_model.z_spectral_densities)
print()
print("\n\n")



# bath.Model test #4.
print("bath.Model test #4")
print("==================")

# Component #1 of zero-temperature spectral density of y-noise.
def A_y_r_0T_1_func_form(omega, lambda_y_1, omega_y_1_c):
    return lambda_y_1 * omega * np.exp(-omega / omega_y_1_c)

lambda_y_1 = 3 * 0.1025 / 4
omega_y_1_c = 3 * 10.0 / 4

A_y_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_1_func_form,
                                func_kwargs={"lambda_y_1": lambda_y_1,
                                             "omega_y_1_c": omega_y_1_c},
                                hard_cutoff_freq=40*omega_y_1_c,
                                zero_pt_derivative=lambda_y_1)

# Component #2 of zero-temperature spectral density of y-noise.
def A_y_r_0T_2_func_form(omega, lambda_y_2, omega_y_2_c):
    return lambda_y_2 * omega * np.exp(-omega / omega_y_2_c)

lambda_y_2 = 0.1025 / 4
omega_y_2_c = 10.0 / 4

A_y_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_2_func_form,
                                func_kwargs={"lambda_y_2": lambda_y_2,
                                             "omega_y_2_c": omega_y_2_c},
                                hard_cutoff_freq=40*omega_y_2_c,
                                zero_pt_derivative=lambda_y_2)

A_y_0_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1])
A_y_1_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1, A_y_r_0T_2])
A_y_2_0T = 0

y_spectral_densities_0T = [A_y_0_0T, A_y_1_0T, A_y_2_0T]

print("Constructing an instance of bath.Model.\n")

bath_model = bath.Model(L=3,
                        beta=beta,
                        memory=0.5,
                        y_spectral_densities_0T=y_spectral_densities_0T)

print("Print attributes of object:")
print("    L =", bath_model.L)
print("    memory =", bath_model.memory)
print("    y_coupling_energy_scales =", bath_model.y_coupling_energy_scales)
print("    z_coupling_energy_scales =", bath_model.z_coupling_energy_scales)
print("    y_spectral_densities =", bath_model.y_spectral_densities)
print("    z_spectral_densities =", bath_model.z_spectral_densities)
print()
print("\n\n")



# bath.Model test #5.
print("bath.Model test #5")
print("==================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}

y_coupling_energy_scale_1 = Scalar(linear_fn, func_kwargs_1)
y_coupling_energy_scale_2 = Scalar(linear_fn, func_kwargs_2)
y_coupling_energy_scale_3 = const_scalar

y_coupling_energy_scales = [y_coupling_energy_scale_1,
                            y_coupling_energy_scale_2,
                            y_coupling_energy_scale_3]

# Component #1 of zero-temperature spectral density of y-noise.
def A_y_r_0T_1_func_form(omega, lambda_y_1, omega_y_1_c):
    return lambda_y_1 * omega * np.exp(-omega / omega_y_1_c)

lambda_y_1 = 3 * 0.1025 / 4
omega_y_1_c = 3 * 10.0 / 4

A_y_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_1_func_form,
                                func_kwargs={"lambda_y_1": lambda_y_1,
                                             "omega_y_1_c": omega_y_1_c},
                                hard_cutoff_freq=40*omega_y_1_c,
                                zero_pt_derivative=lambda_y_1)

# Component #2 of zero-temperature spectral density of y-noise.
def A_y_r_0T_2_func_form(omega, lambda_y_2, omega_y_2_c):
    return lambda_y_2 * omega * np.exp(-omega / omega_y_2_c)

lambda_y_2 = 0.1025 / 4
omega_y_2_c = 10.0 / 4

A_y_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_2_func_form,
                                func_kwargs={"lambda_y_2": lambda_y_2,
                                             "omega_y_2_c": omega_y_2_c},
                                hard_cutoff_freq=40*omega_y_2_c,
                                zero_pt_derivative=lambda_y_2)

A_y_0_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1])
A_y_1_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1, A_y_r_0T_2])
A_y_2_0T = 0

y_spectral_densities_0T = [A_y_0_0T, A_y_1_0T, A_y_2_0T]

print("Constructing an instance of bath.Model.\n")

bath_model = bath.Model(L=3,
                        beta=beta,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T)

print("Print attributes of object:")
print("    L =", bath_model.L)
print("    memory =", bath_model.memory)
print("    y_coupling_energy_scales =", bath_model.y_coupling_energy_scales)
print("    z_coupling_energy_scales =", bath_model.z_coupling_energy_scales)
print("    y_spectral_densities =", bath_model.y_spectral_densities)
print("    z_spectral_densities =", bath_model.z_spectral_densities)
print()
print("\n\n")



# bath.Model test #6.
print("bath.Model test #6")
print("==================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

func_kwargs_1 = {"a": 2.0, "b": -1.0}
func_kwargs_2 = {"a": 4.0, "b": -2.0}

y_coupling_energy_scale_1 = Scalar(linear_fn, func_kwargs_1)
y_coupling_energy_scale_2 = Scalar(linear_fn, func_kwargs_2)
y_coupling_energy_scale_3 = const_scalar

y_coupling_energy_scales = [y_coupling_energy_scale_1,
                            y_coupling_energy_scale_2,
                            y_coupling_energy_scale_3]

# Component #1 of zero-temperature spectral density of y-noise.
def A_y_r_0T_1_func_form(omega, lambda_y_1, omega_y_1_c):
    return lambda_y_1 * omega * np.exp(-omega / omega_y_1_c)

lambda_y_1 = 3 * 0.1025 / 4
omega_y_1_c = 3 * 10.0 / 4

A_y_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_1_func_form,
                                func_kwargs={"lambda_y_1": lambda_y_1,
                                             "omega_y_1_c": omega_y_1_c},
                                hard_cutoff_freq=40*omega_y_1_c,
                                zero_pt_derivative=lambda_y_1)

# Component #2 of zero-temperature spectral density of y-noise.
def A_y_r_0T_2_func_form(omega, lambda_y_2, omega_y_2_c):
    return lambda_y_2 * omega * np.exp(-omega / omega_y_2_c)

lambda_y_2 = 0.1025 / 4
omega_y_2_c = 10.0 / 4

A_y_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_2_func_form,
                                func_kwargs={"lambda_y_2": lambda_y_2,
                                             "omega_y_2_c": omega_y_2_c},
                                hard_cutoff_freq=40*omega_y_2_c,
                                zero_pt_derivative=lambda_y_2)

A_y_0_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1])
A_y_1_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1, A_y_r_0T_2])
A_y_2_0T = 0
A_y_3_0T = 0

y_spectral_densities_0T = [A_y_0_0T, A_y_1_0T, A_y_2_0T, A_y_3_0T]

print("Constructing an instance of bath.Model; Expecting an IndexError "
      "exception.\n")

try:
    bath_model = bath.Model(L=3,
                            beta=beta,
                            memory=0.5,
                            y_coupling_energy_scales=y_coupling_energy_scales,
                            y_spectral_densities_0T=y_spectral_densities_0T)
except IndexError as e:
    print(e)
    print("\n\n")



# bath.Model test #7.
print("bath.Model test #7")
print("==================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

y_func_kwargs_1 = {"a": 2.0, "b": -1.0}
y_func_kwargs_2 = {"a": 4.0, "b": -2.0}

y_coupling_energy_scale_1 = Scalar(linear_fn, y_func_kwargs_1)
y_coupling_energy_scale_2 = Scalar(linear_fn, y_func_kwargs_2)
y_coupling_energy_scale_3 = const_scalar

y_coupling_energy_scales = [y_coupling_energy_scale_1,
                            y_coupling_energy_scale_2,
                            y_coupling_energy_scale_3]

z_func_kwargs_2 = {"a": -4.0, "b": 2.0}

z_coupling_energy_scale_1 = const_scalar
z_coupling_energy_scale_2 = Scalar(linear_fn, z_func_kwargs_2)
z_coupling_energy_scale_3 = const_scalar

z_coupling_energy_scales = [z_coupling_energy_scale_1,
                            z_coupling_energy_scale_2,
                            z_coupling_energy_scale_3]

# Component #1 of zero-temperature spectral density of y-noise.
def A_y_r_0T_1_func_form(omega, lambda_y_1, omega_y_1_c):
    return lambda_y_1 * omega * np.exp(-omega / omega_y_1_c)

lambda_y_1 = 3 * 0.1025 / 4
omega_y_1_c = 3 * 10.0 / 4

A_y_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_1_func_form,
                                func_kwargs={"lambda_y_1": lambda_y_1,
                                             "omega_y_1_c": omega_y_1_c},
                                hard_cutoff_freq=40*omega_y_1_c,
                                zero_pt_derivative=lambda_y_1)

# Component #2 of zero-temperature spectral density of y-noise.
def A_y_r_0T_2_func_form(omega, lambda_y_2, omega_y_2_c):
    return lambda_y_2 * omega * np.exp(-omega / omega_y_2_c)

lambda_y_2 = 0.1025 / 4
omega_y_2_c = 10.0 / 4

A_y_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_2_func_form,
                                func_kwargs={"lambda_y_2": lambda_y_2,
                                             "omega_y_2_c": omega_y_2_c},
                                hard_cutoff_freq=40*omega_y_2_c,
                                zero_pt_derivative=lambda_y_2)

A_y_0_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1])
A_y_1_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1, A_y_r_0T_2])
A_y_2_0T = 0

y_spectral_densities_0T = [A_y_0_0T, A_y_1_0T, A_y_2_0T]

print("Constructing an instance of bath.Model.\n")

bath_model = bath.Model(L=3,
                        beta=beta,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T)

print("Print attributes of object:")
print("    L =", bath_model.L)
print("    memory =", bath_model.memory)
print("    y_coupling_energy_scales =", bath_model.y_coupling_energy_scales)
print("    z_coupling_energy_scales =", bath_model.z_coupling_energy_scales)
print("    y_spectral_densities =", bath_model.y_spectral_densities)
print("    z_spectral_densities =", bath_model.z_spectral_densities)
print()
print("\n\n")



# bath.Model test #8.
print("bath.Model test #8")
print("==================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

y_func_kwargs_1 = {"a": 2.0, "b": -1.0}
y_func_kwargs_2 = {"a": 4.0, "b": -2.0}

y_coupling_energy_scale_1 = Scalar(linear_fn, y_func_kwargs_1)
y_coupling_energy_scale_2 = Scalar(linear_fn, y_func_kwargs_2)
y_coupling_energy_scale_3 = const_scalar

y_coupling_energy_scales = [y_coupling_energy_scale_1,
                            y_coupling_energy_scale_2,
                            y_coupling_energy_scale_3]

# Component #1 of zero-temperature spectral density of y-noise.
def A_y_r_0T_1_func_form(omega, lambda_y_1, omega_y_1_c):
    return lambda_y_1 * omega * np.exp(-omega / omega_y_1_c)

lambda_y_1 = 3 * 0.1025 / 4
omega_y_1_c = 3 * 10.0 / 4

A_y_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_1_func_form,
                                func_kwargs={"lambda_y_1": lambda_y_1,
                                             "omega_y_1_c": omega_y_1_c},
                                hard_cutoff_freq=40*omega_y_1_c,
                                zero_pt_derivative=lambda_y_1)

# Component #2 of zero-temperature spectral density of y-noise.
def A_y_r_0T_2_func_form(omega, lambda_y_2, omega_y_2_c):
    return lambda_y_2 * omega * np.exp(-omega / omega_y_2_c)

lambda_y_2 = 0.1025 / 4
omega_y_2_c = 10.0 / 4

A_y_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_2_func_form,
                                func_kwargs={"lambda_y_2": lambda_y_2,
                                             "omega_y_2_c": omega_y_2_c},
                                hard_cutoff_freq=40*omega_y_2_c,
                                zero_pt_derivative=lambda_y_2)

A_y_0_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1])
A_y_1_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1, A_y_r_0T_2])
A_y_2_0T = 0

y_spectral_densities_0T = [A_y_0_0T, A_y_1_0T, A_y_2_0T]

# Component #1 of zero-temperature spectral density of z-noise.
def A_z_r_0T_1_func_form(omega, lambda_z_1, omega_z_1_c):
    return lambda_z_1 * omega * np.exp(-omega / omega_z_1_c)

lambda_z_1 = 3 * 0.1025 / 8
omega_z_1_c = 3 * 10.0 / 8

A_z_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_z_r_0T_1_func_form,
                                func_kwargs={"lambda_z_1": lambda_z_1,
                                             "omega_z_1_c": omega_z_1_c},
                                hard_cutoff_freq=40*omega_z_1_c,
                                zero_pt_derivative=lambda_z_1)

# Component #2 of zero-temperature spectral density of z-noise.
def A_z_r_0T_2_func_form(omega, lambda_z_2, omega_z_2_c):
    return lambda_z_2 * omega * np.exp(-omega / omega_z_2_c)

lambda_z_2 = 0.1025 / 8
omega_z_2_c = 10.0 / 8

A_z_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_z_r_0T_2_func_form,
                                func_kwargs={"lambda_z_2": lambda_z_2,
                                             "omega_z_2_c": omega_z_2_c},
                                hard_cutoff_freq=40*omega_z_2_c,
                                zero_pt_derivative=lambda_z_2)

A_z_0_0T = 0
A_z_1_0T = bath.SpectralDensity0T(cmpnts=[A_z_r_0T_1, A_z_r_0T_2])
A_z_2_0T = 0

z_spectral_densities_0T = [A_z_0_0T, A_z_1_0T, A_z_2_0T]

print("Constructing an instance of bath.Model.\n")

bath_model = bath.Model(L=3,
                        beta=beta,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T,
                        z_spectral_densities_0T=z_spectral_densities_0T)

print("Print attributes of object:")
print("    L =", bath_model.L)
print("    memory =", bath_model.memory)
print("    y_coupling_energy_scales =", bath_model.y_coupling_energy_scales)
print("    z_coupling_energy_scales =", bath_model.z_coupling_energy_scales)
print("    y_spectral_densities =", bath_model.y_spectral_densities)
print("    z_spectral_densities =", bath_model.z_spectral_densities)
print()
print("\n\n")



# bath.Model test #9.
print("bath.Model test #9")
print("==================")

def linear_fn(t, a, b):
    return a*t+b

const_scalar = 2.5

y_func_kwargs_1 = {"a": 2.0, "b": -1.0}
y_func_kwargs_2 = {"a": 4.0, "b": -2.0}

y_coupling_energy_scale_1 = Scalar(linear_fn, y_func_kwargs_1)
y_coupling_energy_scale_2 = Scalar(linear_fn, y_func_kwargs_2)
y_coupling_energy_scale_3 = const_scalar

y_coupling_energy_scales = [y_coupling_energy_scale_1,
                            y_coupling_energy_scale_2,
                            y_coupling_energy_scale_3]

z_func_kwargs_2 = {"a": -4.0, "b": 2.0}

z_coupling_energy_scale_1 = const_scalar
z_coupling_energy_scale_2 = Scalar(linear_fn, z_func_kwargs_2)
z_coupling_energy_scale_3 = const_scalar

z_coupling_energy_scales = [z_coupling_energy_scale_1,
                            z_coupling_energy_scale_2,
                            z_coupling_energy_scale_3]

# Component #1 of zero-temperature spectral density of y-noise.
def A_y_r_0T_1_func_form(omega, lambda_y_1, omega_y_1_c):
    return lambda_y_1 * omega * np.exp(-omega / omega_y_1_c)

lambda_y_1 = 3 * 0.1025 / 4
omega_y_1_c = 3 * 10.0 / 4

A_y_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_1_func_form,
                                func_kwargs={"lambda_y_1": lambda_y_1,
                                             "omega_y_1_c": omega_y_1_c},
                                hard_cutoff_freq=40*omega_y_1_c,
                                zero_pt_derivative=lambda_y_1)

# Component #2 of zero-temperature spectral density of y-noise.
def A_y_r_0T_2_func_form(omega, lambda_y_2, omega_y_2_c):
    return lambda_y_2 * omega * np.exp(-omega / omega_y_2_c)

lambda_y_2 = 0.1025 / 4
omega_y_2_c = 10.0 / 4

A_y_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_y_r_0T_2_func_form,
                                func_kwargs={"lambda_y_2": lambda_y_2,
                                             "omega_y_2_c": omega_y_2_c},
                                hard_cutoff_freq=40*omega_y_2_c,
                                zero_pt_derivative=lambda_y_2)

A_y_0_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1])
A_y_1_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1, A_y_r_0T_2])
A_y_2_0T = 0

y_spectral_densities_0T = [A_y_0_0T, A_y_1_0T, A_y_2_0T]

# Component #1 of zero-temperature spectral density of z-noise.
def A_z_r_0T_1_func_form(omega, lambda_z_1, omega_z_1_c):
    return lambda_z_1 * omega * np.exp(-omega / omega_z_1_c)

lambda_z_1 = 3 * 0.1025 / 8
omega_z_1_c = 3 * 10.0 / 8

A_z_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_z_r_0T_1_func_form,
                                func_kwargs={"lambda_z_1": lambda_z_1,
                                             "omega_z_1_c": omega_z_1_c},
                                hard_cutoff_freq=40*omega_z_1_c,
                                zero_pt_derivative=lambda_z_1)

# Component #2 of zero-temperature spectral density of z-noise.
def A_z_r_0T_2_func_form(omega, lambda_z_2, omega_z_2_c):
    return lambda_z_2 * omega * np.exp(-omega / omega_z_2_c)

lambda_z_2 = 0.1025 / 8
omega_z_2_c = 10.0 / 8

A_z_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_z_r_0T_2_func_form,
                                func_kwargs={"lambda_z_2": lambda_z_2,
                                             "omega_z_2_c": omega_z_2_c},
                                hard_cutoff_freq=40*omega_z_2_c,
                                zero_pt_derivative=lambda_z_2)

A_z_0_0T = 0
A_z_1_0T = bath.SpectralDensity0T(cmpnts=[A_z_r_0T_1, A_z_r_0T_2])
A_z_2_0T = 0

z_spectral_densities_0T = [A_z_0_0T, A_z_1_0T, A_z_2_0T]

print("Constructing an instance of bath.Model.\n")

bath_model = bath.Model(L=3,
                        beta=beta,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T,
                        z_spectral_densities_0T=z_spectral_densities_0T)

print("Print attributes of object:")
print("    L =", bath_model.L)
print("    memory =", bath_model.memory)
print("    y_coupling_energy_scales =", bath_model.y_coupling_energy_scales)
print("    z_coupling_energy_scales =", bath_model.z_coupling_energy_scales)
print("    y_spectral_densities =", bath_model.y_spectral_densities)
print("    z_spectral_densities =", bath_model.z_spectral_densities)
print()

t = 2.0
unformatted_msg = "    Evaluating at site={}: Result={}"

print("Evaluating y-coupling energy scales at t={}:".format(t))
for r, energy_scale in enumerate(bath_model.y_coupling_energy_scales):
    print(unformatted_msg.format(r, energy_scale.eval(t)))
print()

print("Evaluating z-coupling energy scales at t={}:".format(t))
for r, energy_scale in enumerate(bath_model.z_coupling_energy_scales):
    print(unformatted_msg.format(r, energy_scale.eval(t)))
print()

omega = 2.0
unformatted_msg = "    Evaluating at site={}: Result={}"
print("Evaluating spectral densities of y-noise at omega={}:".format(omega))
for r, spectral_density in enumerate(bath_model.y_spectral_densities):
    print(unformatted_msg.format(r, spectral_density.eval(omega)))
print()

print("Evaluating spectral densities of z-noise at omega={}:".format(omega))
for r, spectral_density in enumerate(bath_model.z_spectral_densities):
    print(unformatted_msg.format(r, spectral_density.eval(omega)))
print("\n\n")



# bath._calc_K_tau test #1.
print("bath._calc_K_tau test #1")
print("========================")

print("Evaluating 'bath._calc_K_tau' for several values of the system's memory "
      "'tau' at fixed time step 'dt':")
unformatted_msg = "    Evaluating for (tau, dt)=({}, {}): Result={}"
print(unformatted_msg.format(0.0, 0.1, bath._calc_K_tau(0.0, 0.1)))
print(unformatted_msg.format(0.1, 0.1, bath._calc_K_tau(0.1, 0.1)))
print(unformatted_msg.format(0.175, 0.1, bath._calc_K_tau(0.175, 0.1)))
print(unformatted_msg.format(0.180, 0.1, bath._calc_K_tau(0.180, 0.1)))
print(unformatted_msg.format(3.0, 0.1, bath._calc_K_tau(3.0, 0.1)))
