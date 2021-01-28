#!/usr/bin/env python

#####################################
## Load libraries/packages/modules ##
#####################################

# For evaluating special math functions.
import numpy as np



# Import class representing time-dependent scalar model parameters.
from sbc.scalar import Scalar

# For specifying bath model components.
from sbc import bath

# Class to test.
from sbc._influence.eta import Eta



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

# _influence.eta.Eta test #1.
print("_influence.eta.Eta test #1")
print("==========================")

# Need to construct a ``bath.Model`` object. In order to do this, we need to a
# few more objects. Starting the the coupling energy scales. We'll assume zero
# noise here.
print("Constructing an instance of ``bath.Model``.\n")

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

# Next we need to construct the spectral-densities of noise. Below we define 
# component #1 of the zero-temperature spectral density of y-noise.
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

# Now we can construct the ``bath.Model`` object.
bath_model = bath.Model(L=3,
                        beta=1.0,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T)

print("Constructing various instances of ``_influence.eta.Eta`` to evaluate "
      "the eta-function for a given set of bath model components for various "
      "(r, l1, l2, n, spin_basis):\n")

dt = 0.1
unformatted_msg = ("    Evaluating for (r, l1, l2, n, spin_basis)="
                   "({}, {}, {}, {}, {}): Result={}")
for spin_basis in ("y", "z"):
    for r in range(bath_model.L):
        eta = Eta(r, bath_model, dt, spin_basis)
        for n in range(1, 11):
            for l1 in range(2*n+2):
                for l2 in range(l1+1):
                    result = eta.eval(l1, l2, n)
                    msg = unformatted_msg.format(r, l1, l2, n,
                                                 spin_basis, result)
                    print(msg)
