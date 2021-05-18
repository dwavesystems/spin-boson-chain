#!/usr/bin/env python
"""An example script using ``sbc``.

In this example, we implement a simulation of the dynamics of a single
flux-qubit subject to quantum annealing, charge noise (modelled by
:math:`y`-noise), and hybrid flux noise (modelled by :math:`z`-noise). The
annealing schedule is taken from a spreadsheet. The simulation tracks the
expectation values of :math:`\hat{\sigma}_{x; r=0}`, :math:`\hat{\sigma}_{y;
r=0}`, and :math:`\hat{\sigma}_{z; r=0}`.

Note that simulating :math:`y`-noise is more computationally expensive than
simulating only :math:`z`-noise. In the former case, there are roughly three
times as many Trotterization slices in the QUAPI algorithm compared to the
latter case. For Markovian noise, this translates to roughly a tripling of the
simulation time, whereas for non-Markovian noise the worst case scenario is
roughly nine times slower compared to the :math:`z`-noise algorithm.

This script includes the following steps:
1. Set default backend for the tensornetwork library (used to construct and
   manipulate tensor networks).
2. Construct the spin-boson model.
    a) Define all system model parameters.
    b) Define all bath model components.
3. Set the truncation parameters for the matrix product states (MPS's) 
   representing the system state and the local influence functionals.
4. Set parameters related to the tensor network algorithm.
5. Construct a MPS representing the initial state of the spin system.
6. Construct an object representing the system state, which also encodes the
   dynamics of the spin system.
7. Construct a 'wish-list' of the quantities that you would like to report to 
   files as the system evolve.
8. Set the output directory to which to report.
9. Run simulation and periodically report to files.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For timing the execution of the main body of the script.
import time

# To determine the current working directory and to set # threads to use.
import os
from os.path import dirname



# Set the number of threads for different kinds of operations.

# The total number of threads to use.
total_num_threads_to_use = 2

# Used explicitly if tensorflow is set as the backend for the tensornetwork
# library. The parameter below specifies the number of threads in a thread pool
# used by tensorflow to execute independent tensor operations in parallel. If
# numpy is used instead as the backend, then the parameter below should be set
# to 1.
num_inter_op_threads = 1

# Used for both numpy and tensorflow. The parameter below specifies the number
# of threads in a thread pool used to execute an operation that can be
# parallelized internally, e.g. matrix multiplication.
num_intra_op_threads = total_num_threads_to_use // num_inter_op_threads

# The following need to be set before importing numpy or tensorflow.
os.environ["OMP_NUM_THREADS"] = str(num_intra_op_threads)
os.environ["MKL_NUM_THREADS"] = str(num_intra_op_threads)



# For evaluating special math functions.
import numpy as np

# For generating interpolation functions from data.
from scipy.interpolate import interp1d

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# For creating objects representing time-dependent scalars.
from sbc import scalar

# For specifying the system model parameters.
from sbc import system

# For specifying the bath model components.
from sbc import bath

# For specifying how to truncate Schmidt spectra in MPS compression.
from sbc import trunc

# For specifying parameters related to the tensor network algorithm.
from sbc import alg

# For creating a system-state object.
from sbc.state import SystemState

# For reporting the instantaneous values of certain quantities during a
# simulation.
from sbc.report import WishList, ReportParams, report



############################
## Authorship information ##
############################

__author__     = "Matthew Fitzpatrick"
__copyright__  = "Copyright 2021"
__credits__    = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__      = "mfitzpatrick@dwavesys.com"
__status__     = "Development"



#########################
## Main body of script ##
#########################

# Begin timer and create a job output object.
start_time = time.time()



# Set the default backend for constructing and manipulating tensor networks.
# The choices are either tensorflow or numpy. numpy is used by default. If
# using tensorflow, please ensure that it is installed.
use_tensorflow_as_backend = False
if use_tensorflow_as_backend:
    import tensorflow 
    tensorflow.config.threading.set_intra_op_parallelism_threads(
        num_inter_op_threads)
    tensorflow.config.threading.set_inter_op_parallelism_threads(
        num_intra_op_threads)
    tn.set_default_backend('tensorflow')



# We need to construct the spin-boson model. In doing so, we must construct
# several other objects. The first step is to define all the system model
# parameters. See the documentation for the module :mod:`sbc.system` for a
# discussion of the system model parameters. The following lines of code
# specifies a single-spin system subject to both time-dependent transverse
# and longitudinal fields.
def normalized_annealing_fraction(t, t_a):
    return t / t_a

# Load annealing schedule from a spreadsheet.
filename = (dirname(dirname(dirname(os.path.abspath(__file__))))
            + "/annealing-schedule.csv")
with open(filename, 'rb', 1) as file_obj:
    data = np.loadtxt(file_obj, skiprows=1, delimiter=",", dtype=np.float)
    s_array = data[:, 0]
    A_over_h_array = data[:, 1]  # Units of GHz; h is Planck's constant.
    B_over_h_array = data[:, 2]  # Units of GHz.
    max_B_over_h = np.amax(B_over_h_array)  # Units of GHz.
    A_array = A_over_h_array / max_B_over_h
    B_array = B_over_h_array / max_B_over_h 
    A = interp1d(s_array, A_array, kind="cubic", fill_value="extrapolate")
    B = interp1d(s_array, B_array, kind="cubic", fill_value="extrapolate")

# Note that all energy quantities are in units of max(B_over_h_array * h);
# frequencies in units of max(B_over_h_array * h / hbar); and time in units of
# max(B_over_h_array * h / hbar)**(-1).

def h_x_func_form(t, t_a, A):
    s = normalized_annealing_fraction(t, t_a)
    return -A(s) / 2

def h_z_func_form(t, t_a, h_0, B):
    s = normalized_annealing_fraction(t, t_a)
    return B(s) * h_0 / 2

h_0 = -0.3
t_a = 1
h_x_func_kwargs = {"t_a": t_a, "A": A}
h_z_func_kwargs = {"t_a": t_a, "h_0": h_0, "B": B}

h_x = scalar.Scalar(h_x_func_form, h_x_func_kwargs)
h_z = scalar.Scalar(h_z_func_form, h_z_func_kwargs)

system_model = system.Model(x_fields=[h_x], z_fields=[h_z])

# The next step towards constructing our spin-boson model is to specify the bath
# model components. See the documentation for the module :mod:`sbc.bath` for a
# discussion on the bath model components. The bath model components include the
# (inverse) temperature, the system's so-called 'memory', the system-environment
# coupling energy scales, and zero-temperature spectral densities which
# characterize the kind of noise the system is subject to. In this example, we
# consider charge noise (modelled by y-noise), and hybrid flux noise (modelled
# by z-noise), which we breakdown into a 1/f component, and an ohmic
# component. The couplings between the y- and z- components of the spin to the
# environment dependent on the energy scales A(s) and B(s). The following code
# specifies the bath model components as described above.

# Constructs zero-temperature spectral density functions.
def A_y_0T_func_form(omega, mu_y, omega_UV_y, beta):
    return mu_y * np.tanh(beta * omega) * np.exp(-omega / omega_UV_y)

def A_z_0T_LF_func_form(omega, mu_z_LF, omega_IR_z_LF, omega_UV_z_LF):
    # Note that this function encodes a smooth IR cutoff such that the 1/f
    # spectral density is ohmic in the neighbourhood of omega=0. This avoids a
    # divergence associated with the 1/f spectral density at omega=0.
    phi = 10 * (omega-omega_IR_z_LF) / omega_IR_z_LF
    nu = 0.5 * (np.tanh(phi) + 1)
    eta = mu_z_LF * np.power(omega / omega_IR_z_LF, -nu)
    
    return eta * omega * np.exp(-omega / omega_UV_z_LF)

def A_z_0T_HF_func_form(omega, mu_z_HF, omega_UV_z_HF):
    return mu_z_HF * omega * np.exp(-omega / omega_UV_z_HF)

mu_y = 2.5e-3
mu_z_LF = 1.75e5
mu_z_HF = 0.375
omega_IR_z_LF = 3.5e-7
omega_UV_y = 30
omega_UV_z_LF = 3.5e-4
omega_UV_z_HF = 30
beta = 50

func_kwargs = {"mu_y": mu_y, "omega_UV_y": omega_UV_y, "beta": beta}
A_y_0T_cmpnt = bath.SpectralDensityCmpnt0T(func_form=A_y_0T_func_form,
                                           func_kwargs=func_kwargs,
                                           uv_cutoff=40*omega_UV_y,
                                           zero_pt_derivative=mu_y*beta)
A_y_0T = bath.SpectralDensity0T(cmpnts=[A_y_0T_cmpnt])

func_kwargs = {"mu_z_LF": mu_z_LF, 
               "omega_IR_z_LF": omega_IR_z_LF,
               "omega_UV_z_LF": omega_UV_z_LF}
A_z_0T_LF = bath.SpectralDensityCmpnt0T(func_form=A_z_0T_LF_func_form,
                                        func_kwargs=func_kwargs,
                                        uv_cutoff=40*omega_UV_z_LF,
                                        zero_pt_derivative=mu_z_LF)
func_kwargs = {"mu_z_HF": mu_z_HF, "omega_UV_z_HF": omega_UV_z_HF}
A_z_0T_HF = bath.SpectralDensityCmpnt0T(func_form=A_z_0T_HF_func_form,
                                        func_kwargs=func_kwargs,
                                        uv_cutoff=40*omega_UV_z_HF,
                                        zero_pt_derivative=mu_z_HF)
A_z_0T = bath.SpectralDensity0T(cmpnts=[A_z_0T_LF, A_z_0T_HF])

# Specify system-bath coupling energy scales.
def E_y_lambda_func_form(t, t_a):
    s = normalized_annealing_fraction(t, t_a)
    return np.sqrt(A(s)) / 2
E_y_lambda_func_kwargs = {"t_a": t_a}
E_y_lambda = scalar.Scalar(E_y_lambda_func_form, E_y_lambda_func_kwargs)

def E_z_lambda_func_form(t, t_a):
    s = normalized_annealing_fraction(t, t_a)
    return np.sqrt(B(s)) / 2
E_z_lambda_func_kwargs = {"t_a": t_a}
E_z_lambda = scalar.Scalar(E_z_lambda_func_form, E_z_lambda_func_kwargs)

bath_model = bath.Model(L=1,  # Number of spins.
                        beta=beta,  # Inverse temperature beta=1/(kB*T).
                        memory=t_a,  # The system's memory.
                        y_coupling_energy_scales=[E_y_lambda],
                        y_spectral_densities_0T=[A_y_0T],
                        z_coupling_energy_scales=[E_z_lambda],
                        z_spectral_densities_0T=[A_z_0T])



# Next, we set the truncation parameters for the matrix product states (MPS's)
# representing the system state and the local path functionals. In short,
# increasing and decreasing the parameters `max_num_singular_values` and
# `max_trunc_err` respectively translate to MPS's with larger bond dimensions
# `chi` which generally translates to a decrease in numerical errors in the
# simulation. However, since the most computationally intensive parts of the
# simulation scale like `chi^3`, increasing and decreasing the parameters
# `max_num_singular_values` and `max_trunc_err` respectively lead to longer
# runtimes.
influence_trunc_params = trunc.Params(max_num_singular_values=64,
                                      max_trunc_err=1.e-14)
state_trunc_params = trunc.Params(max_num_singular_values=1)  # b/c single spin.



# Next, we set the parameters relating to the tensor network algorithm used to
# simulate the dynamics.
alg_params = alg.Params(dt=0.1,
                        influence_trunc_params=influence_trunc_params,
                        state_trunc_params=state_trunc_params)



# Next, we construct a MPS representing the initial state of the spin system. In
# this example, the 1-qubit spin system is initialized in the sz=1 (i.e. up)
# state. Below we construct a `tensornetwork` node to represent the initial
# state. The underlying tensor has the shape (1, 2, 1), where the first and last
# dimensions are the trivial bond dimensions of the 1-node MPS representing the
# initial state, and the second dimension is the physical dimension, which is
# equal to 2 because spins have two physical degrees of freedom. Let `q` be the
# 'physical index' of some MPS node representing a single spin, i.e. let it be
# the second index of the MPS node. In `sbc`, `q=0` corresponds to a `sz=1`
# state, and `q=1` corresponds to a `sz=-1` state. The following code specifies
# the initial state.
tensor = np.zeros([1, 2, 1], dtype=np.complex128)
tensor[0, 0, 0] = 1 / np.sqrt(2)  # sz=1 amplitude.
tensor[0, 1, 0] = 1 / np.sqrt(2)  # sz=-1 amplitude.
node = tn.Node(tensor)
initial_state_nodes = [node]  # Recall we a constructing a 1-qubit spin system.



# Next, we construct an object representing the system state, which also encodes
# the dynamics of the spin system.
system_state = SystemState(system_model,
                           bath_model,
                           alg_params,
                           initial_state_nodes)



# Next, we construct a 'wish-list' of the quantities that you would like to
# report to files as the system evolves. In the case of single-qubit systems, by
# tracking the expectation values of the x-, y-, and z-Pauli spin matrices, we
# track all the information encoded in the system's reduced density matrix.
# Hence, in this example we track the three aforementioned quantities. See the
# documentation for the class :class:`sbc.report.WishList` for more details on
# other quantities that can be tracked.
wish_list = WishList(ev_of_single_site_spin_ops=['sx', 'sy', 'sz'])



# Next, we specify the output directory to which to report and set the report
# parameters.
output_dir = dirname(os.path.abspath(__file__)) + "/output"
report_params = ReportParams(wish_list=wish_list, output_dir=output_dir)



# Finally, we run the simulation and periodically report to files.
t = 0.  # Initial time.

print("Report at t = {}".format(t))
report(system_state, report_params)
while t < t_a:
    system_state.evolve(num_steps=1)
    t = system_state.t
    report(system_state, report_params)
    print("Report at t = {}".format(t))



# End timer and print total execution time.
execution_time = time.time() - start_time
print()
print("Finished script: execution time = %.13f" % (execution_time))

    

# Final note: when looking at the spreadsheets of data, make sure to deselect
# the comma "," as a delimiter, otherwise the spreadsheet headers will not
# appear as they were intended.
