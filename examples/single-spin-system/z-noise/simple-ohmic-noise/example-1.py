#!/usr/bin/env python
"""An example script using ``sbc``.

In this example, we implement a simulation of the dynamics of a single spin
subject to a fixed transverse field, and longitudinal noise (i.e. 
:math:`z`-noise) with a zero-temperature spectral density comprising of a single
Ohmic component. The coupling between the spin and the environment is constant
in time. The simulation tracks the expectation value of 
:math:`\hat{\sigma}_{z; r=0}`.

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

# To determine the current working directory.
import os



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
# The choices are either tensorflow or numpy. Numpy is used by default. If
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
# discussion of the system model parameters. The following line of code
# specifies a single-spin system subject only to a transverse field that is
# time-independent.
system_model = system.Model(x_fields=[0.5])

# The next step towards constructing our spin-boson model is to specify the
# bath model components. See the documentation for the module :mod:`sbc.bath`
# for a discussion on the bath model components. The bath model components
# include the (inverse) temperature, the system's so-called 'memory', the
# system-environment coupling energy scales, and zero-temperature spectral
# densities which characterize the kind of noise the system is subject to. In
# this example, we consider only noise coupled to the :math:`z^{\mathrm{th}}`-
# component of the spin. The noise is characterized by a zero-temperature
# spectral density with a single Ohmic component, and the coupling between the
# spin and the environment is constant in time. The following code specifies
# the bath model components as described above.
def A_z_0T_cmpnt_func_form(omega, alpha, omega_c):
    return np.pi * alpha * omega * np.exp(-omega / omega_c)

alpha = 0.1
omega_c = 5

A_z_0T_cmpnt = bath.SpectralDensityCmpnt0T(func_form=A_z_0T_cmpnt_func_form,
                                           func_kwargs={"alpha": alpha,
                                                        "omega_c": omega_c},
                                           hard_cutoff_freq=40*omega_c,
                                           zero_pt_derivative=np.pi*alpha)

A_z_0T = bath.SpectralDensity0T(cmpnts=[A_z_0T_cmpnt])

bath_model = bath.Model(L=1,  # Number of spins.
                        beta=1000,  # Inverse temperature beta=1/(kB*T).
                        memory=1.0,  # The system's memory.
                        z_coupling_energy_scales=[1.0],  # Constant coupling.
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
state_trunc_params = trunc.Params(max_num_singular_values=1)  # B/c single spin.



# Next, we set the parameters relating to the tensor network algorithm used to
# simulate the dynamics.
alg_params = alg.Params(dt=0.2,
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
tensor[0, 0, 0] = 1  # Set sz=1 amplitude to 1; sz=-1 amplitude set to 0 above.
node = tn.Node(tensor)
initial_state_nodes = [node]  # Recall we a constructing a 1-qubit spin system.



# Next, we construct an object representing the system state, which also encodes
# the dynamics of the spin system.
system_state = SystemState(system_model,
                           bath_model,
                           alg_params,
                           initial_state_nodes)



# Next, we construct a 'wish-list' of the quantities that you would like to
# report to files as the system evolves. In this example we are only interested
# in tracking the expectation value of the 'z'-Pauli spin operator, which we
# denote by the string `'sz'`. See the documentation for the class
# :class:`sbc.report.WishList` for more details on other quantities that can
# be tracked.
wish_list = WishList(ev_of_single_site_spin_ops=['sz'])



# Next, we specify the output directory to which to report and set the report
# parameters.
output_dir = os.path.dirname(os.path.abspath(__file__)) + "/output"
report_params = ReportParams(wish_list=wish_list, output_dir=output_dir)



# Finally, we run the simulation and periodically report to files.
t = 0.  # Initial time.
t_f = 3.  # Final time.

print("Report at t = {}".format(t))
report(system_state, report_params)
while t < t_f:
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
