#!/usr/bin/env python
"""An example script using ``sbc``.

In this example, we implement a simulation of they dynamics of a three-spin
system subject to fixed transverse and longitudinal fields, longitudinal
couplings, and longitudinal ohmic noise (i.e. :math:`z`-noise). The couplings
between the spins and the environment are constant in time. The simulation
tracks the system energy, :math:`\left\langle\hat{\sigma}_{x;
r}(t)\right\rangle`, :math:`\left\langle\hat{\sigma}_{z; r}(t)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{z; r}(t)\hat{\sigma}_{z; r+1}(t)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{y; r}(t)\hat{\sigma}_{z; r+1}(t)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{x; 0}(t) \hat{\sigma}_{z; 1}(t)\hat{\sigma}_{x;
2}(t)\right\rangle`, :math:`\left\langle\hat{\sigma}_{x; 0}(t) \hat{\sigma}_{x;
1}(t)\hat{\sigma}_{x; 2}(t)\right\rangle`, the trace of the system's reduced
density operator, the probability of measuring :math:`\sigma_{z; r}(t)=+1` at
each site :math:`r`, and the probability of measuring :math:`\sigma_{z;
r}(t)=-1` at each site :math:`r`. The simulation also checks whether the system
is entangled using the realignment criterion. 

This script includes the following steps:
1. Set default backend for the tensornetwork library (used to construct and
   manipulate tensor networks).
2. Construct the spin-boson model.
    a) Define all system model parameters.
    b) Define all bath model components.
3. Set the compression parameters for the matrix product states (MPS's) 
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

# For specifying how to compress MPS's.
from sbc import compress

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
# discussion of the system model parameters. The following line of code
# specifies a three-spin system subject to fixed non-uniform transverse and
# longitudinal fields, and longitudinal couplings.
system_model = system.Model(x_fields=[-0.5, -0.49, -0.48],
                            z_fields=[0.0, -0.01, 0.05],
                            zz_couplers=[0.99, 1.05])

# The next step towards constructing our spin-boson model is to specify the bath
# model components. See the documentation for the module :mod:`sbc.bath` for a
# discussion on the bath model components. The bath model components include the
# (inverse) temperature, the system's so-called 'memory', the system-environment
# coupling energy scales, and zero-temperature spectral densities which
# characterize the kind of noise the system is subject to. In this example, we
# consider only noise coupled to the :math:`z^{\mathrm{th}}`- components of the
# spins. The noise at each site is characterized by a spectral density with a
# single ohmic component, and the coupling between the spin and the environment
# is constant in time. The following code specifies the bath model components as
# described above.
def A_z_0T_cmpnt_func_form(omega, mu, omega_UV):
    return mu * omega * np.exp(-omega / omega_UV)

# In this example we assume spatially non-uniform noise to demonstrate how
# ``sbc`` can model more complicated systems.
mu_1 = 0.375
mu_2 = 0.380
mu_3 = 0.370
omega_UV = 40

A_z_1_0T_cmpnt = bath.SpectralDensityCmpnt0T(func_form=A_z_0T_cmpnt_func_form,
                                             func_kwargs={"mu": mu_1,
                                                          "omega_UV": omega_UV},
                                             uv_cutoff=omega_UV,
                                             zero_pt_derivative=mu_1)
A_z_1_0T = bath.SpectralDensity0T(cmpnts=[A_z_1_0T_cmpnt])

A_z_2_0T_cmpnt = bath.SpectralDensityCmpnt0T(func_form=A_z_0T_cmpnt_func_form,
                                             func_kwargs={"mu": mu_2,
                                                          "omega_UV": omega_UV},
                                             uv_cutoff=omega_UV,
                                             zero_pt_derivative=mu_2)
A_z_2_0T = bath.SpectralDensity0T(cmpnts=[A_z_2_0T_cmpnt])

A_z_3_0T_cmpnt = bath.SpectralDensityCmpnt0T(func_form=A_z_0T_cmpnt_func_form,
                                             func_kwargs={"mu": mu_3,
                                                          "omega_UV": omega_UV},
                                             uv_cutoff=omega_UV,
                                             zero_pt_derivative=mu_3)
A_z_3_0T = bath.SpectralDensity0T(cmpnts=[A_z_3_0T_cmpnt])

bath_model = bath.Model(L=system_model.L,  # Number of spins.
                        beta=25,  # Inverse temperature beta=1/(kB*T).
                        memory=2.5,  # The system's memory.
                        z_coupling_energy_scales=[1.0, 1.0, 1.0],
                        z_spectral_densities_0T=[A_z_1_0T, A_z_2_0T, A_z_3_0T])



# Next, we set the compression parameters for the matrix product states (MPS's)
# that span time as well as those that span space. 'Temporal' MPS's are used to
# represent influence functionals/paths, where 'spatial' MPS's are used to
# represent the system's state. See the documentation for the class
# sbc.compress.Params for a description of each available compression parameter.
temporal_compress_params = compress.Params(method="zip-up",
                                           max_num_singular_values=32,
                                           max_trunc_err=1.e-14,
                                           svd_rel_tol=1.e-12,
                                           max_num_var_sweeps=2,
                                           var_rel_tol=1e-8)
spatial_compress_params = compress.Params(method="direct",
                                          max_num_singular_values=16,
                                          max_trunc_err=1.e-14,
                                          svd_rel_tol=1.e-12)



# Next, we set the parameters relating to the tensor network algorithm used to
# simulate the dynamics.
alg_params = alg.Params(dt=0.1,
                        temporal_compress_params=temporal_compress_params,
                        spatial_compress_params=spatial_compress_params)



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
initial_state_nodes = [node] * system_model.L



# Next, we construct an object representing the system state, which also encodes
# the dynamics of the spin system.
system_state = SystemState(system_model,
                           bath_model,
                           alg_params,
                           initial_state_nodes)



# Next, we construct a 'wish-list' of the quantities that you would like to
# report to files as the system evolves. In this example we track several
# quantities. See the documentation for the class :class:`sbc.report.WishList`
# for more details on other quantities that can be tracked.
wish_list = WishList(ev_of_single_site_spin_ops=['sx', 'sz'],
                     ev_of_nn_two_site_spin_ops=[('sz', 'sz'), ('sy', 'sz')],
                     ev_of_multi_site_spin_ops=[('sx', 'sz', 'sx'),
                                                ('sx', 'sx', 'sx')],
                     spin_config_probs=[[1, 1, 1], [-1, -1, -1]],
                     ev_of_energy=True,
                     realignment_criterion=True)



# Next, we specify the output directory to which to report and set the report
# parameters.
output_dir = os.path.dirname(os.path.abspath(__file__)) + "/output"
report_params = ReportParams(wish_list=wish_list, output_dir=output_dir)



# Finally, we run the simulation and periodically report to files.
t = 0.  # Initial time.
t_f = 0.5  # Final time.

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
