#!/usr/bin/env python
"""An example script using ``spinbosonchain``.

In this example, we implement a simulation of the dynamics of a single qubit
subject to time-dependent transverse and longitudinal fields, and environmental
noise that is coupled to the :math:`z`-component of the qubit's spin. The
spectral density of noise comprises of two Dirac delta-like peaks. The
system-environment coupling energy scale is also time-dependent. The simulation
tracks :math:`\left\langle\hat{\sigma}_{x;r}\left(t\right)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{y;r}\left(t\right)\right\rangle`, and
:math:`\left\langle\hat{\sigma}_{z;r}\left(t\right)\right\rangle`. 

This example tests the algorithm implemented in the ``spinbosonchain`` library
that is used to calculate local influence paths for the case where the system is
subject to :math:`z`-noise only, and the system's bath correlation time, or
"memory", is smaller than the total simulation time.

If ``matplotlib`` has been installed, then the results obtained by
``spinbosonchain`` are then compared to the expected results derived in
Ref. [Albash]_.

This script includes the following steps:
1. Set default backend for the tensornetwork library (used to construct and
   manipulate tensor networks).
2. Construct the spin-boson model.
    a) Define all system model parameters.
    b) Define all bath model components.
3. Set the compression parameters for the matrix product states (MPS's) 
   representing the system state and the local influence functionals.
4. Set parameters related to the tensor network algorithm.
5. Construct a MPS representing the initial state of the qubit system.
6. Construct an object representing the system state, which also encodes the
   dynamics of the qubit system.
7. Construct a 'wish-list' of the quantities that you would like to report to 
   files as the system evolve.
8. Set the output directory to which to report.
9. Run simulation and periodically report to files.
11. If ``matplotlib`` has been installed, then generate comparison plots.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# To determine the current working directory and to set # threads to use.
import os

# To check whether the matplotlib library has been installed.
from importlib import util

# To terminate the script before reaching its end.
import sys

# For making directories.
import pathlib



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

# Import the matplotlib library if it exists.
if util.find_spec("matplotlib") is not None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt



# Assign an alias to the ``spinbosonchain`` library.
import spinbosonchain as sbc



############################
## Authorship information ##
############################

__author__     = "D-Wave Systems Inc."
__copyright__  = "Copyright 2021"
__credits__    = ["Matthew Fitzpatrick"]
__maintainer__ = "D-Wave Systems Inc."
__email__      = "support@dwavesys.com"
__status__     = "Development"



#########################
## Main body of script ##
#########################

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
# parameters. See the documentation for the module :mod:`spinbosonchain.system`
# for a discussion of the system model parameters. The following line of code
# specifies the system model parameters.
system_model = sbc.system.Model(x_fields=[-0.5])

# The next step towards constructing our spin-boson model is to specify the bath
# model components. See the documentation for the module
# :mod:`spinbosonchain.bath` for a discussion on the bath model components. The
# bath model components include the (inverse) temperature, the system's
# so-called 'memory', the system-environment coupling energy scales, and the
# zero-temperature spectral densities which characterize the kind of noise the
# system is subject to. The following code specifies the bath model components.
def A_z_r_0T_cmpnt_func_form(omega, eta, omega_c):
    return eta * omega * np.exp(-omega / omega_c)

eta = 6.11e-4
omega_c = 2e3

cmpnt = sbc.bath.SpectralDensityCmpnt0T(func_form=A_z_r_0T_cmpnt_func_form,
                                        func_kwargs={"eta": eta,
                                                     "omega_c": omega_c},
                                        uv_cutoff=40*omega_c)
cmpnts = [cmpnt]
A_z_r_0T = sbc.bath.SpectralDensity0T(cmpnts=cmpnts)

# ``memory`` refers to the system memory. For ohmic noise, the higher the
# cutoff frequency, the less ``memory`` is required. In this example, the
# dynamics are essentially Markovian, therefore the ``memory`` ought to be set
# to approximately the time step size, ``dt``.
L = 1  # System size.
memory = 0.02  # Twice the value of the time step size used in this simulation.
beta = 5e-3  # Inverse temperature beta=1/(kB*T).
ctor_kwargs = {"L": L,
               "beta": beta,
               "memory": memory,
               "z_coupling_energy_scales": [1],
               "z_spectral_densities_0T": [A_z_r_0T]}
bath_model = sbc.bath.Model(**ctor_kwargs)



# Next, we set the compression parameters for the matrix product states (MPS's)
# that span time as well as those that span space. 'Temporal' MPS's are used to
# represent influence functionals/paths, where 'spatial' MPS's are used to
# represent the system's state. See the documentation for the class
# spinbosonchain.compress.Params for a description of each available compression
# parameter. Note that since this is a single qubit system, no compression is
# performed on the spatial MPS's as they are simply scalars: there are no
# spatial bonds in the MPS sense.
temporal_compress_params = sbc.compress.Params(method="direct",
                                               max_num_singular_values=16,
                                               max_trunc_err=1.e-14)
spatial_compress_params = sbc.compress.Params()



# Next, we set the parameters relating to the tensor network algorithm used to
# simulate the dynamics.
dt = 0.01  # Time step size.
alg_params = sbc.alg.Params(dt=dt,
                            temporal_compress_params=temporal_compress_params,
                            spatial_compress_params=spatial_compress_params)



# Next, we construct a MPS representing the initial state of the qubit
# system. In this example, we assume that the initial state is
# |+x>. Accordingly, we construct a `tensornetwork` node to represent this
# state. The underlying tensor has the shape (1, 2, 1), where the first and last
# dimensions are the trivial bond dimensions of the 1-node MPS representing the
# initial state, and the second dimension is the physical dimension, which is
# equal to 2 because qubits have two physical degrees of freedom. Let `q` be the
# 'physical index' of some MPS node representing a single qubit, i.e. let it be
# the second index of the MPS node. In `spinbosonchain`, `q=0` corresponds to a
# `sz=1` state, and `q=1` corresponds to a `sz=-1` state. The following code
# specifies the initial state.
tensor = np.zeros([1, 2, 1], dtype=np.complex128)
tensor[0, 0, 0] = 1 / np.sqrt(2)  # sz=1 amplitude.
tensor[0, 1, 0] = 1 / np.sqrt(2)  # sz=-1 amplitude.
node = tn.Node(tensor)
initial_state_nodes = [node] * L



# Next, we construct an object representing the system state, which also encodes
# the dynamics of the qubit system.
system_state = sbc.state.SystemState(system_model,
                                     bath_model,
                                     alg_params,
                                     initial_state_nodes)



# Next, we construct a 'wish-list' of the quantities that you would like to
# report to files as the system evolves. In the case of single-qubit systems, by
# tracking the expectation values of the x-, y-, and z-Pauli spin matrices, we
# track all the information encoded in the system's reduced density matrix.
# Hence, in this example we track the three aforementioned quantities. See the
# documentation for the class :class:`spinbosonchain.report.WishList` for more
# details on other quantities that can be tracked.
wish_list = sbc.report.WishList(ev_of_single_site_spin_ops=['sx', 'sy', 'sz'])



# Next, we specify the output directory to which to report and set the report
# parameters.
sbc_output_dir = os.path.dirname(os.path.abspath(__file__)) + "/sbc-output"
report_params = sbc.report.ReportParams(wish_list=wish_list,
                                        output_dir=sbc_output_dir)



# Finally, we run the simulation and periodically report to files.
t = 0.  # Initial time.
t_f = 25  # Total simulation time.
print("Running simulation using ``spinbosonchain``...")
sbc.report.report(system_state, report_params)
print("t = {}".format(t))
while t < abs(t_f-0.5*dt):
    system_state.evolve(num_steps=50)
    t = system_state.t
    sbc.report.report(system_state, report_params)
    print("t = {}".format(t))
print()
print("Simulation has finished successfully: "
      "output has been written to the directory ``" + sbc_output_dir + "``.")



# Generate plots comparing the results obtained by ``spinbosonchain`` and
# that obtained by Albash and Lidar (i.e. the expected result).
if util.find_spec("matplotlib") is None:
    print()
    print("``matplotlib`` has not been installed, therefore comparison plots "
          "will not be generated. See file "
          "``<git-repo-root>/docs/INSTALL.rst`` for instructions on how to "
          "install ``matplotlib``.")
    sys.exit()
    
comparison_plot_output_dir = \
    os.path.dirname(os.path.abspath(__file__)) + "/comparison-plot-output"
pathlib.Path(comparison_plot_output_dir).mkdir(parents=True, exist_ok=True)

r_y_label = (r"$r_{\mathbf{a}\left(t\right)}$")
theta_y_label = (r"$\theta_{\mathbf{a}\left(t\right)}$")
phi_y_label = (r"$\phi_{\mathbf{a}\left(t\right)}$")

obs = dict()
for op_string in ("sx", "sy", "sz"):
    filename = sbc_output_dir + "/" + op_string + ".csv"
    with open(filename, 'rb') as file_obj:
        data = np.loadtxt(file_obj, skiprows=1, delimiter=";", dtype=float)
    obs[op_string] = data[:, 1]
times = data[:, 0]

r_sbc = np.sqrt(obs["sx"]*obs["sx"] + obs["sy"]*obs["sy"] + obs["sz"]*obs["sz"])
theta_sbc = np.arctan2(np.sqrt(obs["sx"]*obs["sx"] + obs["sy"]*obs["sy"]),
                       obs["sz"])
phi_sbc = np.arctan2(obs["sy"], obs["sx"])

A_z_r_T = bath_model.z_spectral_densities[0]
T_2_c = 1 / 2 / A_z_r_T.eval(0)
r_expected = np.exp(-times / T_2_c)

theta_expected = 0.5 * np.pi * np.ones([times.size])
phi_expected = np.zeros([times.size])

base_plot_filenames_vs_plot_specs_map = \
    {"/r_a.pdf": (r_sbc, r_expected, r_y_label, None, None),
     "/theta_a.pdf": (theta_sbc, theta_expected, theta_y_label, 0, np.pi),
     "/phi_a.pdf": (phi_sbc, phi_expected, phi_y_label, -np.pi/2, np.pi/2)}

for key, val in base_plot_filenames_vs_plot_specs_map.items():
    base_plot_filename = key
    y_label = val[2]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_filename = comparison_plot_output_dir + base_plot_filename
    x_label = r"$t$"
    linewidth = 3
    markersize = 15
    legend_ft_size = 13
    xy_label_ft_size = 13
    tick_label_ft_size = 13
    major_tick_len=8
    minor_tick_len=5
    legend_labels = ["spinbosonchain", "expected result"]

    x = times
    y = val[0]
    ax.scatter(x, y, s=markersize, marker='s', label=legend_labels[0])

    x = times
    y = val[1]
    ax.scatter(x, y, s=markersize, marker='d', label=legend_labels[1])

    ax.legend(loc='best', frameon=True, fontsize=legend_ft_size)
    ax.set_xlabel(x_label, fontsize=xy_label_ft_size)
    ax.set_ylabel(y_label, fontsize=xy_label_ft_size)

    y_min = val[3]
    y_max = val[4]
    ax.set_ylim(y_min, y_max)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    ax.tick_params(axis='x', which='major',
                   labelsize=tick_label_ft_size,
                   width=2.0 * linewidth / 3.0,
                   length=major_tick_len, direction='in')
    ax.tick_params(axis='x', which='minor',
                   width=2.0 * linewidth / 3.0,
                   length=minor_tick_len, direction='in')

    ax.tick_params(axis='y', which='major',
                   labelsize=tick_label_ft_size,
                   width=2.0 * linewidth / 3.0,
                   length=major_tick_len, direction='in')
    ax.tick_params(axis='y', which='minor',
                   width=2.0 * linewidth / 3.0,
                   length=minor_tick_len, direction='in')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0 * linewidth / 3.0)

    plt.minorticks_on()

    fig.tight_layout(pad=1.08)
    plt.savefig(plot_filename, format='pdf')

print()
print("Comparison plots have been saved in the directory ``"
      + comparison_plot_output_dir + "``.")



# Final note: when looking at the spreadsheets of data, make sure to deselect
# the comma "," as a delimiter, otherwise the spreadsheet headers will not
# appear as they were intended.
