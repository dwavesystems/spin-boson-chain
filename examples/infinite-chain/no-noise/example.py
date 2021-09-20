#!/usr/bin/env python
"""An example script using ``spinbosonchain``.

In this example, we implement a simulation of the dynamics of an infinite
ferromagnetic qubit chain subject to time-dependent transverse and longitudinal
fields, and no environmental noise. The simulation tracks the expectation value
of the system's energy per unit cell :math:`\left\langle
E\left(t\right)\right\rangle`, the first three correlation lengths [see the
documentation for the attribute
:attr:`spinbosonchain.state.SystemState.correlation_lengths` for a discussion on
correlation lengths], :math:`\left\langle
\hat{\sigma}_{x;0}\left(t\right)\right\rangle`, :math:`\left\langle
\hat{\sigma}_{z;0}\left(t\right) \hat{\sigma}_{z;1}\left(t\right)\right\rangle`,
and the probability of measuring :math:`\sigma_{z;r}\left(t\right)=+1` at sites
:math:`r=0` and :math:`r=1`, where :math:`r=0` is the center spin site of the
infinite chain. We also show how one can periodically backup the simulation data
to file in case of a crash and how to recover a simulation.

This example tests the algorithm implemented in the ``spinbosonchain`` library
that is used to calculate the dynamics of an infinite system subject to no
noise.

If ``tenpy`` has been installed, then the same quantities will also be
calculated by the time evolving block decimation (TEBD) method for comparison
sake, except for the correlation lengths and the spin configuration
probabilities.

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
9. Run simulation, periodically report to files, and periodically backup
   simulation data to file.
10. Rerun the simulation using TEBD if ``tenpy`` has been installed.
11. Generate comparison plots if ``matplotlib`` has been installed.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# To determine the current working directory and to set # threads to use.
import os

# To check whether the tenpy and matplotlib libraries have been installed.
from importlib import util

# To terminate the script before reaching its end.
import sys

# For making directories.
import pathlib

# To suppress the warnings generated from ``tenpy`` (if installed).
import warnings



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

# Import the tenpy library if it exists.
if util.find_spec("tenpy") is not None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tenpy

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
def s(t, t_a):
    return t / t_a

def A(t, t_a):
    return np.abs(2 * (1 - s(t, t_a)))

def B(t, t_a):
    return np.abs(2 * s(t, t_a))

def x_field_func_form(t, t_a):
    return -A(t, t_a) / 2

def z_field_func_form(t, t_a, h_r):
    return B(t, t_a) * h_r / 2

def zz_coupler_func_form(t, t_a, J_r):
    return B(t, t_a) * J_r / 2

L = 1  # Number of qubits.
t_a = 2  # Total simulation time.
h = 1/5
J = -1
x_fields = [sbc.scalar.Scalar(x_field_func_form, {"t_a": t_a})]
z_fields = [sbc.scalar.Scalar(z_field_func_form, {"t_a": t_a, "h_r": h})]
zz_couplers = [sbc.scalar.Scalar(zz_coupler_func_form, {"t_a": t_a, "J_r": J})]
system_model = sbc.system.Model(x_fields=x_fields,
                                z_fields=z_fields,
                                zz_couplers=zz_couplers,
                                is_infinite=True)

# The next step towards constructing our spin-boson model is to specify the bath
# model components. See the documentation for the module
# :mod:`spinbosonchain.bath` for a discussion on the bath model components.
# Since in this case there is no noise, the specification of the bath model
# components is trivial.

# ``memory`` refers to the system memory. Since there is no noise, no memory is
# required.
bath_model = sbc.bath.Model(L=L, beta=np.inf, memory=0)



# Next, we set the compression parameters for the matrix product states (MPS's)
# that span time as well as those that span space. 'Temporal' MPS's are used to
# represent influence functionals/paths, where 'spatial' MPS's are used to
# represent the system's state. See the documentation for the class
# spinbosonchain.compress.Params for a description of each available compression
# parameter. Note that since this is a single qubit system, no compression is
# performed on the spatial MPS's as they are simply scalars: there are no
# spatial bonds in the MPS sense.
temporal_compress_params = sbc.compress.Params(method="zip-up",
                                               max_num_singular_values=4,
                                               max_trunc_err=1.e-14,
                                               svd_rel_tol=1.e-12,
                                               max_num_var_sweeps=2,
                                               var_rel_tol=1e-8)
spatial_compress_params = sbc.compress.Params(method="direct",
                                              max_num_singular_values=32,
                                              max_trunc_err=1.e-14,
                                              svd_rel_tol=1.e-12)



# Next, we set the parameters relating to the tensor network algorithm used to
# simulate the dynamics.
dt = 0.01  # Time step size.
alg_params = sbc.alg.Params(dt=dt,
                            temporal_compress_params=temporal_compress_params,
                            spatial_compress_params=spatial_compress_params)



# Next, we construct a MPS representing the initial state of the qubit
# system. In this example, we assume that the initial state is a product of
# single-qubit |+x> states. Accordingly, we construct a `tensornetwork` node to
# represent each one of these single-qubit states. The underlying tensor of each
# one of these nodes has the shape (1, 2, 1), where the first and last
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



# Construct system state. If backup is available from a previous unfinished
# simulation, then use that to reconstruct last save state from said
# simulation and resume simulation from there.
recover = True
sbc_output_dir = os.path.dirname(os.path.abspath(__file__)) + "/sbc-output"
pathlib.Path(sbc_output_dir).mkdir(parents=True, exist_ok=True)
backup_pkl_filename = sbc_output_dir + '/system-state-backup.pkl'
if recover and pathlib.Path(backup_pkl_filename).is_file():
    kwargs = {"pkl_filename": backup_pkl_filename,
              "system_model": system_model,
              "bath_model": bath_model,
              "forced_gc": True}  # Enforce garbage collection.
    system_state = sbc.state.SystemState.recover_and_resume(**kwargs)
    print("Recovered a `spinbosonchain.state.SystemState` object from backup.")
else:
    system_state = sbc.state.SystemState(system_model,
                                         bath_model,
                                         alg_params,
                                         initial_state_nodes)



# Next, we construct a 'wish-list' of the quantities that you would like to
# report to files as the system evolves. In this example we track several
# quantities. See the documentation for the class
# :class:`spinbosonchain.report.WishList` for more details on other quantities
# that can be tracked.
wish_list = sbc.report.WishList(ev_of_single_site_spin_ops=['sx'],
                                ev_of_nn_two_site_spin_ops=[('sz', 'sz')],
                                spin_config_probs=[[1, 1]],
                                ev_of_energy=True,
                                correlation_lengths=3)



# Next, we specify the output directory to which to report and set the report
# parameters.
report_params = sbc.report.ReportParams(wish_list=wish_list,
                                        output_dir=sbc_output_dir)



# Finally, we run the simulation and periodically report to files.
t = system_state.t  # Initial time.
backup_pkl_filename = sbc_output_dir + '/system-state-backup.pkl'
num_steps_per_report = 10
print("Running simulation using ``spinbosonchain``...")
if t < abs(t_a-0.5*dt):
    sbc.report.report(system_state, report_params)
    print("t = {}".format(t))
while t < abs(t_a-0.5*dt):
    # Here we specify additional parameters in our call to the method
    # :meth:`spinbosonchain.state.SystemState.evolve` so that we enforce garbage
    # collection for efficient memory consumption and backup the simulation data
    # to file periodically. In case of a crash, we can recover the simulation
    # data and resume where from the point of the last backup. See the
    # documentation for the method
    # :meth:`spinbosonchain.state.SystemState.evolve` for a more detailed
    # discussion on backups.
    system_state.evolve(num_steps=num_steps_per_report,
                        forced_gc=True,  # Enforce garbage collection.
                        num_k_steps_per_dump=3,
                        pkl_filename=backup_pkl_filename)
    t = system_state.t
    sbc.report.report(system_state, report_params)
    print("t = {}".format(t))
print()
print("Simulation has finished successfully: "
      "output has been written to the directory ``" + sbc_output_dir + "``.")



# Rerun simulation using TEBD if ``tenpy`` has been installed.
if util.find_spec("tenpy") is None:
    print()
    print("``tenpy`` has not been installed, therefore the simulation will not "
          "be reran using TEBD (via ``tenpy``). See file "
          "``<git-repo-root>/docs/INSTALL.rst`` for instructions on how to "
          "install ``tenpy``.")
    sys.exit()



# The following is the ``tenpy`` implementation of the simulation.

# This function constructs an MPO representation of the system Hamiltonian at
# time t.
def gen_model_snapshot(t, h, J, t_a):
    S = 0.5  # Spin angular momentum.

    h_x_tenpy = -x_field_func_form(t, t_a) / S
    h_y_tenpy = 0
    h_z_tenpy = -z_field_func_form(t, t_a, h) / S
    J_xx_tenpy = 0
    J_yy_tenpy = 0
    J_zz_tenpy = zz_coupler_func_form(t, t_a, J) / S / S

    model_params = {"hx": h_x_tenpy,
                    "hy": h_y_tenpy,
                    "hz": h_z_tenpy,
                    "Jx": J_xx_tenpy,
                    "Jy": J_yy_tenpy,
                    "Jz": J_zz_tenpy,
                    "bc_MPS": "infinite",
                    "S": S,
                    "conserve": None}
    model_snapshot = tenpy.models.spins.SpinChain(model_params)

    return model_snapshot



# Calculate initial state, which in this case is the instantaneous ground state
# at t=0.
chi = 32  # Maximum bond dimension of system state MPS.
model_snapshot = gen_model_snapshot(t=0, h=h, J=J, t_a=t_a)
product_state = ["up"] * model_snapshot.lat.N_sites
psi = tenpy.networks.mps.MPS.from_product_state(model_snapshot.lat.mps_sites(),
                                                product_state,
                                                bc=model_snapshot.lat.bc_MPS)
trunc_params = {'chi_max': chi,
                'chi_min': None,
                'degeneracy_tol': None,
                'svd_min': 1.0e-14}
dmrg_params = {'mixer': True,
               'max_E_err': 1.e-10,
               'trunc_params': trunc_params,
               'combine': False,
               'active_sites': 1}
# psi becomes initial state.
tenpy.algorithms.dmrg.run(psi, model_snapshot, dmrg_params)



# TEBD parameters.
trunc_params['trunc_cut'] = 1.0e-14
tebd_params = {'dt': dt,
               'order': 2,
               'N_steps': 1,
               'trunc_params': trunc_params}



# Initialize output data files.
tenpy_output_dir = os.path.dirname(os.path.abspath(__file__)) + "/tenpy-output"
pathlib.Path(tenpy_output_dir).mkdir(parents=True, exist_ok=True)

header = np.array([['t', 'EV']])
with open(tenpy_output_dir + "/energy.csv", 'w', 1) as file_obj:
    np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")

header = np.array([['t'] + ['EV at site #'+str(r) for r in range(L)]])
with open(tenpy_output_dir + "/sx.csv", 'w', 1) as file_obj:
    np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")

header = np.array([['t'] + ['EV at bond #'+str(i) for i in range(L)]])
with open(tenpy_output_dir + "/sz|sz.csv", 'w', 1) as file_obj:
    np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")



# Run simulation and report results.
print()
print("Running simulation using ``tenpy``...")
t = 0
while t < abs(t_a+0.5*dt):
    model_snapshot = gen_model_snapshot(t=t, h=h, J=J, t_a=t_a)

    # tenpy uses a 2-site unit cell: need to divide energy by 2.
    ev_of_E = np.real(np.sum(model_snapshot.bond_energies(psi))) / 2
    line = np.array([[t, ev_of_E]])
    with open(tenpy_output_dir + "/energy.csv", 'a', 1) as file_obj:
        np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    ev_of_sx = 2 * np.real(psi.expectation_value("Sx"))[0]
    line = np.array([[t, ev_of_sx]])
    with open(tenpy_output_dir + "/sx.csv", 'a', 1) as file_obj:
        np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    nn_sz_sz_corr = \
        4 * np.real(psi.expectation_value_term([("Sz", 0), ("Sz", 1)]))
    line = np.array([[t, nn_sz_sz_corr]])
    with open(tenpy_output_dir + "/sz|sz.csv", 'a', 1) as file_obj:
        np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    print("t = {}".format(t))

    for _ in range(num_steps_per_report):
        model_snapshot = gen_model_snapshot(t=t, h=h, J=J, t_a=t_a)
        eng = tenpy.algorithms.tebd.TEBDEngine(psi=psi,
                                               model=model_snapshot,
                                               options=tebd_params)
        eng.run()
        t += dt



# Generate plots comparing the results obtained by ``spinbosonchain`` and
# ``tenpy``.
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

sx_y_label = (r"$\left\langle\hat{\sigma}_{x;r=0}"
            r"\left(t\right)\right\rangle$")
sz_sz_y_label = (r"$\left\langle\hat{\sigma}_{z;r=0}\left(t\right)"
                 r"\hat{\sigma}_{z;r=1}\left(t\right)\right\rangle$")
energy_y_label = (r"$\left\langle E\left(t\right)\right\rangle$")

base_plot_filenames_vs_plot_specs_map = \
    {"/sx.pdf": ("/sx.csv", 1, sx_y_label),
     "/sz|sz.pdf": ("/sz|sz.csv", 1, sz_sz_y_label),
     "/energy.pdf": ("/energy.csv", 1, energy_y_label)}

for key, val in base_plot_filenames_vs_plot_specs_map.items():
    base_plot_filename = key
    base_data_filename = val[0]
    col_idx = val[1]
    y_label = val[2]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sbc_data_filename = sbc_output_dir + base_data_filename
    tenpy_data_filename = tenpy_output_dir + base_data_filename
    plot_filename = comparison_plot_output_dir + base_plot_filename
    x_label = r"$t$"
    linewidth = 3
    markersize = 15
    legend_ft_size = 13
    xy_label_ft_size = 13
    tick_label_ft_size = 13
    major_tick_len=8
    minor_tick_len=5
    legend_labels = ["spinbosonchain", "tenpy/TEBD"]

    with open(sbc_data_filename, 'rb') as file_obj:
        data = np.loadtxt(file_obj, skiprows=1, delimiter=";", dtype=float)
    x = data[:, 0]
    y = data[:, col_idx]
    ax.scatter(x, y, s=markersize, marker='s', label=legend_labels[0])

    with open(tenpy_data_filename, 'rb') as file_obj:
        data = np.loadtxt(file_obj, skiprows=1, delimiter=";", dtype=float)
    x = data[:, 0]
    y = data[:, col_idx]
    ax.scatter(x, y, s=markersize, marker='d', label=legend_labels[1])

    ax.legend(loc='best', frameon=True, fontsize=legend_ft_size)
    ax.set_xlabel(x_label, fontsize=xy_label_ft_size)
    ax.set_ylabel(y_label, fontsize=xy_label_ft_size)

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
