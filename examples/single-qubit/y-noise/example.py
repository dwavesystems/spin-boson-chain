#!/usr/bin/env python
"""An example script using ``spinbosonchain``.

In this example, we implement a simulation of the dynamics of a single qubit
subject to time-dependent transverse and longitudinal fields, and environmental
noise that is coupled to the :math:`y`--component of the qubit's spin. The
spectral density of noise comprises of two Dirac delta-like peak. The
system-environment coupling energy scale is also time-dependent. The simulation
tracks :math:`\left\langle\hat{\sigma}_{x;r}\left(t\right)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{y;r}\left(t\right)\right\rangle`, and
:math:`\left\langle\hat{\sigma}_{z;r}\left(t\right)\right\rangle`.

This example tests the algorithm implemented in the ``spinbosonchain`` library
that is used to calculate local influence paths for the case where the system is
subject to multi-component noise (in this case with a double-peak spectrum), and
the system's bath correlation time, or "memory", is larger than the total
simulation time.

If ``quspin`` has been installed, then the same quantities will also be
calculated by exact diagonalization for comparison sake. If ``matplotlib`` has
been installed as well, then comparison plots will be generated.

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
10. Rerun the simulation using exact diagonalization if ``quspin`` has been
    installed.
11. Generate comparison plots if ``matplotlib`` has been installed.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# To determine the current working directory and to set # threads to use.
import os

# To check whether the quspin and matplotlib libraries have been installed.
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

# Import the quspin library if it exists.
if util.find_spec("quspin") is not None:
    import quspin

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

L = 1  # Number of qubits.
t_a = 2  # Total simulation time.
h = (1/10,)
x_fields = [sbc.scalar.Scalar(x_field_func_form, {"t_a": t_a})] * L
z_fields = [sbc.scalar.Scalar(z_field_func_form, {"t_a": t_a, "h_r": h_r})
            for h_r in h]
zz_couplers = []
system_model = sbc.system.Model(x_fields=x_fields,
                                z_fields=z_fields,
                                zz_couplers=zz_couplers)

# The next step towards constructing our spin-boson model is to specify the bath
# model components. See the documentation for the module
# :mod:`spinbosonchain.bath` for a discussion on the bath model components. The
# bath model components include the (inverse) temperature, the system's
# so-called 'memory', the system-environment coupling energy scales, and the
# zero-temperature spectral densities which characterize the kind of noise the
# system is subject to. The following code specifies the bath model components.
def A_v_r_0T_cmpnt_func_form(omega, omega_0, w, eta):
    return 2 * np.pi * eta * delta(omega, omega_0, w)

def delta(omega, omega_0, w):
    return (np.exp(-((omega-omega_0)/w) * ((omega-omega_0)/w))
            / np.sqrt(w * w * np.pi))

def y_coupling_energy_scale_func_form(t, t_a, E_y_r):
    return E_y_r * sqrt_A(t, t_a)

def sqrt_A(t, t_a):
    return np.sqrt(A(t, t_a))

etas = {("y", 0, 0): 4/5, ("y", 0, 1): 1/5}
mode_frequencies = {("y", 0): 5, ("y", 1): 1}
w = 1e-6
E_y = (1/2,)

spectral_densities_0T = {"y": [0]*L}
coupling_energy_scales = {"y": [0]*L}
for noise_type in ("y",):
    for r in range(L):
        cmpnts = []
        
        for varsigma in (0, 1):
            eta = etas[(noise_type, r, varsigma)]
            if eta == 0.0:
                continue
            omega_0 = mode_frequencies[(noise_type, varsigma)]
            ir_cutoff = omega_0 - 8*w
            uv_cutoff = omega_0 + 8*w
            ctor_kwargs = {"func_form": A_v_r_0T_cmpnt_func_form,
                           "func_kwargs": {"omega_0": omega_0,
                                           "w": w,
                                           "eta": eta},
                           "ir_cutoff": ir_cutoff,
                           "uv_cutoff": uv_cutoff}
            cmpnt = sbc.bath.SpectralDensityCmpnt0T(**ctor_kwargs)
            cmpnts.append(cmpnt)
            
        A_v_r_0T = sbc.bath.SpectralDensity0T(cmpnts=cmpnts)
        spectral_densities_0T[noise_type][r] = A_v_r_0T

        func_form = y_coupling_energy_scale_func_form
        func_kwargs = {"t_a": t_a, "E_y_r": E_y[r]}
            
        coupling_energy_scale = sbc.scalar.Scalar(func_form, func_kwargs)
        coupling_energy_scales[noise_type][r] = coupling_energy_scale

# ``memory`` refers to the system memory. When low frequency noise is present,
# the current dynamics of the system depend on its history. As a general rule:
# the lower the frequency at which the noise is peaked, the more ``memory`` is
# required. In this example, we require memory of the system's entire history.
memory = t_a
beta = 1.0  # Inverse temperature beta=1/(kB*T).
ctor_kwargs = {"L": L,
               "beta": beta,
               "memory": memory,
               "y_coupling_energy_scales": coupling_energy_scales["y"],
               "y_spectral_densities_0T": spectral_densities_0T["y"]}
bath_model = sbc.bath.Model(**ctor_kwargs)



# Next, we set the compression parameters for the matrix product states (MPS's)
# that span time as well as those that span space. 'Temporal' MPS's are used to
# represent influence functionals/paths, where 'spatial' MPS's are used to
# represent the system's state. See the documentation for the class
# spinbosonchain.compress.Params for a description of each available compression
# parameter. Note that since this is a single qubit system, no compression is
# performed on the spatial MPS's as they are simply scalars: there are no
# spatial bonds in the MPS sense.
temporal_compress_params = sbc.compress.Params(method="zip-up",
                                               max_num_singular_values=64,
                                               max_trunc_err=1.e-14,
                                               svd_rel_tol=1.e-12,
                                               max_num_var_sweeps=2,
                                               var_rel_tol=1e-8)
spatial_compress_params = sbc.compress.Params()



# Next, we set the parameters relating to the tensor network algorithm used to
# simulate the dynamics.
dt = 0.1  # Time step size.
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
print("Running simulation using ``spinbosonchain``...")
sbc.report.report(system_state, report_params)
print("t = {}".format(t))
while t < abs(t_a-0.5*dt):
    system_state.evolve(num_steps=1)
    t = system_state.t
    sbc.report.report(system_state, report_params)
    print("t = {}".format(t))
print()
print("Simulation has finished successfully: "
      "output has been written to the directory ``" + sbc_output_dir + "``.")



# Rerun simulation using exact diagonalization if ``quspin`` has been
# installed.
if util.find_spec("quspin") is None:
    print()
    print("``quspin`` has not been installed, therefore the simulation will "
          "not be reran using exact diagonalization (via ``quspin``). See "
          "file ``<git-repo-root>/docs/INSTALL.rst`` for instructions on "
          "how to install ``quspin``.")
    sys.exit()



# The following is the ``quspin`` implementation of the simulation.

# First, we need to construct the qubit-QHO basis, where QHO is an abbreviation
# of 'quantum harmonic oscillator'. Note that ``sps`` in the ``boson_basis_1d``
# specifies the Hilbert space dimension of the QHO.
qho_dim = 6
qubit_basis = quspin.basis.spin_basis_1d(L=L, S="1/2")
qho_basis = quspin.basis.boson_basis_1d(L=2*L, sps=qho_dim)
qubit_qho_basis = quspin.basis.tensor_basis(qubit_basis, qho_basis)

# Construct the system and total Hamiltonians.
sx_coefficients = [[-1/2, r] for r in range(L)]
sz_coefficients = [[h[r]/2, r] for r in range(L)]

number_op_coefficients = []
sy_creation_op_coefficients = []
sy_annihilation_op_coefficients = []

for r in range(L):
    for noise_type in ("y",):
        for varsigma in (0, 1):
            eta = etas[(noise_type, r, varsigma)]
            if eta == 0.0:
                continue
            
            omega_0 = mode_frequencies[(noise_type, varsigma)]

            number_op_coefficients.append([omega_0, varsigma])
            elem = [E_y[r]*np.sqrt(eta), r, varsigma]
            sy_creation_op_coefficients.append(elem)
            sy_annihilation_op_coefficients.append(elem)

func_args = (t_a,)
                
sx_cmpnts = ["x|", sx_coefficients, A, func_args]
sz_cmpnts = ["z|", sz_coefficients, B, func_args]

number_op_cmpnts = ["|n", number_op_coefficients]
sy_creation_op_cmpnts = ["y|+",
                         sy_creation_op_coefficients,
                         sqrt_A,
                         func_args]
sy_annihilation_op_cmpnts = ["y|-",
                             sy_annihilation_op_coefficients,
                             sqrt_A,
                             func_args]

static_list = [number_op_cmpnts]
dynamic_list = [sx_cmpnts, sz_cmpnts,
                sy_creation_op_cmpnts, sy_annihilation_op_cmpnts]
no_checks = {"check_symm": False, "check_pcon": False, "check_herm": False}
H = quspin.operators.hamiltonian(static_list=static_list,
                                 dynamic_list=dynamic_list,
                                 basis=qubit_qho_basis,
                                 **no_checks)



# Next, we construct the system's reduced density matrix at time t=0.
id_plus_sx_all_div_by_2 = np.matrix([[0.5, 0.5], [0.5, 0.5]])
rho_A_i = 1
for r in range(L):
    rho_A_i = np.kron(1, id_plus_sx_all_div_by_2)



# Next, we construct the bath's reduced density matrix at time t=0.
number_op_cmpnts = ["n", number_op_coefficients]
H_B = quspin.operators.hamiltonian(static_list=[number_op_cmpnts],
                                   dynamic_list=[],
                                   basis=qho_basis,
                                   **no_checks)

exp_neg_beta_H_B = \
    quspin.operators.exp_op(O=H_B, a=-beta).get_mat(dense=True).real
Z_B = np.trace(exp_neg_beta_H_B)
rho_B_i = exp_neg_beta_H_B / Z_B



# Next, we construct the density matrix at time t=0.
rho_i = np.kron(rho_A_i, rho_B_i)



# Next, we construct the observables for which to calculate their expectation
# values.
obs = dict()
op_strings = ["x|", "y|", "z|"]
for op_string in op_strings:
    op_coefficients = [[1] + list(range(L))]
    op_cmpnts = [op_string, op_coefficients]
    op = quspin.operators.hamiltonian(static_list=[op_cmpnts],
                                      dynamic_list=[],
                                      basis=qubit_qho_basis,
                                      **no_checks)
    obs[op_string] = op



# Calculate the expectation values of the observables of interest.
print()
print("Running simulation using ``quspin``...")

N_t = int(round(t_a / dt)) + 1
times = [n*dt for n in range(N_t)]

obs_vs_time_result = {op_string: [] for op_string in op_strings}
ED_state_vs_time = quspin.tools.evolution.ED_state_vs_time
for t in times:
    if t == 0:
        rho = rho_i
    else:
        E, V = H.eigh(time=t-dt)
        rho = ED_state_vs_time(rho, E, V, [0.5*dt])[:, :, 0]
        E, V = H.eigh(time=t)
        rho = ED_state_vs_time(rho, E, V, [0.5*dt])[:, :, 0]

    for op_string in op_strings:
        op = obs[op_string]
        obs_vs_time_result[op_string].append(op.expt_value(rho, time=t))
    print("t = {}".format(t))


# Report results.
quspin_output_dir = \
    os.path.dirname(os.path.abspath(__file__)) + "/quspin-output"
pathlib.Path(quspin_output_dir).mkdir(parents=True, exist_ok=True)

site_header = np.array([['t'] + ['EV at site #'+str(r) for r in range(L)]])

filenames_vs_headers_map = {quspin_output_dir + "/sx.csv": site_header,
                            quspin_output_dir + "/sy.csv": site_header,
                            quspin_output_dir + "/sz.csv": site_header}

for filename, header in filenames_vs_headers_map.items():
    with open(filename, 'w', 1) as file_obj:
        np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")

filenames_vs_op_strings_map = {quspin_output_dir + "/sx.csv": ["x|"],
                               quspin_output_dir + "/sy.csv": ["y|"],
                               quspin_output_dir + "/sz.csv": ["z|"]}

for filename, op_strings in filenames_vs_op_strings_map.items():
    to_zip = ([times]
              + [obs_vs_time_result[op_string] for op_string in op_strings])
    zip_obj = zip(*to_zip)
    for item in zip_obj:
        fmt = 'f8' + ', f8'*(len(item)-1)
        line = np.array([tuple(np.real(elem) for elem in item)], dtype=fmt)
        with open(filename, 'a', 1) as file_obj:
            np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

print()
print("Simulation has finished successfully: "
      "output has been written to the directory ``" + quspin_output_dir + "``.")



# Generate plots comparing the results obtained by ``spinbosonchain`` and
# ``quspin``.
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
sy_y_label = (r"$\left\langle\hat{\sigma}_{y;r=0}"
              r"\left(t\right)\right\rangle$")
sz_y_label = (r"$\left\langle\hat{\sigma}_{z;r=0}"
              r"\left(t\right)\right\rangle$")

base_plot_filenames_vs_plot_specs_map = {"/sx.pdf": ("/sx.csv", 1, sx_y_label),
                                         "/sy.pdf": ("/sy.csv", 1, sy_y_label),
                                         "/sz.pdf": ("/sz.csv", 1, sz_y_label)}

for key, val in base_plot_filenames_vs_plot_specs_map.items():
    base_plot_filename = key
    base_data_filename = val[0]
    col_idx = val[1]
    y_label = val[2]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sbc_data_filename = sbc_output_dir + base_data_filename
    quspin_data_filename = quspin_output_dir + base_data_filename
    plot_filename = comparison_plot_output_dir + base_plot_filename
    x_label = r"$t$"
    linewidth = 3
    markersize = 15
    legend_ft_size = 13
    xy_label_ft_size = 13
    tick_label_ft_size = 13
    major_tick_len=8
    minor_tick_len=5
    legend_labels = ["spinbosonchain", "quspin/ED"]

    with open(sbc_data_filename, 'rb') as file_obj:
        data = np.loadtxt(file_obj, skiprows=1, delimiter=";", dtype=float)
    x = data[:, 0]
    y = data[:, col_idx]
    ax.scatter(x, y, s=markersize, marker='s', label=legend_labels[0])

    with open(quspin_data_filename, 'rb') as file_obj:
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
