#!/usr/bin/env python

#####################################
## Load libraries/packages/modules ##
#####################################

# For profiling code.
import cProfile, pstats



# For creating arrays to be used to construct tensor nodes and networks. Also
# for evaluating special math functions.
import numpy as np

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# Import class representing time-dependent scalar model parameters.
from sbc.scalar import Scalar

# For specifying system model parameters.
from sbc import system

# For specifying bath model components.
from sbc import bath

# For calculating the total two-point influence function.
import sbc._influence.twopt

# For specifying how to truncate Schmidt spectra in MPS compression.
from sbc import trunc

# Import module containing class to test.
from sbc import _influence



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

# The following two functions are used to check whether the
# ``_influence.path.Path`` class is calculating the influence path correctly.
def inefficient_way_of_calculating_path_z_noise(total_two_point_influence,
                                                j_path):
    n = len(j_path) - 2
    K_tau = total_two_point_influence.z_bath.K_tau
    mu_m_tau = lambda m: max(0, m-K_tau+1)

    result = 1.0

    for m2 in range(0, n+2):
        mu_m2_tau = mu_m_tau(m=m2)
        for m1 in range(mu_m2_tau, m2+1):
            total_two_point_influence.set_m1_m2_n(m1, m2, n)
            j_r_m1 = j_path[m1]
            j_r_m2 = j_path[m2]
            result *= total_two_point_influence.eval(j_r_m1, j_r_m2)

    return result



def inefficient_way_of_calculating_path_yz_noise(total_two_point_influence,
                                                 j_path):
    n = (len(j_path)-4) // 3
    K_tau = total_two_point_influence.z_bath.K_tau
    mu_m_tau = lambda m: max(0, m-3*K_tau+1)

    result = 1.0

    for m2 in range(0, 3*n+4):
        mu_m2_tau = mu_m_tau(m=m2)
        for m1 in range(mu_m2_tau, m2+1):
            total_two_point_influence.set_m1_m2_n(m1, m2, n)
            j_r_m1 = j_path[m1]
            j_r_m2 = j_path[m2]
            result *= total_two_point_influence.eval(j_r_m1, j_r_m2)

    return result



# To evaluate the influence path, which is represented by a MPS.
def eval_influence_path(influence_path, j_path):
    result = None
    mps_nodes = influence_path.Xi_I_1_1_nodes + influence_path.Xi_I_dashv_nodes
    
    for mps_node, j in zip(mps_nodes, j_path):
        j_tensor = np.zeros([4])
        j_tensor[j] = 1
        j_node = tn.Node(j_tensor)

        nodes_to_contract = [mps_node, j_node]
        network_struct = [(-1, 1, -2), (1,)]
        sliced_mps_node = tn.ncon(nodes_to_contract, network_struct)

        if result == None:
            b_node = tn.Node(np.ones([4]))
            nodes_to_contract = [b_node, sliced_mps_node]
            network_struct = [(1,), (1, -1)]
            result = tn.ncon(nodes_to_contract, network_struct)
        else:
            nodes_to_contract = [result, sliced_mps_node]
            network_struct = [(1,), (1, -2)]
            result = tn.ncon(nodes_to_contract, network_struct)

    result = result.tensor[0]

    return result



# _influence.path.Path test #1.
print("_influence.path.Path test #1")
print("============================")

# Need to construct a ``system.Model`` object, which specifies all the model,
# parameters. Since we are only dealing with influence, all we need are the
# x-fields.
print("Constructing an instance of ``system.Model``.\n")

def quad_fn(t, a, b):
    return a * t * t + b

x_fields = [Scalar(quad_fn, {"a": 0.05, "b": -0.5})]
system_model = system.Model(x_fields=x_fields)

# Need to construct a ``bath.Model`` object. In order to do this, we need to a
# few more objects. Starting the the coupling energy scales. We'll assume zero
# y-noise here.
print("Constructing an instance of ``bath.Model`` with z-noise only.\n")

def linear_fn(t, a, b):
    return a*t+b

z_coupling_energy_scales = [Scalar(linear_fn, {"a": 0.005, "b": 1.0})]

# Next we need to construct the spectral-densities of noise.
def A_v_r_0T_s_func_form(omega, lambda_v_s, omega_v_s_c):
    return lambda_v_s * omega * np.exp(-omega / omega_v_s_c)

lambda_z_1 = 3 * 0.1025 / 8
omega_z_1_c = 3 * 10.0 / 8

A_z_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_v_r_0T_s_func_form,
                                func_kwargs={"lambda_v_s": lambda_z_1,
                                             "omega_v_s_c": omega_z_1_c},
                                hard_cutoff_freq=40*omega_z_1_c,
                                zero_pt_derivative=lambda_z_1)

lambda_z_2 = 0.1025 / 8
omega_z_2_c = 10.0 / 8

A_z_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_v_r_0T_s_func_form,
                                func_kwargs={"lambda_v_s": lambda_z_2,
                                             "omega_v_s_c": omega_z_2_c},
                                hard_cutoff_freq=40*omega_z_2_c,
                                zero_pt_derivative=lambda_z_2)

A_z_0_0T = bath.SpectralDensity0T(cmpnts=[A_z_r_0T_1, A_z_r_0T_2])

z_spectral_densities_0T = [A_z_0_0T]

bath_model = bath.Model(L=1,
                        beta=1.0,
                        memory=0.5,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        z_spectral_densities_0T=z_spectral_densities_0T)

# Next, we need to specify the SVD truncation parameters.
print("Constructing an instance of ``trunc.Params``.\n")

trunc_params = trunc.Params(max_num_singular_values=16,
                            max_trunc_err=1.0e-14)

# Finally, we construct the ``_influence.path.Path`` object.
print("Constructing instance of ``_influence.path.Path``:\n")

r = 0
dt = 0.1
influence_path = _influence.path.Path(r, system_model,
                                      bath_model, dt, trunc_params)

n = 0
num_n_steps = 1
for _ in range(9):
    print("Evolving from time step n={} to n={}:".format(n, n+num_n_steps))
    influence_path.evolve(num_n_steps)
    print("    # of Xi_I_1_1 nodes =", len(influence_path.Xi_I_1_1_nodes))
    print("    # of Xi_I_1_2 nodes =", len(influence_path.Xi_I_1_2_nodes))
    print("    # of Xi_I_dashv_1 nodes =", len(influence_path.Xi_I_dashv_1_nodes))
    print("    # of Xi_I_dashv_2 nodes =", len(influence_path.Xi_I_dashv_2_nodes))
    n += num_n_steps
    print()

print("Evaluating influence path for a random j-path and comparing result to "
      "alternate method for evaluating the path:")

total_two_point_influence = \
    sbc._influence.twopt.Total(r, system_model, bath_model, dt)

# Many j-paths will likely yield near-zero results. If we chose paths that
# contain only j=0 and 3, then the influence path should evaluate to a number
# on the order of unity for our example.
j_path = 3*np.random.randint(low=0, high=1, size=n+2)

result_1 = eval_influence_path(influence_path, j_path)
result_2 = \
    inefficient_way_of_calculating_path_z_noise(total_two_point_influence,
                                                j_path)

unformatted_msg = \
    "    Using TNs: {}, using alternative method: {}, rel-diff: {}"
rel_diff = abs((result_1-result_2) / result_2)
print(unformatted_msg.format(result_1, result_2, rel_diff))
print("\n\n")



# _influence.path.Path test #2.
print("_influence.path.Path test #2")
print("============================")

# Need to construct a ``bath.Model`` object that encodes both y- and z-noise.
print("Constructing an instance of ``bath.Model`` with y- and z-noise.\n")

y_coupling_energy_scales = [Scalar(linear_fn, {"a": 0.01, "b": 0.8})]

lambda_y_1 = 3 * 0.1025 / 4
omega_y_1_c = 3 * 10.0 / 4

A_y_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_v_r_0T_s_func_form,
                                func_kwargs={"lambda_v_s": lambda_y_1,
                                             "omega_v_s_c": omega_y_1_c},
                                hard_cutoff_freq=40*omega_y_1_c,
                                zero_pt_derivative=lambda_y_1)

A_y_0_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1])

y_spectral_densities_0T = [A_y_0_0T]

bath_model = bath.Model(L=1,
                        beta=1.0,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T,
                        z_spectral_densities_0T=z_spectral_densities_0T)

# Finally, we construct the ``_influence.path.Path`` object.
print("Constructing instance of ``_influence.path.Path``:\n")

r = 0
dt = 0.1
influence_path = _influence.path.Path(r, system_model,
                                      bath_model, dt, trunc_params)

profiler = cProfile.Profile()
profiler.enable()

n = 0
num_n_steps = 1
for _ in range(9):
    print("Evolving from time step n={} to n={}:".format(n, n+num_n_steps))
    influence_path.evolve(num_n_steps)
    print("    # of Xi_I_1_1 nodes =", len(influence_path.Xi_I_1_1_nodes))
    print("    # of Xi_I_1_2 nodes =", len(influence_path.Xi_I_1_2_nodes))
    print("    # of Xi_I_dashv_1 nodes =", len(influence_path.Xi_I_dashv_1_nodes))
    print("    # of Xi_I_dashv_2 nodes =", len(influence_path.Xi_I_dashv_2_nodes))
    n += num_n_steps
    print()

profiler.disable()

print("Evaluating influence path for a random j-path and comparing result to "
      "alternate method for evaluating the path:")

total_two_point_influence = \
    sbc._influence.twopt.Total(r, system_model, bath_model, dt)

# Many j-paths will likely yield near-zero results. If we chose paths that
# contain only j=0 and 3, then the influence path should be appreciable.
j_path = 3*np.random.randint(low=0, high=1, size=3*n+4)

result_1 = eval_influence_path(influence_path, j_path)
result_2 = \
    inefficient_way_of_calculating_path_z_noise(total_two_point_influence,
                                                j_path)

unformatted_msg = \
    "    Using TNs: {}, using alternative method: {}, rel-diff: {}"
rel_diff = abs((result_1-result_2) / result_2)
print(unformatted_msg.format(result_1, result_2, rel_diff))

stats = pstats.Stats(profiler).sort_stats("cumulative")
stats.print_stats()
