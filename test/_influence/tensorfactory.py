#!/usr/bin/env python

#####################################
## Load libraries/packages/modules ##
#####################################

# For evaluating special math functions.
import numpy as np



# Import class representing time-dependent scalar model parameters.
from sbc.scalar import Scalar

# For specifying system model parameters.
from sbc import system

# For specifying bath model components.
from sbc import bath

# For constructing 'total' two-point influence objects.
from sbc import _influence

# Module to test.
from sbc._influence import tensorfactory



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

# _influence.tensorfactory.InfluenceNodeRank3 test #1.
print("_influence.tensorfactory.InfluenceNodeRank3 test #1")
print("===================================================")

# Need to construct a ``system.Model`` object, which specifies all the model,
# parameters. Since we are only dealing with influence, all we need are the
# x-fields.
print("Constructing an instance of ``system.Model``.\n")

def quad_fn(t, a, b):
    return a * t * t + b

x_fields = [Scalar(quad_fn, {"a": 0.05, "b": -0.5}),
            Scalar(quad_fn, {"a": -0.05, "b": 0.5}),
            0.5]
system_model = system.Model(x_fields=x_fields)

# Need to construct a ``bath.Model`` object. In order to do this, we need to a
# few more objects. Starting the the coupling energy scales. We'll assume zero
# y-noise here.
print("Constructing an instance of ``bath.Model`` with z-noise only.\n")

def linear_fn(t, a, b):
    return a*t+b

z_coupling_energy_scales = [0.0,
                            Scalar(linear_fn, {"a": 0.2, "b": 0.1}),
                            0.0]

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

A_z_0_0T = 0
A_z_1_0T = bath.SpectralDensity0T(cmpnts=[A_z_r_0T_1, A_z_r_0T_2])
A_z_2_0T = 0

z_spectral_densities_0T = [A_z_0_0T, A_z_1_0T, A_z_2_0T]

bath_model = bath.Model(L=3,
                        beta=1.0,
                        memory=0.5,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        z_spectral_densities_0T=z_spectral_densities_0T)

# Next we construct a list of ``_influence.twopt.Total`` objects, one per site.
dt = 0.1
L = system_model.L
total_two_pt_influence_objs = \
    [_influence.twopt.Total(r, system_model, bath_model, dt) for r in range(L)]

# Finally we construct the ``_influence.tensorfactory.InfluenceNodeRank3``
# object. We test it by building several nodes from it, and then listing all the
# elements of each underlying tensor.
print("Constructing instances of "
      "``_influence.tensorfactory.InfluenceNodeRank3`` to build several "
      "nodes:\n")

influence_node_rank_3_factories = \
    [tensorfactory.InfluenceNodeRank3(total_two_pt_influence_obj)
     for total_two_pt_influence_obj in total_two_pt_influence_objs]

unformatted_msg_1 = "    Building node for (r, m, n)=({}, {}, {}):"
unformatted_msg_2 = "        node.tensor[{}, {}, 0] = {}"
n = 9
for r in range(L):
    for m in (5, n+1):
        msg = unformatted_msg_1.format(r, m, n)
        print(msg)
        node = influence_node_rank_3_factories[r].build(m, n)
        for b_r_m in range(4):
            for j_r_m in range(4):
                elem = node.tensor[b_r_m, j_r_m, 0]
                msg = unformatted_msg_2.format(b_r_m, j_r_m, elem)
                print(msg)
        print()
print("\n\n")



# _influence.tensorfactory.InfluenceNodeRank3 test #2.
print("_influence.tensorfactory.InfluenceNodeRank3 test #2")
print("===================================================")

# Need to construct a ``bath.Model`` object that encodes both y- and z-noise.
print("Constructing an instance of ``bath.Model`` with y- and z-noise.\n")

y_coupling_energy_scales = [Scalar(linear_fn, {"a": 0.4, "b": -0.2}),
                            Scalar(linear_fn, {"a": -0.2, "b": 0.1}),
                            0.25]

lambda_y_1 = 3 * 0.1025 / 4
omega_y_1_c = 3 * 10.0 / 4

A_y_r_0T_1 = \
    bath.SpectralDensityCmpnt0T(func_form=A_v_r_0T_s_func_form,
                                func_kwargs={"lambda_v_s": lambda_y_1,
                                             "omega_v_s_c": omega_y_1_c},
                                hard_cutoff_freq=40*omega_y_1_c,
                                zero_pt_derivative=lambda_y_1)

lambda_y_2 = 0.1025 / 4
omega_y_2_c = 10.0 / 4

A_y_r_0T_2 = \
    bath.SpectralDensityCmpnt0T(func_form=A_v_r_0T_s_func_form,
                                func_kwargs={"lambda_v_s": lambda_y_2,
                                             "omega_v_s_c": omega_y_2_c},
                                hard_cutoff_freq=40*omega_y_2_c,
                                zero_pt_derivative=lambda_y_2)


A_y_0_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1])
A_y_1_0T = bath.SpectralDensity0T(cmpnts=[A_y_r_0T_1, A_y_r_0T_2])
A_y_2_0T = 0

y_spectral_densities_0T = [A_y_0_0T, A_y_1_0T, A_y_2_0T]

bath_model = bath.Model(L=3,
                        beta=1.0,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T,
                        z_spectral_densities_0T=z_spectral_densities_0T)

# Next we construct a list of ``_influence.twopt.Total`` objects, one per site.
total_two_pt_influence_objs = \
    [_influence.twopt.Total(r, system_model, bath_model, dt) for r in range(L)]


# Finally we construct the ``_influence.tensorfactory.InfluenceNodeRank3``
# object. We test it by building several nodes from it, and then listing all the
# elements of each underlying tensor.
print("Constructing instances of "
      "``_influence.tensorfactory.InfluenceNodeRank3`` to build several "
      "nodes:\n")

influence_node_rank_3_factories = \
    [tensorfactory.InfluenceNodeRank3(total_two_pt_influence_obj)
     for total_two_pt_influence_obj in total_two_pt_influence_objs]

unformatted_msg_1 = "    Building node for (r, m, n)=({}, {}, {}):"
unformatted_msg_2 = "        node.tensor[{}, {}, 0] = {}"
n = 4
for r in range(L):
    for m in (7, 3*n+3):
        msg = unformatted_msg_1.format(r, m, n)
        print(msg)
        node = influence_node_rank_3_factories[r].build(m, n)
        for b_r_m in range(4):
            for j_r_m in range(4):
                elem = node.tensor[b_r_m, j_r_m, 0]
                msg = unformatted_msg_2.format(b_r_m, j_r_m, elem)
                print(msg)
        print()
print("\n\n")



# _influence.tensorfactory.InfluenceNodeRank4 test #1.
print("_influence.tensorfactory.InfluenceNodeRank4 test #1")
print("===================================================")

# For this test we consider only z-noise.
bath_model = bath.Model(L=3,
                        beta=1.0,
                        memory=0.5,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        z_spectral_densities_0T=z_spectral_densities_0T)

# Next we construct a list of ``_influence.twopt.Total`` objects, one per site.
total_two_pt_influence_objs = \
    [_influence.twopt.Total(r, system_model, bath_model, dt) for r in range(L)]

# Finally we construct the ``_influence.tensorfactory.InfluenceNodeRank4``
# object. We test it by building several nodes from it, and then listing all the
# elements of each underlying tensor.
print("Constructing instances of "
      "``_influence.tensorfactory.InfluenceNodeRank4`` to build several "
      "nodes:\n")

influence_node_rank_4_factories = \
    [tensorfactory.InfluenceNodeRank4(total_two_pt_influence_obj)
     for total_two_pt_influence_obj in total_two_pt_influence_objs]

unformatted_msg_1 = "    Building node for (r, m1, m2, n)=({}, {}, {}, {}):"
unformatted_msg_2 = "        node.tensor[{}, {}, {}, {}] = {}"
unformatted_msg_3 = "        node.tensor[{}, {}, {}, {}] = {}"
n = 20
K_tau = total_two_pt_influence_objs[0].z_bath.K_tau
for r in range(L):
    for m2 in (n-2, n-1):
        mu_m2_tau = max(0, m2-K_tau+1)
        for m1 in (mu_m2_tau, m2):
            msg = unformatted_msg_1.format(r, m1, m2, n)
            print(msg)
            node = influence_node_rank_4_factories[r].build(m1, m2, n)
            for b_r_m1 in range(node.tensor.shape[0]):
                for j_r_m1 in range(4):
                    for j_r_m1_prime in range(4):
                        for b_r_m1P1 in range(4):
                            elem = node.tensor[b_r_m1, j_r_m1,
                                               j_r_m1_prime, b_r_m1P1]
                            msg = unformatted_msg_2.format(b_r_m1, j_r_m1,
                                                           j_r_m1_prime,
                                                           b_r_m1P1, elem)
                            print(msg)
            print()
print("\n\n")



# _influence.tensorfactory.InfluenceNodeRank4 test #2.
print("_influence.tensorfactory.InfluenceNodeRank4 test #2")
print("===================================================")

# Need to construct a ``bath.Model`` object that encodes both y- and z-noise.
bath_model = bath.Model(L=3,
                        beta=1.0,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T,
                        z_spectral_densities_0T=z_spectral_densities_0T)

# Next we construct a list of ``_influence.twopt.Total`` objects, one per site.
total_two_pt_influence_objs = \
    [_influence.twopt.Total(r, system_model, bath_model, dt) for r in range(L)]

# Finally we construct the ``_influence.tensorfactory.InfluenceNodeRank4``
# object. We test it by building several nodes from it, and then listing all the
# elements of each underlying tensor.
print("Constructing instances of "
      "``_influence.tensorfactory.InfluenceNodeRank4`` to build several "
      "nodes:\n")

influence_node_rank_4_factories = \
    [tensorfactory.InfluenceNodeRank4(total_two_pt_influence_obj)
     for total_two_pt_influence_obj in total_two_pt_influence_objs]

unformatted_msg_1 = "    Building node for (r, m1, m2, n)=({}, {}, {}, {}):"
unformatted_msg_2 = "        node.tensor[{}, {}, {}, {}] = {}"
unformatted_msg_3 = "        node.tensor[{}, {}, {}, {}] = {}"
n = 10
K_tau = total_two_pt_influence_objs[0].z_bath.K_tau
for r in range(L):
    for m2 in (3*n, 3*n+1):
        mu_m2_tau = max(0, m2-3*K_tau+1)
        for m1 in (mu_m2_tau, m2):
            msg = unformatted_msg_1.format(r, m1, m2, n)
            print(msg)
            node = influence_node_rank_4_factories[r].build(m1, m2, n)
            for b_r_m1 in range(node.tensor.shape[0]):
                for j_r_m1 in range(4):
                    for j_r_m1_prime in range(4):
                        for b_r_m1P1 in range(4):
                            elem = node.tensor[b_r_m1, j_r_m1,
                                               j_r_m1_prime, b_r_m1P1]
                            msg = unformatted_msg_2.format(b_r_m1, j_r_m1,
                                                           j_r_m1_prime,
                                                           b_r_m1P1, elem)
                            print(msg)
            print()



# _influence.tensorfactory.InfluenceMPO test #1.
print("_influence.tensorfactory.InfluenceMPO test #1")
print("=============================================")

# For this test we consider only z-noise.
bath_model = bath.Model(L=3,
                        beta=1.0,
                        memory=0.5,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        z_spectral_densities_0T=z_spectral_densities_0T)

# Next we construct a list of ``_influence.twopt.Total`` objects, one per site.
total_two_pt_influence_objs = \
    [_influence.twopt.Total(r, system_model, bath_model, dt) for r in range(L)]

# Finally we construct the ``_influence.tensorfactory.InfluenceMPO``
# object. We test it by building a couple of MPOs from it.
print("Constructing instances of ``_influence.tensorfactory.InfluenceMPO`` to "
      "build a few MPOs:\n")

influence_mpo_factories = \
    [tensorfactory.InfluenceMPO(total_two_pt_influence_obj)
     for total_two_pt_influence_obj in total_two_pt_influence_objs]

r = 1
m2 = 3
n = 5
influence_mpo = influence_mpo_factories[r].build(m2, n)
print("    Building MPO for (r, m2, n)=({}, {}, {}):".format(r, m2, n))
print("        # of nodes in MPO: {}".format(len(influence_mpo)))

r = 1
m2 = 15
n = 20
influence_mpo = influence_mpo_factories[r].build(m2, n)
print("    Building MPO for (r, m2, n)=({}, {}, {}):".format(r, m2, n))
print("        # of nodes in MPO: {}".format(len(influence_mpo)))

r = 1
m2 = 20
n = 20
influence_mpo = influence_mpo_factories[r].build(m2, n)
print("    Building MPO for (r, m2, n)=({}, {}, {}):".format(r, m2, n))
print("        # of nodes in MPO: {}".format(len(influence_mpo)))
print("\n\n")



# _influence.tensorfactory.InfluenceMPO test #2.
print("_influence.tensorfactory.InfluenceMPO test #2")
print("=============================================")

# Need to construct a ``bath.Model`` object that encodes both y- and z-noise.
bath_model = bath.Model(L=3,
                        beta=1.0,
                        memory=0.5,
                        y_coupling_energy_scales=y_coupling_energy_scales,
                        z_coupling_energy_scales=z_coupling_energy_scales,
                        y_spectral_densities_0T=y_spectral_densities_0T,
                        z_spectral_densities_0T=z_spectral_densities_0T)

# Next we construct a list of ``_influence.twopt.Total`` objects, one per site.
total_two_pt_influence_objs = \
    [_influence.twopt.Total(r, system_model, bath_model, dt) for r in range(L)]

# Finally we construct the ``_influence.tensorfactory.InfluenceMPO``
# object. We test it by building a couple of MPOs from it.
print("Constructing instances of ``_influence.tensorfactory.InfluenceMPO`` to "
      "build a few MPOs:\n")

influence_mpo_factories = \
    [tensorfactory.InfluenceMPO(total_two_pt_influence_obj)
     for total_two_pt_influence_obj in total_two_pt_influence_objs]

r = 1
m2 = 9
n = 5
influence_mpo = influence_mpo_factories[r].build(m2, n)
print("    Building MPO for (r, m2, n)=({}, {}, {}):".format(r, m2, n))
print("        # of nodes in MPO: {}".format(len(influence_mpo)))

r = 1
m2 = 45
n = 20
influence_mpo = influence_mpo_factories[r].build(m2, n)
print("    Building MPO for (r, m2, n)=({}, {}, {}):".format(r, m2, n))
print("        # of nodes in MPO: {}".format(len(influence_mpo)))

r = 1
m2 = 62
n = 20
influence_mpo = influence_mpo_factories[r].build(m2, n)
print("    Building MPO for (r, m2, n)=({}, {}, {}):".format(r, m2, n))
print("        # of nodes in MPO: {}".format(len(influence_mpo)))
