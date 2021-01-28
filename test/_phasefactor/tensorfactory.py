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

# Module to test.
from sbc._phasefactor import tensorfactory



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

# _phasefactor.tensorfactory.ZFieldPhaseFactorNodeRank3 test #1.
print("_phasefactor.tensorfactory.ZFieldPhaseFactorNodeRank3 test #1")
print("===========================================================")

# Need to construct a ``system.Model`` object first.
print("Constructing an instance of ``system.Model``.\n")

def linear_fn(t, a, b):
    return a * t + b

z_fields = [Scalar(linear_fn, {"a": 0.05, "b": 0.0}),
            Scalar(linear_fn, {"a": -0.05, "b": 1.0}),
            0.5]
system_model = system.Model(z_fields=z_fields)

# Now we can construct the
# ``_phasefactor.tensorfactory.ZFieldPhaseFactorNodeRank3`` object. We test it
# by building several nodes from it, and then listing all the elements of each
# underlying tensor.
print("Constructing instances of "
      "``_phasefactor.tensorfactory.ZFieldPhaseFactorNodeRank3`` to build "
      "several nodes:\n")

dt = 0.1
L = system_model.L
z_field_phase_factor_node_rank_3_factory = \
    tensorfactory.ZFieldPhaseFactorNodeRank3(system_model, dt)

unformatted_msg_1 = "    Building node for (r, k, n)=({}, {}, {}):"
unformatted_msg_2 = "        node.tensor[{}, {}, {}] = {}"
for r in range(L):
    for n in (1, 5):
        for k in range(n+2):
            msg = unformatted_msg_1.format(r, k, n)
            print(msg)
            node = z_field_phase_factor_node_rank_3_factory.build(r, k, n)
            for j_r_m_lt in range(node.shape[0]):
                for j_r_m in range(4):
                    for j_r_m_gt in range(4):
                        elem = node.tensor[j_r_m_lt, j_r_m, j_r_m_gt]
                        msg = unformatted_msg_2.format(j_r_m_lt, j_r_m,
                                                       j_r_m_gt, elem)
                        print(msg)
            print()
print("\n\n")



# _phasefactor.tensorfactory.ZZCouplerPhaseFactorNodeRank2 test #1.
print("_phasefactor.tensorfactory.ZZCouplerPhaseFactorNodeRank2 test #1")
print("================================================================")

# Need to construct a ``system.Model`` object first.
print("Constructing an instance of ``system.Model``.\n")

def quad_fn(t, a, b):
    return a * t * t + b

zz_couplers = [Scalar(quad_fn, {"a": -0.05, "b": 1.0}),
               Scalar(quad_fn, {"a": 0.05, "b": 0.0})]
system_model = system.Model(z_fields=z_fields, zz_couplers=zz_couplers)

# Now we can construct the
# ``_phasefactor.tensorfactory.ZZcouplerPhaseFactorNodeRank2`` object. We test
# it by building several nodes from it, and then listing all the elements of
# each underlying tensor.
print("Constructing instances of "
      "``_phasefactor.tensorfactory.ZZCouplerPhaseFactorNodeRank2`` to build "
      "several nodes:\n")

z_coupler_phase_factor_node_rank_2_factory = \
    tensorfactory.ZZCouplerPhaseFactorNodeRank2(system_model, dt)

unformatted_msg_1 = "    Building node for (r, k, n)=({}, {}, {}):"
unformatted_msg_2 = "        node.tensor[{}, {}] = {}"
for r in range(L-1):
    for n in (1, 5):
        for k in range(n+2):
            msg = unformatted_msg_1.format(r, k, n)
            print(msg)
            node = z_coupler_phase_factor_node_rank_2_factory.build(r, k, n)
            for j_r_m_gt in range(4):
                for j_r_m_lt in range(4):
                    elem = node.tensor[j_r_m_gt, j_r_m_lt]
                    msg = unformatted_msg_2.format(j_r_m_gt, j_r_m_lt, elem)
                    print(msg)
            print()
print("\n\n")



# _phasefactor.tensorfactory.ZFieldZZCouplerPhaseFactorMPS test #1.
print("_phasefactor.tensorfactory.ZFieldZZCouplerPhaseFactorMPS test #1")
print("================================================================")

# Now we can construct the
# ``_phasefactor.tensorfactory.ZFieldZZCouplerPhaseFactorMPS`` object. We test
# it by building several MPSs from it.
print("Constructing instances of "
      "``_phasefactor.tensorfactory.ZFieldZZCouplerPhaseFactorMPS`` to build "
      "several MPSs:\n")

z_field_zz_coupler_phase_factor_mps_factory = \
    tensorfactory.ZFieldZZCouplerPhaseFactorMPS(system_model, dt)

k = 3
n = 5
mps_nodes = z_field_zz_coupler_phase_factor_mps_factory.build(k, n)
print("    Building MPS for (k, n)=({}, {}):".format(k, n))
print("        # of nodes in MPS: {}".format(len(mps_nodes)))
