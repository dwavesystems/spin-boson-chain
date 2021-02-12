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

# _phasefactor.tensorfactory.ZFieldPhaseFactorNodeRank2 test #1.
print("_phasefactor.tensorfactory.ZFieldPhaseFactorNodeRank2 test #1")
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
# ``_phasefactor.tensorfactory.ZFieldPhaseFactorNodeRank2`` object. We test it
# by building several nodes from it, and then listing all the elements of each
# underlying tensor.
print("Constructing instances of "
      "``_phasefactor.tensorfactory.ZFieldPhaseFactorNodeRank2`` to build "
      "several nodes:\n")

dt = 0.1
L = system_model.L
M_node_factory = tensorfactory.ZFieldPhaseFactorNodeRank2(system_model, dt)

unformatted_msg_1 = "    Building node for (r, k, n)=({}, {}, {}):"
unformatted_msg_2 = "        node.tensor[{}, {}] = {}"
for r in range(L):
    for n in (1, 5):
        for k in range(n+2):
            msg = unformatted_msg_1.format(r, k, n)
            print(msg)
            node = M_node_factory.build(r, k, n)
            for j_r_m in range(4):
                for j_r_m_prime in range(4):
                    elem = node.tensor[j_r_m, j_r_m_prime]
                    msg = unformatted_msg_2.format(j_r_m, j_r_m_prime, elem)
                    print(msg)
            print()
print("\n\n")



# _phasefactor.tensorfactory.ZZCouplerPhaseFactorNodeRank4 test #1.
print("_phasefactor.tensorfactory.ZZCouplerPhaseFactorNodeRank4 test #1")
print("================================================================")

# Need to construct a ``system.Model`` object first.
print("Constructing an instance of ``system.Model``.\n")

def quad_fn(t, a, b):
    return a * t * t + b

zz_couplers = [Scalar(quad_fn, {"a": -0.05, "b": 1.0}),
               Scalar(quad_fn, {"a": 0.05, "b": 0.0})]
system_model = system.Model(z_fields=z_fields, zz_couplers=zz_couplers)

# Now we can construct the
# ``_phasefactor.tensorfactory.ZZcouplerPhaseFactorNodeRank4`` object. We test
# it by building several nodes from it, and then listing all the elements of
# each underlying tensor.
print("Constructing instances of "
      "``_phasefactor.tensorfactory.ZZCouplerPhaseFactorNodeRank4`` to build "
      "several nodes:\n")

X_node_factory = tensorfactory.ZZCouplerPhaseFactorNodeRank4(system_model, dt)

unformatted_msg_1 = "    Building node for (r, k, n)=({}, {}, {}):"
unformatted_msg_2 = "        node.tensor[{}, {}, {}, {}] = {}"
for r in range(L-1):
    for n in (1, 5):
        for k in range(n+2):
            msg = unformatted_msg_1.format(r, k, n)
            print(msg)
            node = X_node_factory.build(r, k, n)
            for j_r_m in range(4):
                for j_r_m_prime in range(4):
                    for j_rP1_m_prime in range(4):
                        for j_rP1_m in range(4):
                            elem = node.tensor[j_r_m, j_r_m_prime,
                                               j_rP1_m_prime, j_rP1_m]
                            msg = unformatted_msg_2.format(j_r_m,
                                                           j_r_m_prime,
                                                           j_rP1_m_prime,
                                                           j_rP1_m,
                                                           elem)
                            print(msg)
            print()
print("\n\n")
