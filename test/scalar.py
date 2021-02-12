#!/usr/bin/env python
r"""This script runs several tests on the :mod:`sbc.system` module."""



#####################################
## Load libraries/packages/modules ##
#####################################

# For deep copies.
import copy



# Import class representing time-dependent scalar model parameters.
from sbc.scalar import Scalar



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

# scalar.Scalar test #1.
print("scalar.Scalar test #1")
print("=====================")

def linear_fn(t, a, b):
    return a*t+b

def quad_fn(t, a, b):
    return a * t * t + b

print("Constructing three scalars.\n")

scalar_1 = Scalar(linear_fn, {"a": 2.0, "b": -1.0})
scalar_2 = Scalar(quad_fn, {"a": 2.0, "b": -3.0})
scalar_3 = Scalar(linear_fn, {"a": 2.0, "b": -1.0})

print("`scalar_1 == scalar_2` evaluates to `{}`".format(scalar_1 == scalar_2))
print("`scalar_1 == scalar_3` evaluates to `{}`".format(scalar_1 == scalar_3))
print()

scalar_list = [scalar_1, scalar_2, scalar_3]
scalar_list_copy = copy.deepcopy(scalar_list)
scalar_set = set(scalar_list)

print("scalar_list:")
print(scalar_list)
print()
print("scalar_set:")
print(scalar_set)
print()

unformatted_msg = "`scalar_list == scalar_list_copy` evaluates to `{}`"
print(unformatted_msg.format(scalar_list == scalar_list_copy))
