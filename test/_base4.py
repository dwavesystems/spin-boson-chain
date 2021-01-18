#!/usr/bin/env python

#####################################
## Load libraries/packages/modules ##
#####################################

# For timing code snippets.
import timeit



# For evaluating special math functions.
import numpy as np



# Module to test.
from sbc import _base4



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

# _base4.ising_pair_to_base_4 test #1.
print("_base4.ising_pair_to_base_4 test #1")
print("===================================")

print("Evaluating function for all acceptable values:")
unformatted_msg = "    Evaluating for ising_pair={}: Result={}"
print(unformatted_msg.format((1, 1), _base4.ising_pair_to_base_4(1, 1)))
print(unformatted_msg.format((1, -1), _base4.ising_pair_to_base_4(1, -1)))
print(unformatted_msg.format((-1, 1), _base4.ising_pair_to_base_4(-1, 1)))
print(unformatted_msg.format((-1, -1), _base4.ising_pair_to_base_4(-1, -1)))
print("\n\n")



# _base4.base_4_to_ising_pair test #1.
print("_base4.base_4_to_ising_pair test #1")
print("===================================")

print("Evaluating function for all acceptable values:")
unformatted_msg = "    Evaluating for base_4_var={}: Result={}"
print(unformatted_msg.format(0, _base4.base_4_to_ising_pair(0)))
print(unformatted_msg.format(1, _base4.base_4_to_ising_pair(1)))
print(unformatted_msg.format(2, _base4.base_4_to_ising_pair(2)))
print(unformatted_msg.format(3, _base4.base_4_to_ising_pair(3)))
print("\n\n")



# _base4.base_4_to_ising_var test #1.
print("_base4.base_4_to_ising_var test #1")
print("==================================")

print("Evaluating function for all acceptable values:")
unformatted_msg = "    Evaluating for base_4_var={} and alpha={}: Result={}"
print(unformatted_msg.format(0, 1, _base4.base_4_to_ising_var(0, 1)))
print(unformatted_msg.format(1, 1, _base4.base_4_to_ising_var(1, 1)))
print(unformatted_msg.format(2, 1, _base4.base_4_to_ising_var(2, 1)))
print(unformatted_msg.format(3, 1, _base4.base_4_to_ising_var(3, 1)))
print(unformatted_msg.format(0, -1, _base4.base_4_to_ising_var(0, -1)))
print(unformatted_msg.format(1, -1, _base4.base_4_to_ising_var(1, -1)))
print(unformatted_msg.format(2, -1, _base4.base_4_to_ising_var(2, -1)))
print(unformatted_msg.format(3, -1, _base4.base_4_to_ising_var(3, -1)))
