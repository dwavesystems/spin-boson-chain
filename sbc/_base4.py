#!/usr/bin/env python

#####################################
## Load libraries/packages/modules ##
#####################################



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Non-Production"



####################################
## Define functions and constants ##
####################################

def ising_pair_to_base_4(left_ising_var, right_ising_var):
    # This function implements Eq. (72) of the detailed manuscript (DM) on our
    # QUAPI-TN approach. See Sec. 3.2 of DM for further context.
    base_4_var = (1-left_ising_var) + (1-right_ising_var) // 2

    return base_4_var



_base_4_to_ising_pair_map = ((1, 1), (1, -1), (-1, 1), (-1, -1))



def base_4_to_ising_pair(base_4_var):
    # This function implements the inverse of Eq. (72) of the detailed
    # manuscript (DM) on our QUAPI-TN approach. See Sec. 3.2 of DM for further
    # context.
    ising_pair = _base_4_to_ising_pair_map[base_4_var]

    return ising_pair



def base_4_to_ising_var(base_4_var, alpha):
    # This function implements Eq. (73) of the detailed manuscript (DM) on our
    # QUAPI-TN approach. See Sec. 3.2 of DM for further context.
    ising_var = _base_4_to_ising_pair_map[base_4_var][(1-alpha)//2]

    return ising_var
