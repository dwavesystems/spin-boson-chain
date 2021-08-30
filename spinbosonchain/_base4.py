#!/usr/bin/env python
r"""This module contains functions for mapping between Ising variable pairs and
base-4 variables. See Secs. 4.1 and 4.2 of the detailed manuscript on our 
QUAPI-TN approach for context.
"""


#####################################
## Load libraries/packages/modules ##
#####################################



############################
## Authorship information ##
############################

__author__     = "D-Wave Systems Inc."
__copyright__  = "Copyright 2021"
__credits__    = ["Matthew Fitzpatrick"]
__maintainer__ = "D-Wave Systems Inc."
__email__      = "support@dwavesys.com"
__status__     = "Development"



####################################
## Define functions and constants ##
####################################

def ising_pair_to_base_4(left_ising_var, right_ising_var):
    r"""This function implements Eq. (93) of the detailed manuscript (DM) on our
    QUAPI-TN approach. See Secs. 4.1 and 4.2 of the DM for further context.
    """
    base_4_var = (1-left_ising_var) + (1-right_ising_var) // 2

    return base_4_var



_base_4_to_ising_pair_map = ((1, 1), (1, -1), (-1, 1), (-1, -1))



def base_4_to_ising_pair(base_4_var):
    r"""This function implements the inverse of Eq. (93) of the detailed
    manuscript (DM) on our QUAPI-TN approach. See Secs. 4.1 and 4.2 of DM for 
    further context.
    """
    ising_pair = _base_4_to_ising_pair_map[base_4_var]

    return ising_pair



def base_4_to_ising_var(base_4_var, alpha):
    r"""This function implements Eq. (94) of the detailed manuscript (DM) on our
    QUAPI-TN approach. See Secs. 4.1 and 4.2 of DM for further context.
    """
    ising_var = _base_4_to_ising_pair_map[base_4_var][(1-alpha)//2]

    return ising_var
