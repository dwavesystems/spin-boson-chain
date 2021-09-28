# Copyright 2021 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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
