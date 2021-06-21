#!/usr/bin/env python
r"""For specifying parameters related to the tensor network algorithm.
"""



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



##################################
## Define classes and functions ##
##################################

# List of public objects in objects.
__all__ = ["Params"]



class Params():
    r"""The simulation parameters relating to the tensor network algorithm.

    ``sbc`` adopts the quasi-adiabatic path integral (QUAPI) formalism to 
    express the spin system's reduced density matrix as a time-discretized
    path integral, comprising of a series of influence functionals that encode
    the non-Markovian dynamics of the system.

    ``sbc`` represents the local path influence functionals --- which are the
    objects in the QUAPI formalism that encode all the information regarding
    the non-Markovian dynamics of the system --- by a set of matrix product
    states (MPS's). Similarly, the system's state --- which is calculated using
    the local path influence functionals --- ultimately is represented by a
    MPS. With each time step, the bond dimensions of these MPS's representing
    the aforementioned objects generally increase, thus increasing memory
    requirements and the simulation runtime. To counter this, one can MPS
    compression for more efficient computations. For further discussion on MPS
    compression in ``sbc``, see the documentation for the 
    :class:`sbc.compress.Params` class.

    Parameters
    ----------
    dt : `float`
        The simulation time step size. Expected to be positive.
    influence_compress_params : :class:`sbc.compress.Params`
        The parameters used in compressing the MPS's representing the local path
        influence functionals.
    state_compress_params : :class:`sbc.compress.Params`
        The parameters used in compressing the MPS representing the system's 
        state, i.e. the system's reduced density matrix. If the system consists 
        of a single-spin, then ``state_compress_params`` is ignored as it is not
        needed/used.

    Attributes
    ----------
    dt : `float`, read-only
        The simulation time step size.
    influence_compress_params : :class:`sbc.compress.Params`, read-only
        The parameters used in compressing the MPS's representing the local path
        influence functionals.
    state_compress_params : :class:`sbc.compress.Params`, read-only
        The parameters used in compressing the MPS representing the system's 
        state, i.e. the system's reduced density matrix.
    """
    def __init__(self, dt, influence_compress_params, state_compress_params):
        if dt <= 0.0:
            raise ValueError("The parameter `dt` must be a positive number.")
        
        self.dt = dt
        self.influence_compress_params = influence_compress_params
        self.state_compress_params = state_compress_params

        return None
