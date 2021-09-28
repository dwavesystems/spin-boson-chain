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

r"""For specifying parameters related to the tensor network algorithm.
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



##################################
## Define classes and functions ##
##################################

# List of public objects in objects.
__all__ = ["Params"]



class Params():
    r"""The simulation parameters relating to the tensor network algorithm.

    ``spinbosonchain`` adopts the quasi-adiabatic path integral (QUAPI)
    formalism to express the spin system's reduced density matrix as a
    time-discretized path integral, comprising of a series of influence
    functionals that encode the non-Markovian dynamics of the system.

    ``spinbosonchain`` represents the local path influence functionals --- which
    are the objects in the QUAPI formalism that encode all the information
    regarding the non-Markovian dynamics of the system --- by a set of matrix
    product states (MPS's). Similarly, the system's state --- which is
    calculated using the local path influence functionals --- ultimately is
    represented by a MPS. With each time step, the bond dimensions of these
    MPS's representing the aforementioned objects generally increase, thus
    increasing memory requirements and the simulation runtime. To counter this,
    one can MPS compression for more efficient computations. For further
    discussion on MPS compression in ``spinbosonchain``, see the documentation
    for the :class:`spinbosonchain.compress.Params` class.

    Parameters
    ----------
    dt : `float`
        The simulation time step size. Expected to be positive.
    temporal_compress_params : :class:`spinbosonchain.compress.Params`
        The parameters used in compressing MPS's that span time, e.g. the MPS's
        representing local path influence functionals
    spatial_compress_params : `None` | :class:`spinbosonchain.compress.Params`
        The parameters used in compressing MPS's that span space, e.g. the MPS
        representing the system's state, i.e. the system's reduced density 
        matrix. If the system consists of a single-spin, then 
        ``spatial_compress_params`` is effectively ignored as it is not 
        needed/used. In this case, one need not specify 
        ``spatial_compress_params``. By default, ``spatial_compress_params`` is 
        set to ``temporal_compress_params``.

    Attributes
    ----------
    dt : `float`, read-only
        The simulation time step size.
    temporal_compress_params : :class:`spinbosonchain.compress.Params`, read-only
        The parameters used in compressing MPS's that span time, e.g. the MPS's
        representing local path influence functionals
    spatial_compress_params : :class:`spinbosonchain.compress.Params`, read-only
        The parameters used in compressing MPS's that span space, e.g. the MPS
        representing the system's state, i.e. the system's reduced density 
        matrix.
    """
    def __init__(self,
                 dt,
                 temporal_compress_params,
                 spatial_compress_params=None):
        if dt <= 0.0:
            raise ValueError("The parameter `dt` must be a positive number.")
        
        self.dt = dt
        self.temporal_compress_params = temporal_compress_params

        self.spatial_compress_params = (temporal_compress_params
                                        if spatial_compress_params is None
                                        else spatial_compress_params)

        return None
