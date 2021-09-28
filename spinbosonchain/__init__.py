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

"""``spinbosonchain`` is a Python library for simulating the dynamics of a
generalized spin-boson chain model, where both the :math:`z`- and
:math:`y`-components of the spins are coupled to bosonic baths, rather than only
the :math:`z`-components. The library adopts the quasi-adiabatic path integral
(QUAPI) formalism to express the spin system's reduced density matrix as a
time-discretized path integral, comprising of a series of influence functionals
that encode the non-Markovian dynamics of the system. The path integral is
decomposed into a series of components that can be represented by tensor
networks. ``spinbosonchain`` currently relies heavily on Google's TensorNetwork_
package for its implementation of tensor networks and related operations.

.. _tensornetwork: https://github.com/google/TensorNetwork
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# Load subpackages.



# Load modules.
from . import scalar
from . import system
from . import bath
from . import compress
from . import alg
from . import state
from . import ev
from . import report
from . import version



############################
## Authorship information ##
############################

__author__       = "D-Wave Systems Inc."
__copyright__    = "Copyright 2021"
__credits__      = ["Matthew Fitzpatrick"]
__version__      = version.version
__full_version__ = version.full_version
__maintainer__   = "D-Wave Systems Inc."
__email__        = "support@dwavesys.com"
__status__       = "Development"



###################################
## Useful background information ##
###################################

# See e.g. ``https://docs.python.org/3/reference/import.html#regular-packages``
# for a brief discussion of ``__init__.py`` files.



##################################
## Define classes and functions ##
##################################

# List of public objects in package.
__all__ = ["scalar",
           "system",
           "bath",
           "compress",
           "alg",
           "state",
           "ev",
           "report",
           "version",
           "show_config"]



def show_config():
    """Print information about the version of ``spinbosonchain`` and libraries 
    it uses.

    Parameters
    ----------

    Returns
    -------
    """
    print(version.version_summary)

    return None
