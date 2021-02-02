#!/usr/bin/env python
"""``sbc`` (short for 'Spin-Boson Chain') is a Python library for simulating the
dynamics of a generalized one-dimensional spin-boson model, where both
the :math:`z`- and :math:`y`-components of the spins are coupled to bosonic
baths, rather than only the :math:`z`-components. The library adopts the
quasi-adiabatic path integral (QUAPI) formalism to express the spin system's
reduced density matrix as a time-discretized path integral, comprising of a
series of influence functionals that encode the non-Markovian dynamics of the
system. The path integral is decomposed into a series of components that can be
represented by tensor networks. ``sbc`` currently relies heavily on Google's
TensorNetwork_ package for its implementation of tensor networks and related
operations.

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
from . import trunc
from . import version



############################
## Authorship information ##
############################

__author__       = "Matthew Fitzpatrick"
__copyright__    = "Copyright 2021"
__credits__      = ["Matthew Fitzpatrick"]
__version__      = version.version
__full_version__ = version.full_version
__maintainer__   = "Matthew Fitzpatrick"
__email__        = "mfitzpatrick@dwavesys.com"
__status__       = "Development"



##################################
## Define classes and functions ##
##################################

# List of public objects in package.
__all__ = ["tn",
           "version",
           "show_config"]



def show_config():
    """Print information about the version of ``sbc`` and libraries it uses.

    Parameters
    ----------

    Returns
    -------
    """
    print(version.version_summary)

    return None
