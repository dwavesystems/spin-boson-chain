#!/usr/bin/env python
r"""For specifying how to truncate Schmidt spectra in MPS compression.
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
    r"""Parameters specifying how to truncate Schmidt spectrum in a SVD.

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
    requirements and the simulation runtime. To counter this, one can perform 
    singular value decomposition (SVD) truncations to compress efficiently the 
    MPS's. In short, increasing and decreasing the parameters 
    ``max_num_singular_values`` and ``max_trunc_err`` respectively translate to 
    MPS's with larger bond dimensions :math:`\chi` which generally translates to
    a decrease in the numerical errors of the simulation. However, since the 
    most computationally intensive parts of the simulation scale like 
    :math:`\chi^3`, increasing and decreasing the parameters 
    ``max_num_singular_values`` and ``max_trunc_err`` respectively lead to 
    longer runtimes.

    After the Schmidt spectrum has been truncated according to the parameters
    ``max_num_singular_values`` and ``max_trunc_err``, one can perform an
    additional truncation step by setting the parameter ``rel_tol`` to a
    positive number. If ``rel_tol`` is set as such, then ``sbc`` will discard
    singular values smaller than ``rel_tol * s_max``, where ``s_max`` is the
    largest singular value after the initial truncation procedure has been
    performed.

    For more information on local path influence functionals, and the tensor
    network (TN) algorithm used to calculate the system's state, see our 
    detailed exposition on our QUAPI+TN approach found :manual:`here <>`.

    Parameters
    ----------
    max_num_singular_values : `None` | `int`, optional
        The maximum number of singular values to keep in a SVD.
    max_trunc_err : `None` | `float`, optional
        The maximum allowed truncation error in a SVD truncation. If 
        ``max_num_singular_values`` is set to an `int`, and the truncation error
        cannot go below ``max_trunc_err`` without keeping more than
        ``max_num_singular_values`` singular values, then the parameter
        ``max_trunc_err`` is ignored while performing the SVD truncation.
    rel_tol : `None` | `float`, optional
        The relative tolerance after the initial truncation procedure has been
        performed. If not set to `None`, ``rel_tol`` is expected to be positive.

    Attributes
    ----------
    max_num_singular_values : `None` | `int`, read-only
        The maximum number of singular values to keep in a SVD.
    max_trunc_err : `None` | `float`, read-only
        The maximum allowed truncation error in a SVD truncation. If 
        ``max_num_singular_values`` is set to an `int`, and the truncation error
        cannot go below ``max_trunc_err`` without keeping more than
        ``max_num_singular_values`` singular values, then the attribute
        ``max_trunc_err`` is ignored while performing the SVD truncation.
    rel_tol : `None` | `float`, read-only
        The relative tolerance after the initial truncation procedure has been
        performed. If not set to `None`, ``rel_tol`` is expected to be positive.
    """
    def __init__(self,
                 max_num_singular_values=None,
                 max_trunc_err=None,
                 rel_tol=None):
        if max_num_singular_values == None:
            self.max_num_singular_values = None
        else:
            if max_num_singular_values < 1:
                raise ValueError("The parameter `max_num_singular_values` must "
                                 "be a positive integer or set to type `None`.")

        if max_trunc_err == None:
            self.max_trunc_err = None
        else:
            if max_trunc_err < 0:
                raise ValueError("The parameter `max_trunc_err` must be a "
                                 "non-negative number or set to type `None`.")

        if rel_tol == None:
            self.rel_tol = None
        else:
            if rel_tol <= 0:
                raise ValueError("The parameter `rel_tol` must be a "
                                 "positive number or set to type `None`.")
        
        self.max_num_singular_values = max_num_singular_values
        self.max_trunc_err = max_trunc_err
        self.rel_tol = rel_tol

        return None
