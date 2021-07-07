#!/usr/bin/env python
r"""For specifying parameters related to MPS compression.
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
    r"""Parameters specifying how to compress a matrix-product state.

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

    One can improve the compression of a finite MPS by performing a subsequent
    set of variational compression sweeps. This may be worthwhile for a MPS
    representing a local path influence functional, however it is likely to
    be too costly for a MPS that spans space (e.g. one representing the system
    state) rather than time (e.g. a local path influence functional). It is
    recommended to consider only this feature for MPS's spanning time. Moreover,
    this variational compression feature is not available infinite MPS's
    spanning space.

    For more information on local path influence functionals, and the tensor
    network (TN) algorithm used to calculate the system's state, see our 
    detailed exposition on our QUAPI+TN approach found :manual:`here <>`.

    Parameters
    ----------
    method : ``"zip-up"`` | ``"direct"``, optional
        The method used to apply a MPO to a MPS with subsequent compression.
        If ``method="zip-up"`` (i.e. the default value) then the zip-up method 
        is used (see e.g. Ref. [Paeckel1]_ for a discussion of the method). If 
        ``method="direct"`` then the MPO is directly applied to the MPS,
        followed by a full SVD truncation sweep that goes from right to left, 
        then left to right. The zip-up method is faster, though it may be 
        somewhat less accurate. The smaller your timestep, the more accurate the
        zip-up method will be compared to the direct method. If one subsequently
        applies one or more variational compression sweeps, then it should not 
        matter much whether the zip-up or the direct method is used. For finite
        MPS's, we recommend using the zip-up method, followed by one or more
        variational sweeps.
    max_num_singular_values : `None` | `int`, optional
        The maximum number of singular values to keep in a SVD.
    max_trunc_err : `None` | `float`, optional
        The maximum allowed truncation error in a SVD truncation. If 
        ``max_num_singular_values`` is set to an `int`, and the truncation error
        cannot go below ``max_trunc_err`` without keeping more than
        ``max_num_singular_values`` singular values, then the parameter
        ``max_trunc_err`` is ignored while performing the SVD truncation.
    svd_rel_tol : `None` | `float`, optional
        The relative tolerance after the initial truncation procedure has been
        performed. If not set to `None`, ``svd_rel_tol`` is expected to be 
        positive.
    max_num_var_sweeps : `int`, optional
        The maximum number of additional variational compression sweeps to 
        perform. Note that a single sweep goes right to left, then left to 
        right. If ``max_num_var_sweeps=0`` and/or the MPS is infinite, then no 
        sweeps are performed.
    var_rel_tol : `float`, optional
        To check the accuracy of the variational compression, we track

        .. math ::
            \frac{\left|\left|\psi\right\rangle
            -\left|\tilde{\psi}\right\rangle\right|}
            {\left|\left|\psi\right\rangle\right|},

        where :math:`\left|\psi\right\rangle` is the MPS to be compressed,
        and :math:`\left|\tilde{\psi}\right\rangle` is the compressed MPS. 
        Before starting a new sweep, the above quantity is calculated. If said
        quantity is less than ``var_rel_tol`` then no new sweep is initiated
        and the variational compression procedure stops, returning the current
        compressed MPS. This parameter is ignored for infinite MPS's.

    Attributes
    ----------
    method : ``"zip-up"`` | ``"direct"``, read-only
        The method used to apply a MPO to a MPS with subsequent compression.
    max_num_singular_values : `None` | `int`, read-only
        The maximum number of singular values to keep in a SVD.
    max_trunc_err : `None` | `float`, read-only
        The maximum allowed truncation error in a SVD truncation. If 
        ``max_num_singular_values`` is set to an `int`, and the truncation error
        cannot go below ``max_trunc_err`` without keeping more than
        ``max_num_singular_values`` singular values, then the attribute
        ``max_trunc_err`` is ignored while performing the SVD truncation.
    svd_rel_tol : `None` | `float`, read-only
        The relative tolerance after the initial truncation procedure has been
        performed. If not set to `None`, ``svd_rel_tol`` is expected to be 
        positive.
    max_num_var_sweeps : `int`, read-only
        The maximum number of additional variational compression sweeps to 
        perform. Note that a single sweep goes right to left, then left to 
        right. If ``max_num_var_sweeps=0`` and/or the MPS is infinite, then no 
        sweeps are 
    var_rel_tol : `float`, optional
        To check the accuracy of the variational compression, we track

        .. math ::
            \frac{\left|\left|\psi\right\rangle
            -\left|\tilde{\psi}\right\rangle\right|}
            {\left|\left|\psi\right\rangle\right|},

        where :math:`\left|\psi\right\rangle` is the MPS to be compressed,
        and :math:`\left|\tilde{\psi}\right\rangle` is the compressed MPS. 
        Before starting a new sweep, the above quantity is calculated. If said
        quantity is less than ``var_rel_tol`` then no new sweep is initiated
        and the variational compression procedure stops, returning the current
        compressed MPS.
    """
    def __init__(self,
                 method="zip-up",
                 max_num_singular_values=None,
                 max_trunc_err=None,
                 svd_rel_tol=None,
                 max_num_var_sweeps=0,
                 var_rel_tol=1e-6):
        if (method != "zip-up") and (method != "direct"):
            raise ValueError(_params_init_err_msg_1)
            
        if max_num_singular_values is not None:
            if max_num_singular_values < 1:
                raise ValueError(_params_init_err_msg_2)

        if max_trunc_err is not None:
            if max_trunc_err < 0:
                raise ValueError(_params_init_err_msg_3)

        if svd_rel_tol is not None:
            if svd_rel_tol <= 0:
                raise ValueError(_params_init_err_msg_4)

        if max_num_var_sweeps < 0:
            raise ValueError(_params_init_err_msg_5)

        if var_rel_tol <= 0:
            raise ValueError(_params_init_err_msg_6)

        self.method = method
        self.max_num_singular_values = max_num_singular_values
        self.max_trunc_err = max_trunc_err
        self.svd_rel_tol = svd_rel_tol
        self.max_num_var_sweeps = max_num_var_sweeps
        self.var_rel_tol = var_rel_tol

        return None



_params_init_err_msg_1 = \
    ("The parameter `method` must be set to either 'zip-up' or 'direct'.")
_params_init_err_msg_2 = \
    ("The parameter `max_num_singular_values` must be a positive integer or "
     "set to type `None`.")
_params_init_err_msg_3 = \
    ("The parameter `max_trunc_err` must be a non-negative number or set to "
     "type `None`.")
_params_init_err_msg_4 = \
    ("The parameter `svd_rel_tol` must be a positive number or set to type "
     "`None`.")
_params_init_err_msg_5 = \
    ("The parameter `max_num_var_sweeps` must be a non-negative number.")
_params_init_err_msg_6 = \
    ("The parameter `var_rel_tol` must be a positive number.")
