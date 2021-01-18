#!/usr/bin/env python
r"""Contains class definitions for the model parameters of the system.

``sbc`` is a library for simulating the dynamics of a generalized
one-dimensional spin-boson model, where both the :math:`z`- and 
:math:`y`-components of the spins are coupled to bosonic baths, rather than 
only the :math:`z`-components. The Hamiltonian of this model can be broken down
into the following components:

.. math ::
    \hat{H}(t) = \hat{H}^{(A)}(t) + \hat{H}^{(B)} + \hat{H}^{(AB)},
    :label: system_total_Hamiltonian

where :math:`\hat{H}^{(A)}(t)` is the system Hamiltonian, which encodes all
information regarding energies associated exclusively with the spins; 
:math:`\hat{H}^{(B)}` is the bath Hamiltonian, which encodes all information
regarding energies associated with the components of the bosonic environment; 
and :math:`\hat{H}^{(AB)}` is the system-bath coupling Hamiltonian, which 
describes all energies associated with the coupling between the system and the 
environment.

The system Hamiltonian :math:`\hat{H}^{(A)}(t)` is simply the Hamiltonian
of the one-dimensional transverse-field Ising model:

.. math ::
    \hat{H}^{(A)}(t) = \sum_{r=1}^{L} \left\{h_{z; r}(t) \hat{\sigma}_{z; r}
    + h_{x; r}(t) \hat{\sigma}_{x; r}\right\} + \sum_{r=1}^{L-1} 
    J_{z, z; r, r+1}(t) \hat{\sigma}_{z; r} \hat{\sigma}_{z; r+1},
    :label: system_TFIM

with :math:`\sigma_{\nu; r}` being the :math:`\nu^{\mathrm{th}}` Pauli operator
acting on site :math:`r`, :math:`h_{z; r}(t)` being the longitudinal field
energy scale for site :math:`r` at time :math:`t`, :math:`h_{x; r}(t)` being the
transverse field energy scale for site :math:`r` at time :math:`t`, 
:math:`J_{z, z; r, r+1}(t)` being the longitudinal coupling energy scale
between sites :math:`r` and :math:`r+1` at time :math:`t`, and :math:`L` being
the number of sites.

This module contains classes to specify all the model parameters of the system, 
namely the :math:`h_{z; r}(t)`, :math:`h_{x; r}(t)`, and 
:math:`J_{z, z; r, r+1}(t)`.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For deep copies of objects.
import copy

# For checking whether an object is a numerical scalar.
import numbers



# For general array handling.
import numpy as np



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
__all__ = ["ModelParam", "Model"]



def _time_independent_fn(t, fn_result):
    return fn_result



class ModelParam():
    r"""A scalar model parameter.

    Parameters
    ----------
    func_form : `float` | `func` (`float`, `**kwargs`)
        If ``func_form`` is of type `float`, then a time-independent model
        parameter is constructed with a constant value of ``func_form``. In this
        case, the other construction parameter ``func_kwargs`` is ignored. If 
        ``func_form`` is of type `func`, then a time-dependent model 
        parameter is constructed with ``func_form`` being the time-dependent 
        functional form. In this case, the first function argument of 
        ``func_form`` is expected to be time :math:`t`.
    func_kwargs : `dict`, optional
        A dictionary specifying specific values of the keyword arguments of
        ``func_form``. If there are no keyword arguments, then an empty
        dictionary should be passed (i.e. its default value).

    Attributes
    ----------
    func_form : `float` | `func` (`float`, `**kwargs`), read-only
        If ``func_form`` is of type `float`, then the model parameter is
        time-independent with a constant value of ``func_form``. If 
        ``func_form`` is of type `func`, then the model parameter is
        time-dependent with ``func_form`` being the time-dependent functional
        form. In this case, the first function argument of ``func_form`` is
        expected to be time :math:`t`.
    func_kwargs : `dict`, read-only
        A dictionary specifying specific values of the keyword arguments of
        ``func_form``. 
    """
    def __init__(self, func_form, func_kwargs=dict()):
        t = 0

        if isinstance(func_form, numbers.Number):
            self._func_form = _time_independent_fn
            self._func_kwargs = {"fn_result": func_form}
            self.func_kwargs = dict()
        else:
            try:
                func_form(t, **func_kwargs)  # Check TypeErrors.
                self._func_form = func_form
                self._func_kwargs = copy.deepcopy(func_kwargs)
                self.func_kwargs = self._func_kwargs
            except:
                raise TypeError("The given dictionary `func_kwargs` that is "
                                "suppose to specify the keyword arguments of "
                                "the given function `func_form`, used to "
                                "construct an instance of the "
                                "`sbc.system.ModelParam`, is not compatible "
                                "with `func_form`.")
            
        self.func_form = func_form

        return None


    
    def eval(self, t):
        r"""Evaluate model parameter at time ``t``.

        Parameters
        ----------
        t : `float`
            Time.

        Returns
        -------
        result : `float`
            The value of the model parameter at time ``t``.
        """
        result = self._func_form(t, **self._func_kwargs)

        return result



class Model():
    r"""The system's model parameter set.

    The system Hamiltonian :math:`\hat{H}^{(A)}(t)` is simply the Hamiltonian
    of the one-dimensional transverse-field Ising model:

    .. math ::
        \hat{H}^{(A)}(t) = \sum_{r=1}^{L} \left\{h_{z; r}(t) \hat{\sigma}_{z; r}
        + h_{x; r}(t) \hat{\sigma}_{x; r}\right\} + \sum_{r=1}^{L-1} 
        J_{z, z; r, r+1}(t) \hat{\sigma}_{z; r} \hat{\sigma}_{z; r+1},
        :label: system_Model_TFIM

    with :math:`\sigma_{\nu; r}` being the :math:`\nu^{\mathrm{th}}` Pauli 
    operator acting on site :math:`r`, :math:`h_{z; r}(t)` being the 
    longitudinal field energy scale for site :math:`r` at time :math:`t`, 
    :math:`h_{x; r}(t)` being the transverse field energy scale for site 
    :math:`r` at time :math:`t`, :math:`J_{z, z; r, r+1}(t)` being the 
    longitudinal coupling energy scale between sites :math:`r` and :math:`r+1` 
    at time :math:`t`, and :math:`L` being the number of sites. The system
    model parameters refer to the :math:`h_{z; r}(t)`, :math:`h_{x; r}(t)`, and
    :math:`J_{z, z; r, r+1}(t)`.

    Parameters
    ----------
    z_fields : `array_like` (`float` | :class:`sbc.system.ModelParam`, shape=(``L``,)) | `None`, optional
        The longitudinal field energy scales for a spin chain of size ``L``. If
        ``z_fields`` is an array, then ``z_fields[r]`` is the field energy 
        scale for site ``r``. Note that the field energy scales can be either 
        time-dependent or independent. All time-independent field energy scales 
        are converted to trivial instances of :class:`sbc.system.ModelParam`. If
        ``z_fields`` is set to `None` (i.e. the default value), then no
        longitudinal fields are applied to the spins.
    x_fields : `array_like` (`float` | :class:`sbc.system.ModelParam`, shape=(``L``,)) | `None`, optional
        The transverse field energy scales for a spin chain of size ``L``. If
        ``x_fields`` is an array, then ``x_fields[r]`` is the field energy 
        scale for site ``r``. Note that the field energy scales can be either 
        time-dependent or independent. All time-independent field energy scales 
        are converted to trivial instances of :class:`sbc.system.ModelParam`. If
        ``x_fields`` is set to `None` (i.e. the default value), then no
        transverse fields are applied to the spins.
    zz_couplers : `array_like` (`float` | :class:`sbc.system.ModelParam`, shape=(``L-1``,)) | `None`, optional
        The longitudinal coupling energy scales for a spin chain of size ``L``.
        If ``zz_couplers[r]`` is an array, then ``zz_couplers[r]`` is the 
        coupling energy scale for sites ``r`` and ``r+1``. Note that the 
        coupling energy scales can be either time-dependent or independent. All 
        time-independent coupling strengths are converted to trivial instances 
        of :class:`sbc.system.ModelParam`. If ``zz_couplers`` is set to `None`
        (i.e. the default value), then no longitudinal couplers are applied to
        the spins.

    Attributes
    ----------
    z_fields : `array_like` (:class:`sbc.system.ModelParam`, shape=(``L``,)), read-only
        The longitudinal field energy scales for a spin chain of size ``L``.
        ``z_fields[r]`` is the field energy scale for site ``r``. Note that the 
        field energy scales can be either time-dependent or independent. 
    x_fields : `array_like` (:class:`sbc.system.ModelParam`, shape=(``L``,)), read-only
        The transverse field energy scales for a spin chain of size ``L``.
        ``x_fields[r]`` is the field energy scale for site ``r``. Note that the 
        field energy scales can be either time-dependent or independent. 
    zz_couplers : `array_like` (:class:`sbc.system.ModelParam`, shape=(``L-1``,)), read-only
        The longitudinal coupling energy scales for a spin chain of size ``L``.
        ``zz_couplers[r]`` is the coupling energy scales for sites ``r`` and 
        ``r+1``. Note that the coupling energy scales can be either 
        time-dependent or independent. 
    L : `int`, read-only
        Number of sites. This is calculated automatically from ``z_fields``,
        ``x_fields``, and ``zz_couplers``.
    """
    def __init__(self, z_fields=None, x_fields=None, zz_couplers=None):
        self.L = self._determine_L(z_fields, x_fields, zz_couplers) 
        
        if z_fields == None:
            z_fields = np.zeros([self.L])
        if len(z_fields) == 0:
            z_fields = np.zeros([self.L])
        if x_fields == None:
            x_fields = np.zeros([self.L])
        if len(x_fields) == 0:
            x_fields = np.zeros([self.L])
        if zz_couplers == None:
            zz_couplers = np.zeros([self.L-1])
        if len(zz_couplers) == 0:
            zz_couplers = np.zeros([self.L-1])
            
        self.z_fields = []
        for z_field in z_fields:
            if isinstance(z_field, ModelParam):
                self.z_fields += [z_field]
            else:
                self.z_fields += [ModelParam(z_field)]

        self.x_fields = []
        for x_field in x_fields:
            if isinstance(x_field, ModelParam):
                self.x_fields += [x_field]
            else:
                self.x_fields += [ModelParam(x_field)]
                
        self.zz_couplers = []
        for zz_coupler in zz_couplers:
            if isinstance(zz_coupler, ModelParam):
                self.zz_couplers += [zz_coupler]
            else:
                self.zz_couplers += [ModelParam(zz_coupler)]
                
        return None


    
    def _determine_L(self, z_fields, x_fields, zz_couplers):
        candidate_Ls = set()
        if z_fields != None:
            if len(z_fields) != 0:
                candidate_Ls.add(len(z_fields))
        if x_fields != None:
            if len(x_fields) != 0:
                candidate_Ls.add(len(x_fields))
        if zz_couplers != None:
            if len(zz_couplers) != 0:
                candidate_Ls.add(len(zz_couplers)+1)

        if len(candidate_Ls) == 0:
            L = 1
        elif len(candidate_Ls) != 1:
            raise IndexError("Parameters ``z_fields``, ``x_fields``, and "
                             "``zz_couplers`` are of incompatible dimensions: "
                             "the parameters should satisfy `len(z_fields) == "
                             "len(x_fields) == len(zz_couplers)+1`.")
        else:
            L = candidate_Ls.pop()

        return L
