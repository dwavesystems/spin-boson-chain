#!/usr/bin/env python
r"""Contains class definition for the model parameters of the system.

``sbc`` is a library for simulating the dynamics of a generalized
one-dimensional spin-boson chain model, where both the :math:`z`- and 
:math:`y`-components of the spins are coupled to bosonic baths, rather than 
only the :math:`z`-components. A convenient way to discuss both finite and
infinite chains is to express the Hamiltonian of the aforementioned spin-boson
model as a sum of :math:`2N+1` 'unit cell' Hamiltonians:

.. math ::
    \hat{H}\left(t\right)\equiv\sum_{u=-N}^{N}\hat{H}_{u}\left(t\right),
    :label: system_total_Hamiltonian

where :math:`N` is a non-negative integer, and :math:`\hat{H}_{u}\left(t\right)`
is the Hamiltonian of the :math:`u^{\mathrm{th}}` 'unit cell' of the model:

.. math ::
    \hat{H}_{u}\left(t\right)=\hat{H}_{u}^{\left(A\right)}\left(t\right)
    +\hat{H}_{u}^{\left(B\right)}+\hat{H}_{u}^{\left(AB\right)},
    :label: system_unit_cell_Hamiltonian

with :math:`\hat{H}_{u}^{\left(A\right)}\left(t\right)` being the system part of
:math:`\hat{H}_{u}\left(t\right)`, which encodes all information regarding
energies associated exclusively with the spins;
:math:`\hat{H}_{u}^{\left(B\right)}` being the bath part of
:math:`\hat{H}_{u}\left(t\right)`, which encodes all information regarding
energies associated with the components of the bosonic environment; and
:math:`\hat{H}_{u}^{\left(AB\right)}` is the system-bath coupling part of
:math:`\hat{H}_{u}\left(t\right)`, which describes all energies associated with
the coupling between the system and the environment.

:math:`\hat{H}_{u}^{\left(A\right)}\left(t\right)` is simply an instance of the
one-dimensional transverse-field Ising model:

.. math ::
    \hat{H}_{u}^{\left(A\right)}\left(t\right) 
    & \equiv\sum_{r=0}^{L-1}\left\{ h_{z;r}\left(t\right)\hat{\sigma}_{z;r+uL}
    +h_{x;r}\left(t\right)\hat{\sigma}_{x;r+uL}\right\} \\
    & \mathrel{\phantom{\equiv}}\mathop{+}\sum_{r=0}^{L-1}
    J_{z,z;r,r+1}\left(t\right)\hat{\sigma}_{z;r+uL}\hat{\sigma}_{z;r+uL+1},
    :label: system_TFIM

with :math:`\sigma_{\nu; r}` being the :math:`\nu^{\mathrm{th}}` Pauli operator
acting on site :math:`r`, :math:`h_{z; r}(t)` being the longitudinal field
energy scale for site :math:`r` at time :math:`t`, :math:`h_{x; r}(t)` being the
transverse field energy scale for site :math:`r` at time :math:`t`, 
:math:`J_{z, z; r, r+1}(t)` being the longitudinal coupling energy scale
between sites :math:`r` and :math:`r+1` at time :math:`t`, and :math:`L` being
the number of sites in every 'unit cell'.

For finite chains, we set :math:`N=0` and :math:`J_{z,z;L-1,L}\left(t\right)=0`,
whereas for infinite chains, we take the limit of :math:`N\to\infty`.

This module defines a class for specifying all the model parameters of the 
system, namely the :math:`h_{z; r}(t)`, :math:`h_{x; r}(t)`, and 
:math:`J_{z, z; r, r+1}(t)`.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# Import class representing time-dependent scalar model parameters.
from sbc.scalar import Scalar



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
__all__ = ["Model"]



class Model():
    r"""The system's model parameter set.

    The documentation for this class makes reference to the concepts of a 
    'system' and 'unit cell', which are introduced in the documentation for the 
    module :mod:`sbc.system`.

    The Hamiltonian for the system part of the :math:`u^{\mathrm{th}}` unit
    cell is simply an instance of the one-dimensional transverse-field Ising
    model:

    .. math ::
        \hat{H}_{u}^{\left(A\right)}\left(t\right) 
        & \equiv\sum_{r=0}^{L-1}\left\{ h_{z;r}\left(t\right)
        \hat{\sigma}_{z;r+uL}
        +h_{x;r}\left(t\right)\hat{\sigma}_{x;r+uL}\right\} \\
        & \mathrel{\phantom{\equiv}}\mathop{+}\sum_{r=0}^{L-1}
        J_{z,z;r,r+1}\left(t\right)\hat{\sigma}_{z;r+uL}\hat{\sigma}_{z;r+uL+1},
        :label: system_Model_TFIM

    with :math:`\sigma_{\nu; r}` being the :math:`\nu^{\mathrm{th}}` Pauli 
    operator acting on site :math:`r`, :math:`h_{z; r}(t)` being the 
    longitudinal field energy scale for site :math:`r` at time :math:`t`, 
    :math:`h_{x; r}(t)` being the transverse field energy scale for site 
    :math:`r` at time :math:`t`, :math:`J_{z, z; r, r+1}(t)` being the 
    longitudinal coupling energy scale between sites :math:`r` and :math:`r+1` 
    at time :math:`t`, and :math:`L` being the number of sites in every unit 
    cell. 

    The system model parameters refer to the :math:`h_{z; r}(t)`, 
    :math:`h_{x; r}(t)`, and :math:`J_{z, z; r, r+1}(t)`.

    For finite chains, we assume one unit cell (i.e. the :math:`u=0` cell) and
    set :math:`J_{z,z;L-1,L}\left(t\right)=0`, whereas for infinite chains, we
    take the limit of the number of unit cells to infinity.

    Parameters
    ----------
    z_fields : `array_like` (`float` | :class:`sbc.scalar.Scalar`, shape=(``L``,)) | `None`, optional
        The longitudinal field energy scales for a unit cell of size ``L``. If
        ``z_fields`` is an array, then ``z_fields[r]`` is the field energy 
        scale for site ``r``. Note that the field energy scales can be either 
        time-dependent or independent. All time-independent field energy scales 
        are converted to trivial instances of :class:`sbc.scalar.Scalar`. If
        ``z_fields`` is set to `None` (i.e. the default value), then no
        longitudinal fields are applied to the spins.
    x_fields : `array_like` (`float` | :class:`sbc.scalar.Scalar`, shape=(``L``,)) | `None`, optional
        The transverse field energy scales for a unit cell of size ``L``. If
        ``x_fields`` is an array, then ``x_fields[r]`` is the field energy 
        scale for site ``r``. Note that the field energy scales can be either 
        time-dependent or independent. All time-independent field energy scales 
        are converted to trivial instances of :class:`sbc.scalar.Scalar`. If
        ``x_fields`` is set to `None` (i.e. the default value), then no
        transverse fields are applied to the spins.
    zz_couplers : `array_like` (`float` | :class:`sbc.scalar.Scalar`, shape=(``L-1+int(is_infinite)``,)) | `None`, optional
        The longitudinal coupling energy scales for a unit cell of size ``L``.
        If ``zz_couplers[r]`` is an array, then ``zz_couplers[r]`` is the 
        coupling energy scale for sites ``r`` and ``r+1``. Note that the 
        coupling energy scales can be either time-dependent or independent. All 
        time-independent coupling strengths are converted to trivial instances 
        of :class:`sbc.scalar.Scalar`. If ``zz_couplers`` is set to `None`
        (i.e. the default value), then no longitudinal couplers are applied to
        the spins.
    is_infinite : `bool`, optional
        Specifies whether or not the system is an infinite chain: if 
        ``is_infinite`` is set to ``True``, the system is an infinite chain,
        otherwise it is a finite chain.

    Attributes
    ----------
    z_fields : `array_like` (:class:`sbc.scalar.Scalar`, shape=(``L``,)), read-only
        The longitudinal field energy scales for a unit cell of size ``L``.
        ``z_fields[r]`` is the field energy scale for site ``r``. Note that the 
        field energy scales can be either time-dependent or independent. 
    x_fields : `array_like` (:class:`sbc.scalar.Scalar`, shape=(``L``,)), read-only
        The transverse field energy scales for a unit cell of size ``L``.
        ``x_fields[r]`` is the field energy scale for site ``r``. Note that the 
        field energy scales can be either time-dependent or independent. 
    zz_couplers : `array_like` (:class:`sbc.scalar.Scalar`, shape=(``L-1+int(is_infinite)``,)), read-only
        The longitudinal coupling energy scales for a unit cell of size ``L``.
        ``zz_couplers[r]`` is the coupling energy scales for sites ``r`` and 
        ``r+1``. Note that the coupling energy scales can be either 
        time-dependent or independent. 
    L : `int`, read-only
        The number of sites in every unit cell. This is calculated automatically
        from ``z_fields``, ``x_fields``, ``zz_couplers``, and ``is_infinite``.
    is_infinite : `bool`, read-only
        Specifies whether or not the system is an infinite chain: if 
        ``is_infinite`` is set to ``True`` the system is an infinite chain,
        otherwise it is a finite chain.
    """
    def __init__(self,
                 z_fields=None,
                 x_fields=None,
                 zz_couplers=None,
                 is_infinite=False):
        self.L = self._determine_L(z_fields, x_fields, zz_couplers, is_infinite)
        num_couplers = self.L - 1 + int(is_infinite)
        
        self.x_fields = self._construct_attribute(x_fields, self.L)
        self.z_fields = self._construct_attribute(z_fields, self.L)

        self.zz_couplers = self._construct_attribute(zz_couplers, num_couplers)

        self.is_infinite = is_infinite

        self._map_btwn_site_indices_and_unique_x_fields = \
            self._calc_map_btwn_site_indices_and_unique_x_fields()
        
        return None


    
    def _determine_L(self, z_fields, x_fields, zz_couplers, is_infinite):
        candidate_Ls = set()
        if z_fields != None:
            if len(z_fields) != 0:
                candidate_Ls.add(len(z_fields))
        if x_fields != None:
            if len(x_fields) != 0:
                candidate_Ls.add(len(x_fields))
        if zz_couplers != None:
            if len(zz_couplers) != 0:
                candidate_Ls.add(len(zz_couplers)+1-int(is_infinite))

        if len(candidate_Ls) == 0:
            L = 1 + int(is_infinite)
        elif len(candidate_Ls) != 1:
            raise IndexError(_model_determine_L_err_msg_1)
        else:
            L = candidate_Ls.pop()

        return L



    def _construct_attribute(self, ctor_param, array_size):
        if ctor_param == None:
            ctor_param = [0] * array_size
        if len(ctor_param) == 0:
            ctor_param = [0] * array_size

        attribute = ctor_param[:]
        elem_already_set = [False] * self.L
        for idx1 in range(0, array_size):
            if not isinstance(attribute[idx1], Scalar):
                attribute[idx1] = Scalar(attribute[idx1])
            for idx2 in range(idx1+1, array_size):
                if ctor_param[idx2] == ctor_param[idx1]:
                    attribute[idx2] = attribute[idx1]
                    elem_already_set[idx2] = True

        return attribute



    def _calc_map_btwn_site_indices_and_unique_x_fields(self):
        result = list(range(self.L))
        for idx1 in range(self.L):
            for idx2 in range(idx1+1, self.L):
                if self.x_fields[idx2] == self.x_fields[idx1]:
                    result[idx2] = result[idx1]

        return result



_model_determine_L_err_msg_1 = \
    ("Parameters ``z_fields``, ``x_fields``, and ``zz_couplers`` are of "
     "incompatible dimensions.")
