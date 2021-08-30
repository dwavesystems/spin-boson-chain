#!/usr/bin/env python
r"""For calculating the expectation values of certain observables.

Due to the way MPS's are normalized upon truncations or applications of MPO's in
``spinbosonchain``, the MPS that encodes the current state of the system might
not yield a unit trace of the system's reduced density matrix. As a result, all
calculations of expectation values in ``spinbosonchain`` are renormalized by the
trace of the system's reduced density matrix, i.e.

.. math ::
    \left\langle\hat{O}^{\left(A\right)}\left(t\right)\right\rangle 
    = \frac{\text{Tr}^{\left(A\right)}\left\{
    \hat{\rho}^{\left(A\right)}\left(t\right)
    \hat{O}^{\left(A\right)}\left(t\right)\right\}}
    {\text{Tr}^{\left(A\right)}
    \left\{\hat{\rho}^{\left(A\right)}\left(t\right)\right\}},
    :label: ev_calculating_ev

where :math:`\hat{O}^{\left(A\right)}\left(t\right)` is an operator in the
Heisenberg picture defined on the system's Hilbert space, 
:math:`\hat{\rho}^{\left(A\right)}\left(t\right)` is the system's reduced state
operator at time :math:`t`, and 
:math:`\text{Tr}^{\left(A\right)}\left\{ \cdots\right\}` is the partial trace
with respect to the system degrees of freedom.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For general array handling.
import numpy as np

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# Assign an alias to the ``spinbosonchain`` library.
import spinbosonchain as sbc



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
__all__ = ["single_site_spin_op",
           "multi_site_spin_op",
           "nn_two_site_spin_op",
           "energy"]



def single_site_spin_op(op_string,
                        system_state,
                        site_indices=None):
    r"""Calculate the expectation value of a given single-site spin operator.

    The documentation for this function makes reference to the concept of a 
    'unit cell', which is introduced in the documentation for the module 
    :mod:`spinbosonchain.system`.

    This function calculates the expectation value of a given single-site spin
    operator, as specified by the string ``op_string``, with respect to the
    system state represented by the :obj:`spinbosonchain.state.SystemState`
    object ``system_state``, at the spin sites specified by the integer array
    ``site_indices``.

    Parameters
    ----------
    op_string : `str`
        A string specifying the single-site spin operator. Only concatenations
        of the strings ``'sx'``, ``'sy'``, ``'sz'``, and ``'id'``, separated by
        periods ``'.'``, are accepted. E.g. ``'sx.sx.sz'`` represents the
        single-site spin operator 
        :math:`\hat{\sigma}_{x}^2\hat{\sigma}_{z}` and ``'sz'``
        represents :math:`\hat{\sigma}_{z}`.
    system_state : :class:`spinbosonchain.state.SystemState`
        The system state.
    site_indices : `None` | `array_like` (`int`, ndim=1), optional
        The site indices corresponding to the sites at which to evaluate the
        expectation value of the single-site spin operator. If set to `None`,
        then ``site_indices`` is reset to ``range(system_state.L)``, i.e. the
        expectation value is calculated at each spin site in the :math:`u=0`
        unit cell. Note that in the case of a finite chain there is only one 
        unit cell (i.e. the :math:`u=0` unit cell), whereas for an infinite 
        chain there is an arbitrarily large number of unit cells.

    Returns
    -------
    result : `array_like` (`complex`, shape=(``len(site_indices)``,))
        For ``0<=r<len(site_indices)``, ``result[r]`` is the expectaion value
        of the single-site spin operator at site ``site_indices[r]``.

    """
    # DM: Detailed manuscript.

    # This function essentially implements a special case of Eq. (252) of
    # DM. For additional context see Sec. 4.10 of DM.
    
    L = system_state.system_model.L
    is_infinite = system_state.system_model.is_infinite
    
    if site_indices == None:
        site_indices = range(L)

    result = []

    try:
        for site_idx in site_indices:
            op_strings = ['id'] * L
            site_idx = site_idx % L if is_infinite else site_idx
            op_strings[site_idx] = op_string

            ev = multi_site_spin_op(op_strings, system_state)
            result.append(ev)

    except IndexError:
        raise IndexError(_single_site_spin_op_err_msg_1)

    return result



def multi_site_spin_op(op_strings, system_state):
    r"""Calculate the expectation value of a given multi-site spin operator.

    The documentation for this function makes reference to the concept of a 
    'unit cell', which is introduced in the documentation for the module 
    :mod:`spinbosonchain.system`.

    This function calculates the expectation value of a given multi-site spin
    operator, as specified by the array of strings ``op_strings``, with respect
    to the system state represented by the
    :obj:`spinbosonchain.state.SystemState` object ``system_state``.

    Parameters
    ----------
    op_strings : `array_like` (`str`, ndim=1)
        ``op_strings`` is expected to be a one-dimensional array of size
        ``M*system_state.system_model.L``, where ``M`` can be any positive
        integer for infinite chains, and ``M=1`` for finite chains.
        ``op_strings[r]`` is a string specifying the single-site spin operator 
        at site ``r``. Only concatenations of the strings ``'sx'``, ``'sy'``, 
        ``'sz'``, and ``'id'``, separated by periods ``'.'``, are accepted. 
        E.g. ``'sx.sx.sz'`` represents the single-site spin operator 
        :math:`\hat{\sigma}_{x}^2\hat{\sigma}_{z}` and ``'sz'`` represents 
        :math:`\hat{\sigma}_{z}`.
    system_state : :class:`spinbosonchain.state.SystemState`
        The system state.

    Returns
    -------
    result : `complex`
        The expectation value of the multi-site spin operator.
    """
    # DM: Detailed manuscript.

    # This function implements Eq. (252) of DM. For additional context see
    # Sec. 4.10 of DM.
    
    L = system_state.system_model.L
    if system_state.system_model.is_infinite:
        if len(op_strings) % L != 0:
            raise ValueError(_multi_site_spin_op_err_msg_1a)
    else:
        if len(op_strings) != L:
            raise ValueError(_multi_site_spin_op_err_msg_1b)

    one_legged_nodes = []
    for op_string in op_strings:
        tensor = _array_rep_of_op_string(op_string)
        one_legged_node = tn.Node(tensor)
        one_legged_nodes.append(one_legged_node)

    kwargs = {"physical_1_legged_nodes": one_legged_nodes,
              "system_state": system_state}
    result = sbc.state._apply_1_legged_nodes_to_system_state_mps(**kwargs)
    result = complex(result)

    return result



def nn_two_site_spin_op(op_string_1,
                        op_string_2,
                        system_state,
                        bond_indices=None):
    r"""Calculate the expectation value of a given NN two-site spin operator.

    The documentation for this function makes reference to the concept of a 
    'unit cell', which is introduced in the documentation for the module 
    :mod:`spinbosonchain.system`.

    This function calculates the expectation value of a given nearest-neighbour
    (NN) two-site spin operator, with respect to the system state represented by
    the :obj:`spinbosonchain.state.SystemState` object ``system_state``, at the
    NN bonds specified by the integer array ``bond_indices``. The NN two-site
    spin operator is specified by two strings: ``op_string_1`` specifies the
    left single-site spin operator, and and ``op_string_2`` specifies the right
    single-site spin operator.

    Parameters
    ----------
    op_string_1 : `str`
        A string specifying the left single-site spin operator. Only 
        concatenations of the strings ``'sx'``, ``'sy'``, ``'sz'``, and 
        ``'id'``, separated by periods ``'.'``, are accepted. E.g. 
        ``'sx.sx.sz'`` represents the single-site spin operator 
        :math:`\hat{\sigma}_{x}^2\hat{\sigma}_{z}` and ``'sz'``
        represents :math:`\hat{\sigma}_{z}`.
    op_string_2 : `str`
        A string specifying the right single-site spin operator. Only 
        concatenations of the strings ``'sx'``, ``'sy'``, ``'sz'``, and 
        ``'id'``, separated by periods ``'.'``, are accepted. E.g. 
        ``'sx.sx.sz'`` represents the single-site spin operator 
        :math:`\hat{\sigma}_{x}^2\hat{\sigma}_{z}` and ``'sz'``
        represents :math:`\hat{\sigma}_{z}`.
    system_state : :class:`spinbosonchain.state.SystemState`
        The system state.
    bond_indices : `None` | `array_like` (`int`, ndim=1), optional
        The bond indices corresponding to the bonds at which to evaluate the
        expectation value of the NN two-site spin operator. If set to `None`,
        then ``bond_indices`` is reset to 
        ``range(system_state.system_model.L-1)`` for finite chains, and
        ``range(system_state.system_model.L)`` for infinite chains. In other
        words, by default the expectation value is calculated at each bond in
        the :math:`u=0` unit cell for both finite and infinite chains, but also 
        the bond between the :math:`u=0` and :math:`u=1` unit cells for infinite
        chains. Note that in the case of a finite chain there is only one unit 
        cell (i.e. the :math:`u=0` unit cell), whereas for an infinite chain 
        there is an arbitrarily large number of unit cells.

    Returns
    -------
    result : `array_like` (`complex`, shape=(``len(bond_indices)``,))
        For ``0<=r<len(bond_indices)``, ``result[r]`` is the expectaion value
        of the NN two-site spin operator at bond ``bond_indices[r]``.
    """
    # DM: Detailed manuscript.

    # This function essentially implements a special case of Eq. (252) of
    # DM. For additional context see Sec. 4.10 of DM.
    
    L = system_state.system_model.L
    is_infinite = system_state.system_model.is_infinite
    
    if bond_indices == None:
        bond_indices = range(L - 1 + int(is_infinite))

    result = []

    try:
        for bond_idx in bond_indices:
            op_strings = ['id'] * L
            if is_infinite:
                bond_idx = bond_idx % L
                if bond_idx == L-1:
                    op_strings += op_strings
            op_strings[bond_idx] = op_string_1
            op_strings[bond_idx+1] = op_string_2

            ev = multi_site_spin_op(op_strings, system_state)
            result.append(ev)

    except IndexError:
        raise IndexError(_nn_two_site_spin_op_err_msg_1)

    return result



def energy(system_state):
    r"""Calculate the expectation value of the system's energy.

    The documentation for this function makes reference to the concept of a 
    'unit cell', which is introduced in the documentation for the module 
    :mod:`spinbosonchain.system`.

    This function calculates the expectation value of the system's :math:`u=0`
    unit cell energy with respect to the system state represented by the 
    :obj:`ostfic.state.SystemState` object ``system_state``. This is the
    quantity :math:`\left\langle\hat{H}_{u=0}^{\left(A\right)}
    \left(t\right)\right\rangle` where :math:`\hat{H}_{u}^{\left(A\right)}
    \left(t\right)` is given by Eq. :eq:`system_TFIM`.

    Note that in the case of a finite chain there is only one unit cell (i.e. 
    the :math:`u=0` unit cell), whereas for an infinite chain there is an 
    arbitrarily large number of unit cells.

    Parameters
    ----------
    system_state : :class:`spinbosonchain.state.SystemState`
        The system state. 

    Returns
    -------
    result : `float`
        The expectation value of the system's :math:`u=0` unit cell energy.

    """
    # DM: Detailed manuscript.

    # This function implements Eq. (258) of DM, where it makes specific use of
    # Eq. (252) of DM. For additional context see Sec. 4.10 of DM.
    
    t = system_state.t
    x_fields = system_state.system_model.x_fields
    z_fields = system_state.system_model.z_fields
    zz_couplers = system_state.system_model.zz_couplers

    hx = [x_field.eval(t) for x_field in x_fields]
    hz = [z_field.eval(t) for z_field in z_fields]
    Jzz = [zz_coupler.eval(t) for zz_coupler in zz_couplers]
    
    sx = np.array(single_site_spin_op('sx', system_state))
    sz = np.array(single_site_spin_op('sz', system_state))
    sz_sz = np.array(nn_two_site_spin_op('sz', 'sz', system_state))
    
    result = (np.dot(hx, sx) + np.dot(hz, sz) + np.dot(Jzz, sz_sz)).real

    return result



def _array_rep_of_op_string(op_string):
    # DM: Detailed manuscript.

    # This function constructs array representation of Ising spin operators in
    # the base-4 basis. See Secs. 4.1 and 4.2 for discussions on base-4
    # variables.
    
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    identity = np.array([[1, 0], [0, 1]], dtype=np.complex128)

    array_rep = identity

    ops = op_string.split(".")
    for op in ops:
        if op == 'sx':
            array_rep = np.matmul(array_rep, sigma_x)
        elif op == 'sy':
            array_rep = np.matmul(array_rep, sigma_y)
        elif op == 'sz':
            array_rep = np.matmul(array_rep, sigma_z)
        elif op == 'id':
            array_rep = np.matmul(array_rep, identity)
        else:
            raise ValueError(_array_rep_of_op_string_err_msg_1)

    array_rep = array_rep.flatten(order='F')  # Column major order.

    return array_rep



_single_site_spin_op_err_msg_1 = \
    ("Valid site indices range from 0 to `L-1`, where `L` is the number of "
     "spin sites in the system.")

_multi_site_spin_op_err_msg_1a = \
    ("The number of operator strings should be a positive multiple of the "
     "unit cell size.")
_multi_site_spin_op_err_msg_1b = \
    ("An operator string needs to be specified for each spin site: the number "
     "of operator strings given does not match the number of spin sites in the "
     "spin system.")

_nn_two_site_spin_op_err_msg_1 = \
    ("Valid bond indices range from 0 to `L-2`, where `L` is the number of "
     "spin sites in the system.")

_array_rep_of_op_string_err_msg_1 = \
    ("The given operator string is not of the correct form: only "
     "concatenations of the strings 'sx', 'sy', 'sz', and 'id', separated by "
     "periods '.', are accepted.")
