#!/usr/bin/env python
r"""For calculating the expectation values of certain observables.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For general array handling.
import numpy as np

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# For creating system state objects.
from sbc.state import trace
from sbc.state import _apply_1_legged_nodes_to_system_state_mps



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
__all__ = ["single_site_spin_op",
           "multi_site_spin_op",
           "nn_two_site_spin_op",
           "energy"]



def single_site_spin_op(op_string,
                        system_state,
                        site_indices=None,
                        normalize=False):
    r"""Calculate the expectation value of a given single-site spin operator.

    This function calculates the expectation value of a given single-site spin
    operator, as specified by the string ``op_string``, with respect to the 
    system state represented by the :obj:`sbc.state.SystemState` object 
    ``system_state``, at the spin sites specified by the integer array 
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
    system_state : :class:`sbc.state.SystemState`
        The system state.
    site_indices : `None` | `array_like` (`int`, ndim=1), optional
        The site indices corresponding to the sites at which to evaluate the
        expectation value of the single-site spin operator. If set to `None`,
        then ``site_indices`` is reset to ``range(system_state.L)``, i.e. the
        expectation value is calculated at all spin sites.
    normalize : `bool`, optional
        Since the QUAPI algorithm does not preserve the unitarity of the time
        evolution of the system state, the system state may not be properly
        normalized, i.e. its trace may not be equal to 1. If ``normalize`` is to
        ``True``, then the system state is renormalized such that its trace is
        equal to 1, after which the exceptation value of the single-site spin
        operator is calculated. Otherwise, the system is not renormalized. Note
        that ``system_state`` is not actually modified in the renormalization
        procedure.

    Returns
    -------
    result : `array_like` (`complex`, shape=(``len(site_indices)``,))
        For ``0<=r<len(site_indices)``, ``result[r]`` is the expectaion value
        of the single-site spin operator at site ``site_indices[r]``.
    """
    if site_indices == None:
        L = system_state.system_model.L
        site_indices = range(L)

    result = []

    try:
        for site_idx in site_indices:
            op_strings = ['id'] * L
            op_strings[site_idx] = op_string

            ev = multi_site_spin_op(op_strings,
                                    system_state,
                                    normalize=normalize)
            result.append(ev)

    except IndexError:
        raise IndexError("Valid site indices range from 0 to `L-1`, where "
                         "`L` is the number of spin sites in the system.")

    return result



def multi_site_spin_op(op_strings, system_state, normalize=False):
    r"""Calculate the expectation value of a given multi-site spin operator.

    This function calculates the expectation value of a given multi-site spin
    operator, as specified by the array of strings ``op_strings``, with respect 
    to the system state represented by the :obj:`sbc.state.SystemState` 
    object ``system_state``.

    Parameters
    ----------
    op_strings : `array_like` (`str`, shape=(``system_state.system_model.L``,))
        For ``0<=i<system_state.system_model.L``, ``op_strings[i]`` is a string specifying
        the single-site spin operator at site ``i``. Only concatenations of the
        strings ``'sx'``, ``'sy'``, ``'sz'``, and ``'id'``, separated by periods
        ``'.'``, are accepted. E.g. ``'sx.sx.sz'`` represents the
        single-site spin operator 
        :math:`\hat{\sigma}_{x}^2\hat{\sigma}_{z}` and ``'sz'``
        represents :math:`\hat{\sigma}_{z}`.
    system_state : :class:`sbc.state.SystemState`
        The system state.
    normalize : `bool`, optional
        Since the QUAPI algorithm does not preserve the unitarity of the time
        evolution of the system state, the system state may not be properly
        normalized, i.e. its trace may not be equal to 1. If ``normalize`` is to
        ``True``, then the system state is renormalized such that its trace is
        equal to 1, after which the exceptation value of the multi-site spin
        operator is calculated. Otherwise, the system is not renormalized. Note
        that ``system_state`` is not actually modified in the renormalization
        procedure.

    Returns
    -------
    result : `complex`
        The expectation value of the multi-site spin operator.
    """
    if len(op_strings) != system_state.system_model.L:
        raise ValueError("An operator string needs to be specified for each "
                         "spin site: the number of operator strings given does "
                         "not match the number of spin sites.")

    one_legged_nodes = []
    for op_string in op_strings:
        tensor = _array_rep_of_op_string(op_string)
        one_legged_node = tn.Node(tensor)
        one_legged_nodes.append(one_legged_node)

    result = _apply_1_legged_nodes_to_system_state_mps(one_legged_nodes,
                                                       system_state)
    result = complex(result)

    if normalize == True:
        result /= trace(system_state)

    return result



def nn_two_site_spin_op(op_string_1,
                        op_string_2,
                        system_state,
                        bond_indices=None,
                        normalize=False):
    r"""Calculate the expectation value of a given NN two-site spin operator.

    This function calculates the expectation value of a given nearest-neighbour
    (NN) two-site spin operator, with respect to the system state represented by
    the :obj:`sbc.state.SystemState` object ``system_state``, at the NN bonds
    specified by the integer array ``bond_indices``. The NN two-site spin
    operator is specified by two strings: ``op_string_1`` specifies the left
    single-site spin operator, and and ``op_string_2`` specifies the right
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
    system_state : :class:`sbc.state.SystemState`
        The system state.
    bond_indices : `None` | `array_like` (`int`, ndim=1), optional
        The bond indices corresponding to the bonds at which to evaluate the
        expectation value of the NN two-site spin operator. If set to `None`,
        then ``bond_indices`` is reset to 
        ``range(system_state.system_model.L-1)``, i.e. the expectation value is 
        calculated at all bonds.
    normalize : `bool`, optional
        Since the QUAPI algorithm does not preserve the unitarity of the time
        evolution of the system state, the system state may not be properly
        normalized, i.e. its trace may not be equal to 1. If ``normalize`` is to
        ``True``, then the system state is renormalized such that its trace is
        equal to 1, after which the exceptation value of the single-site spin
        operator is calculated. Otherwise, the system is not renormalized. Note
        that ``system_state`` is not actually modified in the renormalization
        procedure.

    Returns
    -------
    result : `array_like` (`complex`, shape=(``len(bond_indices)``,))
        For ``0<=r<len(bond_indices)``, ``result[r]`` is the expectaion value
        of the NN two-site spin operator at bond ``bond_indices[r]``.
    """
    if bond_indices == None:
        L = system_state.system_model.L
        bond_indices = range(L-1)

    result = []

    try:
        for bond_idx in bond_indices:
            op_strings = ['id'] * L
            op_strings[bond_idx] = op_string_1
            op_strings[bond_idx+1] = op_string_2

            ev = multi_site_spin_op(op_strings,
                                    system_state,
                                    normalize=normalize)
            result.append(ev)

    except IndexError:
        raise IndexError("Valid bond indices range from 0 to `L-2`, where "
                         "`L` is the number of spin sites in the system.")

    return result



def energy(system_state, normalize=False):
    r"""Calculate the expectation value of the system's energy.

    This function calculates the expectation value of the system's energy with
    respect to the system state represented by the 
    :obj:`ostfic.state.SystemState` object ``system_state``.

    Parameters
    ----------
    system_state : :class:`sbc.state.SystemState`
        The system state. 
    normalize : `bool`, optional
        Since the QUAPI algorithm does not preserve the unitarity of the time
        evolution of the system state, the system state may not be properly
        normalized, i.e. its trace may not be equal to 1. If ``normalize`` is to
        ``True``, then the system state is renormalized such that its trace is
        equal to 1, after which the exceptation value of the multi-site spin
        operator is calculated. Otherwise, the system is not renormalized. Note
        that ``system_state`` is not actually modified in the renormalization
        procedure.

    Returns
    -------
    result : `float`
        The expectation value of the system's energy.
    """
    t = system_state.t
    x_fields = system_state.system_model.x_fields
    z_fields = system_state.system_model.z_fields
    zz_couplers = system_state.system_model.zz_couplers

    hx = [x_field.eval(t) for x_field in x_fields]
    hz = [z_field.eval(t) for z_field in z_fields]
    Jzz = [zz_coupler.eval(t) for zz_coupler in zz_couplers]
    
    sx = np.array(single_site_spin_op('sx', system_state, normalize=normalize))
    sz = np.array(single_site_spin_op('sz', system_state, normalize=normalize))
    sz_sz = np.array(nn_two_site_spin_op('sz',
                                         'sz',
                                         system_state,
                                         normalize=normalize))
    
    result = (np.dot(hx, sx) + np.dot(hz, sz) + np.dot(Jzz, sz_sz)).real

    return result



def _array_rep_of_op_string(op_string):
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
            raise ValueError("The given operator string is not of the correct "
                             "form: only concatenations of the strings 'sx', "
                             "'sy', 'sz', and 'id', separated by periods '.', "
                             "are accepted.")

    array_rep = array_rep.flatten(order='F')  # Column major order.

    return array_rep
