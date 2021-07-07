#!/usr/bin/env python
r"""For reporting to file instantaneous values of properties of the spin system.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For getting filename and directory paths.
import os

# For making directories (i.e. emulating mkdir).
import pathlib



# For general array handling.
import numpy as np



# For calculating instantaneous values of various properties of the spin system.
import sbc.state
import sbc.ev



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
__all__ = ["WishList", "ReportParams", "report"]



class WishList():
    r"""A list of properties of a spin system to report to file.

    The documentation for this function makes reference to the concept of a 
    'unit cell', which is introduced in the documentation for the module 
    :mod:`sbc.system`.

    An instance of this class can be used to construct a 
    :obj:`sbc.report.ReportParams` object, which itself can be used in
    conjunction with the function :func:`sbc.report.report` to report
    the properties, specified in the aforementioned instance of the
    :class:`sbc.report.WishList` class, of a given spin system to various
    files. All output files are created/updated in an output directory, whose
    path is stored in the attribute 
    :attr:`sbc.report.ReportParams.output_dir` of the 
    :class:`sbc.report.ReportParams` class.

    Parameters
    ----------
    state_trace : `bool`, optional
        If set to ``True``, then the trace of the system's reduced density 
        matrix is to be reported to the output file ``'state-trace.csv'`` upon 
        calling the function :func:`sbc.report.report` using the 
        :obj:`sbc.report.WishList` object. Note that since the QUAPI 
        algorithm used in the ``sbc`` library does not preserve the unitarity
        of the time evolution of the system state, the trace may not necessarily
        evaluate to unity for finite chains. For infinite chains, the
        algorithm explicitly renormalizes the system state such that the
        trace is always unity.
    schmidt_spectrum_sum : `bool`, optional
        Note that ``sbc`` can only calculate the Schmidt spectrum sum for
        **finite** chains. If the system is finite and ``schmidt_spectrum_sum``
        is set to ``True``, then the Schmidt spectrum sum for all bonds is to be
        reported to the output file ``'schmidt-spectrum-sum.csv'`` upon calling 
        the function :func:`sbc.report.report` using the 
        :obj:`sbc.report.WishList` object. See the documentation for the 
        function :func:`sbc.state.schmidt_spectrum_sum` for a discussion 
        regarding Schmidt spectra and Schmidt spectrum sums.
    realignment_criterion : `bool`, optional
        Note that ``sbc`` can only determine whether **finite** chains are
        entangled via the realignment criterion. If the system is finite and 
        ``realignment_criterion`` is set to ``True``, then the realignment 
        criterion is to be applied to the system's current state to determine 
        whether it is entangled, and the result of this operation is to be 
        reported to the output file ``'realignment-criterion.csv'`` upon calling
        the function :func:`sbc.report.report` using the 
        :obj:`sbc.report.WishList` object. See the documentation for the 
        function :func:`sbc.state.realignment_criterion` for a brief discussion 
        regarding the realignment criterion.
    spin_config_probs : ``[]`` | `array_like` (``-1`` | ``1``, shape=(``num_configs``, ``M*L``)), optional
        If ``spin_config_probs`` is of the form 
        `array_like` (``-1`` | ``1``, shape=(``num_configs``, ``M*L``)),
        where ``num_configs`` is a positive integer, ``L`` is the unit cell size
        in the system of interest, and :math:`M` is a positive integer that is 
        less than or equal to the number of unit cells in said system, then 
        ``spin_config_probs[0<=i<num_configs]`` specifies a classical spin
        configuration. Moreover, for ``0<=i<num_configs``, the spin 
        configuration specified in ``spin_config_probs[i]`` is to be written to
        the ``i`` th column of the output file ``'spin-config-list.csv'`` and
        the corresponding spin configuration probability to the output file
        ``'spin-config-'+str(i+1)+'.csv'`` upon calling the function 
        :func:`sbc.report.report` using the :obj:`sbc.report.WishList` 
        object. For finite chains, there is only one unit cell, hence 
        :math:`M=1`, whereas for infinite chains :math:`M` can be any positive
        number. Note that a spin configuration is represented by an array of the
        integers ``-1`` and ``1``, where the former represents a spin-down 
        state, and the latter a spin-up state. If ``spin_config_probs`` is set 
        to ``[]``, i.e. the default value, then no spin configuration 
        probabilities are to be reported.
    ev_of_single_site_spin_ops : ``[]`` | `array_like` (`str`, ndim=1), optional
        ``ev_of_single_site_spin_ops`` is an array of strings, where each
        string specifies a single-site spin operator. Only concatenations of
        ``'sx'``, ``'sy'``, ``'sz'``, and ``'id'``, separated by periods 
        ``'.'``, are accepted elements of the array. E.g. ``'sx.sx.sz'``
        represents the single-site spin operator 
        :math:`\hat{\sigma}_{x}^2\hat{\sigma}_{z}` and ``'sz'``
        represents :math:`\hat{\sigma}_{z}`. If ``ev_of_single_site_spin_ops``
        is not an empty array, then for 
        ``0<=i<len(ev_of_single_site_spin_ops)``, the expectation value of the
        single-site spin operator specified in ``ev_of_single_site_spin_ops[i]``
        is to be evaluated at each site in the :math:`u=0` unit cell. Note that
        in the case of a finite chain there is only one unit cell (i.e. the 
        :math:`u=0` unit cell). The results are to be reported to the output 
        file ``ev_of_single_site_spin_ops[i].replace('.', '')+'.csv'`` upon 
        calling the function :func:`sbc.report.report` using the
        :obj:`sbc.report.WishList` object.
    ev_of_multi_site_spin_ops : ``[]`` | `array_like` (`str`, shape=(``num_ops``, ``M*L``)), optional
        If ``ev_of_multi_site_spin_ops`` is of the form 
        `array_like` (`str`, shape=(``num_ops``, ``M*L``)),
        where ``num_ops`` is a positive integer, and ``L`` and ``M`` were
        introduced in the description of the ``spin_config_probs`` parameter 
        above, then ``ev_of_multi_site_spin_ops[0<=i<num_ops]`` specifies a 
        multi-site spin operator, with 
        ``ev_of_multi_site_spin_ops[0<=i<num_ops][0<=r<M*L]`` specifying the
        single-site spin operator at site ``r`` of the ``i`` th multi-site 
        spin operator [note that the single-site spin operator may be the 
        trivial identity operator]. Single-site spin operators are expressed as 
        strings, where concatenations of ``'sx'``, ``'sy'``, ``'sz'``, and 
        ``'id'``, separated by periods ``'.'``, are accepted values. E.g. 
        ``'sx.sx.sz'`` represents the single-site spin operator 
        :math:`\hat{\sigma}_{x}^2\hat{\sigma}_{z}` and ``'sz'``
        represents :math:`\hat{\sigma}_{z}`. If ``ev_of_multi_site_spin_ops``
        is not an empty array, then for ``0<=i<num_ops``, the multi-site spin
        operator specified in ``ev_of_multi_site_spin_ops[i]`` is to be written
        to the ``i`` th column of the output file 
        ``'multi-site-spin-op-list.csv'`` and the corresponding expectation
        value to the output file ``'multi-site-spin-op-'+str(i+1)+'.csv'`` upon
        calling the function :func:`sbc.report.report` using the 
        :obj:`sbc.report.WishList` object.
    ev_of_nn_two_site_spin_ops : ``[]`` | `array_like` (`str`, shape=(``num_two_site_ops``, 2)), optional
        ``ev_of_nn_two_site_spin_ops`` is an array of ``num_two_site_ops``
        string pairs, where each pair specifies a nearest-neighbour (NN) 
        two-site spin operator, i.e. the first and second strings in a given 
        pair specify the left and right single-site spin operators of the NN 
        two-site spin operator represented by the given pair. Only 
        concatenations of ``'sx'``, ``'sy'``, ``'sz'``, and ``'id'``, separated 
        by periods ``'.'``, are accepted representations of of single-site spin 
        operators. E.g. ``'sx.sx.sz'`` represents the single-site spin operator 
        :math:`\hat{\sigma}_{x}^2\hat{\sigma}_{z}` and ``'sz'``
        represents :math:`\hat{\sigma}_{z}`. If ``ev_of_nn_two_site_spin_ops``
        is not an empty array, then for 
        ``0<=i<len(ev_of_nn_two_site_spin_ops)``, the expectation value of the
        NN two-site spin operator specified in ``ev_of_nn_two_site_spin_ops[i]``
        is to be evaluated at each bond in the :math:`u=0` unit cell for both
        finite and infinite chains, but also the bond between the :math:`u=0`
        and :math:`u=1` unit cells for infinite chains. The results are to be 
        reported to the output file 
        ``'|'.join(ev_of_nn_two_site_spin_ops[i]).replace('.', '')+'.csv'`` upon
        calling the function :func:`sbc.report.report` using the
        :obj:`sbc.report.WishList` object.
    ev_of_energy : `bool`, optional
        If set to ``True``, then the expectation value of the system's 
        :math:`u=0` unit cell energy is to be reported to the output file 
        ``'energy.csv'`` upon calling the function :func:`sbc.report.report` 
        using the :obj:`sbc.report.WishList` object. See the documentation for 
        the function :func:`sbc.ev.energy` for further details on the
        aforementioned energy.
    correlation_length : `bool`, optional
        Note that ``sbc`` can only calculate the correlation length for
        **infinite** chains. If the system is infinite and 
        ``correlation_length`` is set to ``True``, then the correlation length 
        is to be reported to the output file ``'correlation-length.csv'`` upon 
        calling the function :func:`sbc.report.report` using the 
        :obj:`sbc.report.WishList` object.

    Attributes
    ----------
    state_trace : `bool`, read-only
        Same description as that in the parameters section above.
    schmidt_spectrum_sum : `bool`, read-only
        Same description as that in the parameters section above.
    realignment_criterion : `bool`, read-only
        Same description as that in the parameters section above.
    spin_config_probs : ``[]`` | `array_like` (``-1`` | ``1``, shape=(``num_configs``, ``M*L``)), read-only
        Same description as that in the parameters section above.
    ev_of_single_site_spin_ops : ``[]`` | `array_like` (`str`, ndim=1), read-only
        Same description as that in the parameters section above.
    ev_of_multi_site_spin_ops : ``[]`` | `array_like` (`str`, shape=(``num_ops``, ``M*L``)), read-only
        Same description as that in the parameters section above.
    ev_of_nn_two_site_spin_ops : ``[]`` | `array_like` (`str`, shape=(``num_two_site_ops``, 2)), read-only
        Same description as that in the parameters section above.
    ev_of_energy : `bool`, read-only
        Same description as that in the parameters section above.
    correlation_length : `bool`, read-only
        Same description as that in the parameters section above.
    """
    def __init__(self,
                 state_trace=False,
                 schmidt_spectrum_sum=False,
                 realignment_criterion=False,
                 spin_config_probs=[],
                 ev_of_single_site_spin_ops=[],
                 ev_of_multi_site_spin_ops=[],
                 ev_of_nn_two_site_spin_ops=[],
                 ev_of_energy=False,
                 correlation_length=False):
        self.state_trace = state_trace
        self.schmidt_spectrum_sum = schmidt_spectrum_sum
        self.realignment_criterion = realignment_criterion
        self.spin_config_probs = spin_config_probs
        self.ev_of_single_site_spin_ops = ev_of_single_site_spin_ops
        self.ev_of_multi_site_spin_ops = ev_of_multi_site_spin_ops
        self.ev_of_nn_two_site_spin_ops = ev_of_nn_two_site_spin_ops
        self.ev_of_energy = ev_of_energy
        self.correlation_length = correlation_length

        self._check_spin_config_probs()
        self._check_ev_of_single_site_spin_ops()
        self._check_ev_of_multi_site_spin_ops()
        self._check_ev_of_nn_two_site_spin_ops()

        return None



    def _check_spin_config_probs(self):
        try:
            if not self.spin_config_probs:
                return None

            for spin_config in self.spin_config_probs:
                spin_config = np.array(spin_config)
                if not np.all(np.logical_or(spin_config == 1,
                                            spin_config == -1)):
                    msg = _wish_list_check_spin_config_probs_err_msg_1
                    raise ValueError(msg)

        except TypeError:
            msg = _wish_list_check_spin_config_probs_err_msg_2
            raise TypeError(msg)

        return None



    def _check_ev_of_single_site_spin_ops(self):
        if not self.ev_of_single_site_spin_ops:
            return None

        parameter_str = 'sbc.report.ev_of_single_site_spin_ops'
        for op_string in self.ev_of_single_site_spin_ops:
            self._check_op_string(op_string, parameter_str)
        return None



    def _check_ev_of_multi_site_spin_ops(self):
        try:
            if not self.ev_of_multi_site_spin_ops:
                return None

            parameter_str = 'sbc.report.ev_of_multi_site_spin_ops'
            for op_strings in self.ev_of_multi_site_spin_ops:
                for op_string in op_strings:
                    self._check_op_string(op_string, parameter_str)

        except TypeError:
            msg = _wish_list_check_ev_of_multi_site_spin_ops_err_msg_1
            raise TypeError(msg)

        return None



    def _check_ev_of_nn_two_site_spin_ops(self):
        if not self.ev_of_nn_two_site_spin_ops:
            return None

        parameter_str = 'sbc.report.ev_of_nn_two_site_spin_ops'
        for op_string_pair in self.ev_of_nn_two_site_spin_ops:
            if len(op_string_pair) != 2:
                msg = _wish_list_check_ev_of_nn_two_site_spin_ops_err_msg_1
                raise ValueError(msg)
            self._check_op_string(op_string_pair[0], parameter_str)
            self._check_op_string(op_string_pair[1], parameter_str)
        return None



    def _check_op_string(self, op_string, parameter_str):
        ops = op_string.split(".")
        for op in ops:
            if op not in ('sx', 'sy', 'sz', 'id'):
                msg = _wish_list_check_op_string_err_msg_1.format(parameter_str)
                raise ValueError(msg)

        return None


    
class ReportParams():
    r"""The parameters required for the function :func:`sbc.report.report`.

    An instance of this class can be used with the function
    :func:`sbc.report.report` to report various properties of a given spin
    system. All output files are created/updated in an output directory, whose
    path is stored in the attribute 
    :attr:`sbc.report.ReportParams.output_dir` of the 
    :class:`sbc.report.ReportParams` class.

    Parameters
    ----------
    wish_list : :class:`sbc.report.WishList`
        A list of properties of a spin system to report to file. See the 
        documentation for class :class:`sbc.report.WishList` for details on
        which properties can be reported, and to which output files are they
        written.
    output_dir : `None` | `str`, optional
        If set to a string, then ``output_dir`` is the absolute or relative path
        to the output directory, in which all output files are created/updated.
        Otherwise, if set to `None`, i.e. the default value, then the current
        working directory is chosen as the output directory.

    Attributes
    ----------
    wish_list : :class:`sbc.report.WishList`, read-only
        A list of properties of a spin system to report to file.
    output_dir : `str`, read-only
        The absolute or relative path to the output directory, in which all 
        output files are created/updated.
    """
    def __init__(self, wish_list, output_dir=None):
        self.wish_list = wish_list

        if output_dir == None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir

        return None



def report(system_state, report_params):
    r"""Report user-specified properties of system at current moment in time.

    This function reports the properties specified in ``report_params`` of a
    given system, represented by the :obj:`sbc.state.SystemState` object
    ``system_state``, to output files in an output directory also specified in
    ``report_params``. See the documentation for the classes
    :class:`sbc.report.ReportParams` and :class:`sbc.report.WishList` for
    more details.

    Parameters
    ----------
    system_state : :class:`sbc.state.SystemState`
        The system state.
    report_params : :class:`sbc.report.ReportParams`
        Contains the list of properties to report and the path to the output
        directory. See the documentation for the class 
        :class:`sbc.report.ReportParams` for more details.

    Returns
    -------
    """
    _check_system_state_and_report_params(system_state, report_params)

    output_dir = report_params.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    L = system_state.system_model.L
    is_infinite = system_state.system_model.is_infinite
    num_bonds = L - 1 + int(is_infinite)
    
    t = system_state.t
    if t == 0.0:
        _overwrite_data_from_report_files(system_state, report_params)

    wish_list = report_params.wish_list

    if wish_list.state_trace:
        state_trace = sbc.state.trace(system_state)
        line = np.array([[t, state_trace]])
        filename = 'state-trace.csv'
        with open(output_dir + '/' + filename, 'a', 1) as file_obj:
            np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    if wish_list.schmidt_spectrum_sum or wish_list.realignment_criterion:
        S_sum = sbc.state.schmidt_spectrum_sum(system_state)
        if wish_list.schmidt_spectrum_sum:
            line = np.array([[t] + S_sum])
            filename = 'schmidt-spectrum-sum.csv'
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

        if wish_list.realignment_criterion:
            S_sum = np.array(S_sum)
            entangled = np.any(S_sum > 1)
            line = np.array([(t, entangled)], dtype='f8, ?')
            filename = 'realignment-criterion.csv'
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    if wish_list.spin_config_probs:
        for i, spin_config in enumerate(wish_list.spin_config_probs):
            prob = sbc.state.spin_config_prob(spin_config, system_state)
            line = np.array([[t, prob]])
            filename = 'spin-config-'+str(i+1)+'.csv'
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    sx_precalculated = False
    sz_precalculated = False
    sz_sz_precalculated = False
    
    if wish_list.ev_of_energy:
        x_fields = system_state.system_model.x_fields
        z_fields = system_state.system_model.z_fields
        zz_couplers = system_state.system_model.zz_couplers

        hx = [x_field.eval(t) for x_field in x_fields]
        hz = [z_field.eval(t) for z_field in z_fields]
        Jzz = [zz_coupler.eval(t) for zz_coupler in zz_couplers]

        sx = np.array(sbc.ev.single_site_spin_op('sx', system_state)).real
        sx_precalculated = True

        sz = np.array(sbc.ev.single_site_spin_op('sz', system_state)).real
        sz_precalculated = True
        
        sz_sz = \
            np.array(sbc.ev.nn_two_site_spin_op('sz', 'sz', system_state)).real
        sz_sz = np.real(sz_sz)
        sz_sz_precalculated = True
        
        ev_of_energy = \
            (np.dot(hx, sx) + np.dot(hz, sz) + np.dot(Jzz, sz_sz)).real

        values = (t, ev_of_energy)
        fmt = 'f8, ' + str(ev_of_energy.dtype)
        line = np.array([values], dtype=fmt)
        filename = 'energy.csv'
        with open(output_dir + '/' + filename, 'a', 1) as file_obj:
            np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")
        
    if wish_list.ev_of_single_site_spin_ops:
        for op_string in wish_list.ev_of_single_site_spin_ops:
            if (op_string == 'sx') and sx_precalculated:
                ev_of_op = sx
                        
            elif (op_string == 'sz') and sz_precalculated:
                ev_of_op = sz

            else:
                ev_of_op = np.array(sbc.ev.single_site_spin_op(op_string,
                                                               system_state))
                

            if _is_hermitian([op_string]):
                values = tuple([t] + [value.real for value in ev_of_op])
                fmt = 'f8, ' + 'f8, ' * (L-1) + 'f8'
            else:
                values = tuple([t] + [value for value in ev_of_op])
                fmt = 'f8, ' + 'c16, ' * (L-1) + 'c16'
                
            line = np.array([values], dtype=fmt)
            filename = op_string.replace('.', '') +'.csv'
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")
            
    if wish_list.ev_of_nn_two_site_spin_ops:
        for op_string_pair in wish_list.ev_of_nn_two_site_spin_ops:
            if ((op_string_pair[0] == 'sz') and (op_string_pair[1] == 'sz')
                and sz_sz_precalculated):
                ev_of_op = sz_sz
                    
            else:
                ev_of_op = \
                    np.array(sbc.ev.nn_two_site_spin_op(op_string_pair[0],
                                                        op_string_pair[1],
                                                        system_state))

            if _is_hermitian(op_string_pair):
                values = tuple([t] + [value.real for value in ev_of_op])
                fmt = 'f8, ' + 'f8, ' * (num_bonds-1) + 'f8'
            else:
                values = tuple([t] + [value for value in ev_of_op])
                fmt = 'f8, ' + 'c16, ' * (num_bonds) + 'c16'
            
            line = np.array([values], dtype=fmt)        
            filename = (op_string_pair[0].replace('.', '') + '|'
                        + op_string_pair[1].replace('.', '') +'.csv')
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    if wish_list.ev_of_multi_site_spin_ops:
        for i, op_strings in enumerate(wish_list.ev_of_multi_site_spin_ops):
            ev_of_op = sbc.ev.multi_site_spin_op(op_strings, system_state)

            if _is_hermitian(op_strings):
                fmt = 'f8, f8'
                line = np.array([(t, np.real(ev_of_op))], dtype=fmt)
            else:
                fmt = 'f8, c16'
                line = np.array([(t, ev_of_op)], dtype=fmt)
                    
            filename = 'multi-site-spin-op-'+str(i+1)+'.csv'
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    if wish_list.correlation_length:
        line = np.array([[t, system_state.correlation_length]])
        filename = 'correlation-length.csv'
        with open(output_dir + '/' + filename, 'a', 1) as file_obj:
            np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")
                        
    return None



def _check_system_state_and_report_params(system_state, report_params):
    L = system_state.system_model.L
    wish_list = report_params.wish_list

    if system_state.system_model.is_infinite:
        for spin_config in wish_list.spin_config_probs:
            if len(spin_config) % L != 0:
                msg = _check_system_state_and_report_params_err_msg_1a
                raise ValueError(msg)
        for str_seq_of_multi_spin_op in wish_list.ev_of_multi_site_spin_ops:
            if len(str_seq_of_multi_spin_op) % L != 0:
                msg = _check_system_state_and_report_params_err_msg_2a
                raise ValueError(msg)
    else:
        for spin_config in wish_list.spin_config_probs:
            if len(spin_config) != L:
                msg = _check_system_state_and_report_params_err_msg_1b
                raise ValueError(msg)
        for str_seq_of_multi_spin_op in wish_list.ev_of_multi_site_spin_ops:
            if len(str_seq_of_multi_spin_op) != L:
                msg = _check_system_state_and_report_params_err_msg_2b
                raise ValueError(msg)
        if wish_list.correlation_length:
            msg = _check_system_state_and_report_params_err_msg_3
            raise ValueError(msg)

    return None



def _overwrite_data_from_report_files(system_state, report_params):
    output_dir = report_params.output_dir
    wish_list = report_params.wish_list

    L = system_state.system_model.L
    is_infinite = system_state.system_model.is_infinite
    num_bonds = L - 1 + int(is_infinite)

    bond_header_1 = \
        np.array([['t'] + ['at bond #'+str(i) for i in range(num_bonds)]])
    bond_header_2 = \
        np.array([['t'] + ['EV at bond #'+str(i) for i in range(num_bonds)]])
    site_header = np.array([['t'] + ['EV at site #'+str(r) for r in range(L)]])

    
    if wish_list.state_trace:
        header = np.array([['t', 'state trace']])
        filename = 'state-trace.csv'
        with open(output_dir + '/' + filename, 'w', 1) as file_obj:
            np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")

    if wish_list.schmidt_spectrum_sum:
        filename = 'schmidt-spectrum-sum.csv'
        with open(output_dir + '/' + filename, 'w', 1) as file_obj:
            np.savetxt(file_obj, bond_header_1, fmt="%-20s", delimiter=";")

    if wish_list.realignment_criterion:
        header = np.array([['t', 'entangled']])
        filename = 'realignment-criterion.csv'
        with open(output_dir + '/' + filename, 'w', 1) as file_obj:
            np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")

    if wish_list.spin_config_probs:
        configs = wish_list.spin_config_probs
        num_configs = len(configs)
        largest_config_size = max([len(configs[i]) for i in range(num_configs)])
        
        header = \
            np.array([['spin config #'+str(i+1) for i in range(num_configs)]])
        
        data = []
        for config in configs:
            config = np.array(config, dtype='str')
            config = np.pad(config, (0, largest_config_size-config.size),
                            "constant", constant_values=" ")
            data.append(config.tolist())
        data = np.array(data).transpose()
        
        header_and_data = np.concatenate((header, data))
        filename = 'spin-config-list.csv'
        with open(output_dir + '/' + filename, 'w', 1) as file_obj:
            np.savetxt(file_obj, header_and_data, fmt="%-20s", delimiter=";")

        for i in range(num_configs):
            header = np.array([['t', 'probability']])
            filename = 'spin-config-'+str(i+1)+'.csv'
            with open(output_dir + '/' + filename, 'w', 1) as file_obj:
                np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")

    if wish_list.ev_of_single_site_spin_ops:
        for op_string in wish_list.ev_of_single_site_spin_ops:
            filename = op_string.replace('.', '') +'.csv'
            with open(output_dir + '/' + filename, 'w', 1) as file_obj:
                np.savetxt(file_obj, site_header, fmt="%-20s", delimiter=";")

    if wish_list.ev_of_nn_two_site_spin_ops:
        for op_string_pair in wish_list.ev_of_nn_two_site_spin_ops:
            filename = (op_string_pair[0].replace('.', '') + '|'
                        + op_string_pair[1].replace('.', '') +'.csv')
            with open(output_dir + '/' + filename, 'w', 1) as file_obj:
                np.savetxt(file_obj, bond_header_2, fmt="%-20s", delimiter=";")

    if wish_list.ev_of_multi_site_spin_ops:
        str_seqs_of_multi_spin_ops = wish_list.ev_of_multi_site_spin_ops
        num_ops = len(str_seqs_of_multi_spin_ops)
        longest_str_seq_size = max([len(str_seqs_of_multi_spin_ops[i])
                                    for i in range(num_ops)])

        header = \
            np.array([['multi-site op #'+str(i+1) for i in range(num_ops)]])

        data = []
        for str_seq in str_seqs_of_multi_spin_ops:
            str_seq = np.array(str_seq, dtype='str')
            str_seq = np.pad(str_seq, (0, str_seq.size-longest_str_seq_size),
                            "constant", constant_values="id")
            data.append(str_seq.tolist())
        data = np.array(data).transpose()
        
        header_and_data = np.concatenate((header, data))
        filename = 'multi-site-spin-op-list.csv'
        with open(output_dir + '/' + filename, 'w', 1) as file_obj:
            np.savetxt(file_obj, header_and_data, fmt="%-20s", delimiter=";")

        num_spin_ops = len(wish_list.ev_of_multi_site_spin_ops)
        for i in range(num_spin_ops):
            header = np.array([['t', 'EV']])
            filename = 'multi-site-spin-op-'+str(i+1)+'.csv'
            with open(output_dir + '/' + filename, 'w', 1) as file_obj:
                np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")

    if wish_list.ev_of_energy:
        header = np.array([['t', 'EV']])
        filename = 'energy.csv'
        with open(output_dir + '/' + filename, 'w', 1) as file_obj:
            np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")

    if wish_list.correlation_length:
        header = np.array([['t', 'correlation-length']])
        filename = 'correlation-length.csv'
        with open(output_dir + '/' + filename, 'w', 1) as file_obj:
            np.savetxt(file_obj, header, fmt="%-20s", delimiter=";")

    return None



def _is_hermitian(op_strings):
    bools = np.array([op_string.split('.') == op_string.split('.')[::-1]
                      for op_string in op_strings])

    return np.all(bools)



_wish_list_check_spin_config_probs_err_msg_1 = \
    ("A valid spin configuration consists of an array with each element equal "
     "to either 1 (signifying an Ising spin pointing 'up'), or -1 (signifying "
     "an Ising spin pointing 'down').")

_wish_list_check_spin_config_probs_err_msg_2 = \
    ("The parameter `spin_config_probs` of the `sbc.report.WishList` class is "
     "expected to be a sequence of sequences, where each contained sequence is "
     "a sequence of the integers 1 and -1, representing a classical Ising spin "
     "configuration.")

_wish_list_check_ev_of_multi_site_spin_ops_err_msg_1 = \
    ("The parameter `ev_of_multi_site_spin_ops` of the `sbc.report.WishList` "
     "class is expected to be a sequence of sequences, where each contained "
     "sequence is a sequence of operator strings representing a multi-site "
     "spin operator. Only concatenations of the strings 'sx', 'sy', 'sz', and "
     "'id', separated by periods '.', are accepted as valid operator strings.")

_wish_list_check_ev_of_nn_two_site_spin_ops_err_msg_1 = \
    ("The parameter `ev_of_nn_two_site_spin_ops` of the `sbc.report.WishList` "
     "class is expected to be a sequence of operator string pairs. Only "
     "concatenations of the strings 'sx', 'sy', 'sz', and 'id', separated by "
     "periods '.', are accepted as operator strings.")

_wish_list_check_op_string_err_msg_1 = \
    ("One of the given operator strings in the parameter `{}` is not of the "
     "correct form: only concatenations of the strings 'sx', 'sy', 'sz', and "
     "'id', separated by periods '.', are accepted.")

_check_system_state_and_report_params_err_msg_1a = \
    ("The parameter `spin_config_probs` of the `sbc.report.WishList` class is "
     "expected to be a sequence of classical Ising spin configurations, where "
     "the number of spins in each configuration is a positive multiple of the "
     "system's unit cell size.")
_check_system_state_and_report_params_err_msg_1b = \
    ("The parameter `spin_config_probs` of the `sbc.report.WishList` class is "
     "expected to be a sequence of classical Ising spin configurations, where "
     "the number of spins in each configuration is equal to the number of "
     "spins in the system.")

_check_system_state_and_report_params_err_msg_2a = \
    ("The parameter `ev_of_multi_site_spin_ops` of the `sbc.report.WishList` "
     "class is expected to be a sequence of operator string sequences, where "
     "the number of operator strings in each operator string sequence is equal "
     "to a positive multiple of the system's unit cell size.")
_check_system_state_and_report_params_err_msg_2b = \
    ("The parameter `ev_of_multi_site_spin_ops` of the `sbc.report.WishList` "
     "class is expected to be a sequence of operator string sequences, where "
     "the number of operator strings in each operator string sequence is equal "
     "to the number of spins in the system.")

_check_system_state_and_report_params_err_msg_3 = \
    ("Cannot calculate the correlation length for finite chains.")
