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
from sbc.state import trace, schmidt_spectrum_sum
from sbc.state import realignment_criterion, spin_config_prob
from sbc.ev import single_site_spin_op, multi_site_spin_op
from sbc.ev import nn_two_site_spin_op, energy



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
        evaluate to unity.
    schmidt_spectrum_sum : `bool`, optional
        If set to ``True``, then the Schmidt spectrum sum for all bonds is to be
        reported to the output file ``'schmidt-spectrum-sum.csv'`` upon calling 
        the function :func:`sbc.report.report` using the 
        :obj:`sbc.report.WishList` object. See the documentation for the 
        function :func:`sbc.state.schmidt_spectrum_sum` for a discussion 
        regarding Schmidt spectra and Schmidt spectrum sums.
    realignment_criterion : `bool`, optional
        If set to ``True``, then the realignment criterion is to be applied to
        the system's current state to determine whether it is entangled, and the
        result of this operation is to be reported to the output file 
        ``'realignment-criterion.csv'`` upon calling the 
        function :func:`sbc.report.report` using the 
        :obj:`sbc.report.WishList` object. See the documentation for the
        function :func:`sbc.state.realignment_criterion` for a brief 
        discussion regarding the realignment criterion.
    spin_config_probs : ``[]`` | `array_like` (``-1`` | ``1``, shape=(``num_configs``, ``L``)), optional
        If ``spin_config_probs`` is of the form 
        `array_like` (``-1`` | ``1``, shape=(``num_configs``, ``L``)),
        where ``num_configs`` is a positive integer, and ``L`` is the number of 
        sites in the system of interest, then 
        ``spin_config_probs[0<=i<num_configs]`` specifies a classical spin
        configuration. Moreover, for ``0<=i<num_configs``, the spin 
        configuration specified in ``spin_config_probs[i]`` is to be written to
        the ``i`` th column of the output file ``'spin-config-list.csv'`` and
        the corresponding spin configuration probability to the output file
        ``'spin-config-'+str(i+1)+'.csv'`` upon calling the function 
        :func:`sbc.report.report` using the :obj:`sbc.report.WishList` 
        object. Note that a spin configuration is represented by an array of the
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
        is to be evaluated at each site in the system and the results of which
        are to be reported to the output file 
        ``ev_of_single_site_spin_ops[i].replace('.', '')+'.csv'`` upon calling
        the function :func:`sbc.report.report` using the
        :obj:`sbc.report.WishList` object.
    ev_of_multi_site_spin_ops : ``[]`` | `array_like` (`str`, shape=(``num_ops``, ``L``)), optional
        If ``ev_of_multi_site_spin_ops`` is of the form 
        `array_like` (`str`, shape=(``num_ops``, ``L``)),
        where ``num_ops`` is a positive integer, and ``L`` is the number of 
        sites in the system of interest, then 
        ``ev_of_multi_site_spin_ops[0<=i<num_ops]`` specifies a multi-site spin
        operator, with 
        ``ev_of_multi_site_spin_ops[0<=i<num_ops][0<=r<L]`` specifying the
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
        is to be evaluated at each bond in the system and the results of which
        are to be reported to the output file 
        ``'|'.join(ev_of_nn_two_site_spin_ops[i]).replace('.', '')+'.csv'`` upon
        calling the function :func:`sbc.report.report` using the
        :obj:`sbc.report.WishList` object.
    ev_of_energy : `bool`, optional
        If set to ``True``, then the expectation value of the system's energy is
        to be reported to the output file ``'energy.csv'`` upon calling the 
        function :func:`sbc.report.report` using the :obj:`sbc.report.WishList` 
        object.

    Attributes
    ----------
    state_trace : `bool`, read-only
        Same description as that in the parameters section above.
    schmidt_spectrum_sum : `bool`, read-only
        Same description as that in the parameters section above.
    realignment_criterion : `bool`, read-only
        Same description as that in the parameters section above.
    spin_config_probs : ``[]`` | `array_like` (``-1`` | ``1``, shape=(``num_configs``, ``L``)), read-only
        Same description as that in the parameters section above.
    ev_of_single_site_spin_ops : ``[]`` | `array_like` (`str`, ndim=1), read-only
        Same description as that in the parameters section above.
    ev_of_multi_site_spin_ops : ``[]`` | `array_like` (`str`, shape=(``num_ops``, ``L``)), read-only
        Same description as that in the parameters section above.
    ev_of_nn_two_site_spin_ops : ``[]`` | `array_like` (`str`, shape=(``num_two_site_ops``, 2)), read-only
        Same description as that in the parameters section above.
    ev_of_energy : `bool`, read-only
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
                 ev_of_energy=False):
        self.state_trace = state_trace
        self.schmidt_spectrum_sum = schmidt_spectrum_sum
        self.realignment_criterion = realignment_criterion
        self.spin_config_probs = spin_config_probs
        self.ev_of_single_site_spin_ops = ev_of_single_site_spin_ops
        self.ev_of_multi_site_spin_ops = ev_of_multi_site_spin_ops
        self.ev_of_nn_two_site_spin_ops = ev_of_nn_two_site_spin_ops
        self.ev_of_energy = ev_of_energy

        self._check_spin_config_probs()
        self._check_ev_of_single_site_spin_ops()
        self._check_ev_of_multi_site_spin_ops()
        self._check_ev_of_nn_two_site_spin_ops()

        return None



    def _check_spin_config_probs(self):
        try:
            if not self.spin_config_probs:
                return None
            L = len(self.spin_config_probs[0])

            for spin_config in self.spin_config_probs:
                spin_config = np.array(spin_config)
                if not np.all(np.logical_or(spin_config == 1,
                                            spin_config == -1)):
                    raise ValueError("A valid spin configuration consists of "
                                     "an array with each element equal to "
                                     "either 1 (signifying an Ising spin "
                                     "pointing 'up'), or -1 (signifying an "
                                     "Ising spin pointing 'down').")

                if len(spin_config) != L:
                    raise ValueError("All spin configurations need to have the "
                                     "same number of spins.")

        except TypeError:
            raise TypeError("The parameter `spin_config_probs` of the "
                            "`sbc.report.WishList` class is expected to be "
                            "a sequence of sequences, where each contained "
                            "sequence is a sequence of the integers 1 and -1, "
                            "representing a classical Ising spin "
                            "configuration. Each one these spin configurations "
                            "is expected to have the same number of spins.")

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
            L = len(self.ev_of_multi_site_spin_ops[0])

            parameter_str = 'sbc.report.ev_of_multi_site_spin_ops'
            for op_strings in self.ev_of_multi_site_spin_ops:
                for op_string in op_strings:
                    self._check_op_string(op_string, parameter_str)

                if len(op_strings) != L:
                    raise ValueError("Each sequence of operator strings "
                                     "representing a multi-site spin operator "
                                     "is expected to have the same number of "
                                     "operator strings, where the ith operator "
                                     "string in a given sequence represents "
                                     "a single-site spin operator at site i.")

        except TypeError:
            raise TypeError("The parameter `ev_of_multi_site_spin_ops` of the "
                            "`sbc.report.WishList` class is expected to be "
                            "a sequence of sequences, where each contained "
                            "sequence is a sequence of operator strings "
                            "representing a multi-site spin operator. Each of "
                            "of these sequences of strings is expected to be "
                            "of the same length. Only concatenations of the "
                            "strings 'sx', 'sy', 'sz', and 'id', separated by "
                            "periods '.', are accepted as valid operator "
                            "strings.")

        return None



    def _check_ev_of_nn_two_site_spin_ops(self):
        if not self.ev_of_nn_two_site_spin_ops:
            return None

        parameter_str = 'sbc.report.ev_of_nn_two_site_spin_ops'
        for op_string_pair in self.ev_of_nn_two_site_spin_ops:
            if len(op_string_pair) != 2:
                raise ValueError("The parameter `ev_of_nn_two_site_spin_ops` "
                                 "of the `sbc.report.WishList` class is "
                                 "expected to be a sequence of operator string "
                                 "pairs. Only concatenations of the strings "
                                 "'sx', 'sy', 'sz', and 'id', separated by "
                                 "periods '.', are accepted as operator "
                                 "strings.")
            self._check_op_string(op_string_pair[0], parameter_str)
            self._check_op_string(op_string_pair[1], parameter_str)
        return None



    def _check_op_string(self, op_string, parameter_str):
        ops = op_string.split(".")
        for op in ops:
            if op not in ('sx', 'sy', 'sz', 'id'):
                msg = ("One of the given operator strings in the parameter "
                       "`{}` is not of the correct form: only concatenations "
                       "of the strings 'sx', 'sy', 'sz', and 'id', separated "
                       "by periods '.', are accepted.")
                raise ValueError(msg.format(parameter_str))

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
    output_dir = report_params.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    L = system_state.system_model.L
    
    t = system_state.t
    if t == 0.0:
        _overwrite_data_from_report_files(system_state, report_params)

    wish_list = report_params.wish_list

    if (wish_list.state_trace or wish_list.spin_config_probs
        or wish_list.ev_of_single_site_spin_ops
        or wish_list.ev_of_multi_site_spin_ops
        or wish_list.ev_of_nn_two_site_spin_ops
        or wish_list.ev_of_energy):
        state_trace = trace(system_state)
        if wish_list.state_trace:
            line = np.array([[t, state_trace]])
            filename = 'state-trace.csv'
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    if wish_list.schmidt_spectrum_sum or wish_list.realignment_criterion:
        S_sum = schmidt_spectrum_sum(system_state)
        if wish_list.schmidt_spectrum_sum:
            line = np.array([[t] + S_sum])
            filename = 'schmidt-spectrum-sum.csv'
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

        if wish_list.realignment_criterion:
            S_sum = np.array(S_sum)
            entangled = np.any(S_sum > 1)
            line = np.array([[t, entangled]])
            filename = 'realignment-criterion.csv'
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    if wish_list.spin_config_probs:
        for i, spin_config in enumerate(wish_list.spin_config_probs):
            prob = spin_config_prob(spin_config, system_state) / state_trace
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

        sx = np.array(single_site_spin_op('sx', system_state)).real
        sx /= state_trace
        sx_precalculated = True

        sz = np.array(single_site_spin_op('sz', system_state)).real
        sz /= state_trace
        sz_precalculated = True
        
        sz_sz = np.array(nn_two_site_spin_op('sz', 'sz', system_state)).real
        sz_sz /= state_trace
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
                ev_of_op = np.array(single_site_spin_op(op_string,
                                                        system_state))
                ev_of_op /= state_trace
                

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
                ev_of_op = np.array(nn_two_site_spin_op(op_string_pair[0],
                                                        op_string_pair[1],
                                                        system_state))
                ev_of_op /= state_trace

            if _is_hermitian(op_string_pair):
                values = tuple([t] + [value.real for value in sz_sz])
                fmt = 'f8, ' + 'f8, ' * (L-2) + 'f8'
            else:
                values = tuple([t] + [value for value in sz_sz])
                fmt = 'f8, ' + 'c16, ' * (L-2) + 'c16'
            
            line = np.array([values], dtype=fmt)        
            filename = (op_string_pair[0].replace('.', '') + '|'
                        + op_string_pair[1].replace('.', '') +'.csv')
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")

    if wish_list.ev_of_multi_site_spin_ops:
        for i, op_strings in enumerate(wish_list.ev_of_multi_site_spin_ops):
            ev_of_op = multi_site_spin_op(op_strings, system_state)
            ev_of_op /= state_trace

            if _is_hermitian(op_strings):
                fmt = 'f8, f8'
                line = np.array([(t, np.real(ev_of_op))], dtype=fmt)
            else:
                fmt = 'f8, c16'
                line = np.array([(t, ev_of_op)], dtype=fmt)
                    
            filename = 'multi-site-spin-op-'+str(i+1)+'.csv'
            with open(output_dir + '/' + filename, 'a', 1) as file_obj:
                np.savetxt(file_obj, line, fmt="%-20s", delimiter=";")
                        
    return None
                    


def _overwrite_data_from_report_files(system_state, report_params):
    output_dir = report_params.output_dir
    wish_list = report_params.wish_list

    L = system_state.system_model.L

    bond_header_1 = np.array([['t'] + ['at bond #'+str(i) for i in range(L-1)]])
    bond_header_2 = np.array([['t']
                              + ['EV at bond #'+str(i) for i in range(L-1)]])
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
        num_configs = len(wish_list.spin_config_probs)
        header = \
            np.array([['spin config #'+str(i+1) for i in range(num_configs)]])
        data = np.array(wish_list.spin_config_probs).transpose()
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
        num_ops = len(wish_list.ev_of_multi_site_spin_ops)
        header = \
            np.array([['multi-site op #'+str(i+1) for i in range(num_ops)]])
        data = np.array(wish_list.ev_of_multi_site_spin_ops).transpose()
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

    return None



def _is_hermitian(op_strings):
    bools = np.array([op_string.split('.') == op_string.split('.')[::-1]
                      for op_string in op_strings])

    return np.all(bools)
