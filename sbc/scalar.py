#!/usr/bin/env python
r"""Contains class definition for time-dependent scalar model parameters.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For deep copies of objects.
import copy

# For checking whether an object is a numerical scalar.
import numbers



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Development"



##################################
## Define classes and functions ##
##################################

# List of public objects in objects.
__all__ = ["Scalar"]



class Scalar():
    r"""A time-dependent scalar model parameter.

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
                                "`sbc.scalar.Scalar`, is not compatible "
                                "with `func_form`.")
            
        self.func_form = func_form

        return None


    
    def eval(self, t):
        r"""Evaluate scalar model parameter at time ``t``.

        Parameters
        ----------
        t : `float`
            Time.

        Returns
        -------
        result : `float`
            The value of the scalar model parameter at time ``t``.
        """
        result = self._func_form(t, **self._func_kwargs)

        return result



    def __eq__(self, obj):
        # Defining custom equality method.
        if not isinstance(obj, Scalar):
            result = False
        else:
            co_code_1 = self._func_form.__code__.co_code  # Bytecode.
            co_code_2 = obj._func_form.__code__.co_code
            func_kwargs_1 = self._func_kwargs
            func_kwargs_2 = obj._func_kwargs
            if (co_code_1 == co_code_2) and (func_kwargs_1 == func_kwargs_2):
                result = True
            else:
                result = False

        return result



    def __hash__(self):
        # Custom __eq__ makes class unhashable by default. The following is
        # necessary in order for the class to behave properly with sets and
        # dictionaries.
        func_form_co_code = self._func_form.__code__.co_code  # Bytecode.
        func_kwargs_in_tuple_form = tuple(sorted(self._func_kwargs.items()))
        result = hash((func_form_co_code, func_kwargs_in_tuple_form))

        return result

        

def _time_independent_fn(t, fn_result):
    return fn_result
