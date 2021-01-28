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
__status__ = "Non-Production"



##################################
## Define classes and functions ##
##################################

# List of public objects in objects.
__all__ = ["Scalar"]



def _time_independent_fn(t, fn_result):
    return fn_result



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
