#!/usr/bin/env python
r"""Contains class definitions for the model components of the bath.

``sbc`` is a library for simulating the dynamics of a generalized
one-dimensional spin-boson model, where both the :math:`z`- and 
:math:`y`-components of the spins are coupled to bosonic baths, rather than 
only the :math:`z`-components. The Hamiltonian of this model can be broken down
into the following components:

.. math ::
    \hat{H}(t) = \hat{H}^{(A)}(t) + \hat{H}^{(B)} + \hat{H}^{(AB)},
    :label: bath_total_Hamiltonian

where :math:`\hat{H}^{(A)}(t)` is the system Hamiltonian, which encodes all
information regarding energies associated exclusively with the spins; 
:math:`\hat{H}^{(B)}` is the bath Hamiltonian, which encodes all information
regarding energies associated with the components of the bosonic environment; 
and :math:`\hat{H}^{(AB)}` is the system-bath coupling Hamiltonian, which 
describes all energies associated with the coupling between the system and the 
environment.

The bath Hamiltonian :math:`\hat{H}^{(B)}` describes a collection of decoupled
harmonic oscillators:

.. math ::
    \hat{H}^{(B)} = \sum_{r=1}^{L}\sum_{\epsilon} 
    \left\{\omega_{y; \epsilon} \hat{b}_{y; r; \epsilon}^{\dagger} 
    \hat{b}_{y; r; \epsilon}^{\vphantom{\dagger}}
    + \omega_{z; \epsilon} \hat{b}_{z; r; \epsilon}^{\dagger} 
    \hat{b}_{z; r; \epsilon}^{\vphantom{\dagger}}\right\},
    :label: bath_bath_hamiltonian

with :math:`\epsilon` being the oscillator mode index, 
:math:`\hat{b}_{\nu; r; \epsilon}^{\dagger}` and
:math:`\hat{b}_{\nu; r; \epsilon}^{\vphantom{\dagger}}` being the bosonic
creation and annihilation operators respectively for the harmonic oscillator at
site :math:`r` in mode :math:`\epsilon` with angular frequency 
:math:`\omega_{\nu; \epsilon}`, coupled to the :math:`\nu^{\mathrm{th}}` 
component of the spin at the same site.

The system-bath coupling Hamiltonian :math:`\hat{H}^{(AB)}` is given by

.. math ::
    \hat{H}^{(AB)}=-\sum_{r=1}^{L}\left\{
    \hat{\sigma}_{y;r}\hat{\mathcal{Q}}_{y;r}
    + \hat{\sigma}_{z;r}\hat{\mathcal{Q}}_{z;r}\right\},
    :label: bath_system_bath_hamiltonian

where :math:`\hat{\mathcal{Q}}_{\nu;r}` is the generalized
reservoir force operator at site :math:`r` that acts on the 
:math:`\nu^{\mathrm{th}}` component of the spin at the same site:

.. math ::
    \hat{\mathcal{Q}}_{\nu;r}=-\sum_{\epsilon}\lambda_{\nu;\epsilon}
    \left\{ \hat{b}_{\nu;r;\epsilon}^{\dagger}
    +\hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}}\right\},
    :label: bath_generalized_reservoir_force

with :math:`\lambda_{\nu;\epsilon}` being the coupling strength between the
:math:`\nu^{\mathrm{th}}` component of a spin and the harmonic oscillator at the
same site in mode :math:`\epsilon`. 

Rather than specify the model parameters of bath and system-bath coupling, one
can alternatively specify the spectral densities of the :math:`y^{\mathrm{th}}`-
and :math:`z^{\mathrm{th}}`-components of the noise at temperature :math:`T`,
which contain the same information as the aforementioned model parameters.
This alternative approach is easier to incorporate into the QUAPI formalism,
upon which ``sbc`` is based on.

The spectral density of the :math:`\nu^{\mathrm{th}}`-component of the noise at
temperature :math:`T` is defined as:

.. math ::
    A_{\nu;T}(\omega)=\int_{-\infty}^{\infty}dt\,e^{i\omega t}C_{\nu;T}(t),
    :label: bath_spectral_density_cmpnt_1

where :math:`C_{\nu;T}(t)` is the :math:`\nu^{\mathrm{th}}`-component of the
bath correlation function at temperature :math:`T`:

.. math ::
    C_{\nu;T}(t)=\text{Tr}^{(B)}\left\{ \hat{\rho}^{(i,B)}
    \hat{\mathcal{Q}}_{\nu;r=1}^{(B)}(t)
    \hat{\mathcal{Q}}_{\nu;r=1}^{(B)}(0)\right\},
    :label: bath_bath_correlation_function

.. math ::
    \hat{\mathcal{Q}}_{\nu;r}^{(B)}(t)=e^{i\hat{H}^{(B)}t}
    \hat{\mathcal{Q}}_{\nu;r}e^{-i\hat{H}^{(B)}t},
    :label: bath_Q_heisenberg_picture

:math:`\mathrm{Tr}^{(B)}\left\{\cdots\right\}` is the partial trace with
respect to the bath degrees of freedom, :math:`\hat{\rho}^{(i,B)}` is the 
bath's reduced state operator at :math:`t=0`:

.. math ::
    \hat{\rho}^{(i,B)}\equiv\frac{e^{-\beta\hat{H}^{(B)}}}
    {\mathrm{Tr}^{(B)}\left\{e^{-\beta\hat{H}^{(B)}}\right\}},
    :label: bath_initial_bath_reduced_state_operator

with :math:`\beta=1/(k_B T)`, and :math:`k_B` being the Boltzmann constant. 

Because the system is coupled to its environment, the dynamics of the system not
only depend on its current state, but also its history. The degree to which the
dynamics depend on the history of the system can be characterized by a quantity
known as the bath correlation time, or “memory”. Informally, one may define the 
system's memory :math:`\tau` as the time beyond which both the :math:`y`- and 
the :math:`z`-components of the bath correlation function 
[Eq. :eq:`bath_bath_correlation_function`] are negligibly small:

.. math ::
    \left|C_{\nu; T}\right| \ll 1.
    :label: bath_defining_memory

In ``sbc``, the user specifies the system's memory: if chosen too small, the
simulation will not be accurate.

This module contains classes to specify all the necessary model components of 
the bath, namely the :math:`A_{\nu;T}(\omega)` and the memory :math:`\tau`.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For deep copies of objects.
import copy

# Import ceiling function.
from math import ceil



# For evaluating special math functions.
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
__all__ = ["SpectralDensityCmpnt",
           "SpectralDensityCmpnt0T",
           "SpectralDensitySubcmpnt",
           "SpectralDensitySubcmpnt0T",
           "Model"]



class SpectralDensityCmpnt():
    r"""The finite-temperature spectral density of some component of the noise.

    For background information on spectral densities, see the documentation for
    the module :mod:`sbc.bath`. 

    This class represents the spectral density of some component of the noise 
    [e.g. the :math:`y`-component] at a given temperature :math:`T`. This 
    quantity, which we denote by :math:`A_{\nu;T}(\omega)` with 
    :math:`\nu\in\{y, z\}`, was introduced in 
    Eq. :eq:`bath_spectral_density_cmpnt_1` in the documentation for the module 
    :mod:`sbc.bath`.

    In our detailed exposition on our QUAPI+TN approach found :manual:`here <>`,
    we show that :math:`A_{\nu;T}(\omega)` can be expressed as

    .. math ::
        A_{\nu;T}(\omega)=\text{sign}(\omega)
        \frac{A_{\nu;T=0}\left(\left|\omega\right|\right)}{1-e^{-\beta\omega}},
        :label: bath_spectral_density_cmpnt_expr

    where

    .. math ::
        \text{sign}\left(\omega\right)=\begin{cases}
        1 & \text{if }\omega>0,\\
        -1 & \text{if }\omega<0,
        \end{cases}
        :label: bath_spectral_density_cmpnt_sign_function

    :math:`A_{\nu;T=0}(\omega)` is the zero-temperature limit of
    :math:`A_{\nu;T}(\omega)`, and :math:`\beta=1/(k_B T)` with :math:`k_B` 
    being the Boltzmann constant. The quantity :math:`A_{\nu;T=0}(\omega)` is
    represented by the :class:`sbc.bath.SpectralDensityCmpnt0T` class. See
    the documentation for the aforementioned class for more details on
    :math:`A_{\nu;T=0}(\omega)`.

    In many applications of interest, :math:`A_{\nu;T=0}(\omega)` will comprise
    of multiple subcomponents:

    .. math ::
        A_{\nu;T=0}(\omega)=\sum_{\varsigma}A_{\nu;T=0;\varsigma}(\omega),
        :label: bath_spectral_density_cmpnt_0T_breakdown_1

    where :math:`A_{\nu;T=0;\varsigma}(\omega)` is the 
    :math:`\varsigma^{\mathrm{th}}` subcomponent of :math:`A_{\nu;T=0}(\omega)`.
    As an example, :math:`A_{z;T=0}(\omega)` may be strongly peaked at multiple 
    frequencies, in which case it naturally decomposes into multiple 
    subcomponents. If :math:`A_{\nu;T=0}(\omega)` naturally decomposes into 
    multiple subcomponents, it is best to treat each subcomponent separately in 
    our QUAPI+TN approach rather than treat :math:`A_{\nu;T=0}(\omega)` as a 
    single entity. For further discussion on this matter, see our detailed 
    exposition of our QUAPI+TN approach found :manual:`here <>`. 

    From Eqs. :eq:`bath_spectral_density_cmpnt_expr` and 
    :eq:`bath_spectral_density_cmpnt_0T_breakdown_1`, we can write

    .. math ::
        A_{\nu;T}(\omega)=\sum_{\varsigma}A_{\nu;T;\varsigma}(\omega),
        :label: bath_spectral_density_cmpnt_breakdown

    where

    .. math ::
        A_{\nu;T;\varsigma}(\omega)=\text{sign}(\omega)
        \frac{A_{\nu;T=0;\varsigma}\left(\left|\omega\right|\right)}
        {1-e^{-\beta\omega}}.
        :label: bath_spectral_density_subcmpnt_expr_1

    We refer to the :math:`A_{\nu;T;\varsigma}(\omega)` as the subcomponents of
    :math:`A_{\nu;T}(\omega)`. The quantity :math:`A_{\nu;T;\varsigma}(\omega)` 
    is represented by the :class:`sbc.bath.SpectralDensitySubcmpnt` class.

    Parameters
    ----------
    limit_0T : :class:`sbc.bath.SpectralDensityCmpnt0T`
        The zero-temperature limit of the spectral density of the component of
        the noise of interest, :math:`A_{\nu;T=0}(\omega)`.
    beta : `float`
        The inverse temperature, :math:`\beta=1/(k_B T)`, with :math:`k_B` 
        being the Boltzmann constant and :math:`T` being the temperature.

    Attributes
    ----------
    subcmpnts : `array_like` (:class:`sbc.bath.SpectralDensitySubcmpnt`, ndim=1), read-only
        The subcomponents of :math:`A_{\nu;T}(\omega)`, i.e. the
        :math:`A_{\nu;T;\varsigma}(\omega)`.
    """
    def __init__(self, limit_0T, beta):
        self.subcmpnts = []
        for subcmpnt_0T in limit_0T.subcmpnts:
            self.subcmpnts += [SpectralDensitySubcmpnt(subcmpnt_0T, beta)]

        return None



    def eval(self, omega):
        r"""Evaluate :math:`A_{\nu;T}(\omega)` at frequency ``omega``.

        Parameters
        ----------
        omega : `float`
            Frequency.

        Returns
        -------
        result : `float`
            The value of :math:`A_{\nu;T}(\omega)` at frequency ``omega``.
        """
        result = 0.0
        for subcmpnt in self.subcmpnts:
            result += subcmpnt.eval(omega)

        return result



class SpectralDensityCmpnt0T():
    r"""The zero-temperature spectral density of some component of the noise.

    For background information on spectral densities, see the documentation for
    the module :mod:`sbc.bath`, and the class 
    :class:`sbc.bath.SpectralDensityCmpnt`. 

    This class represents the spectral density of some component of the noise 
    [e.g. the :math:`y`-component] at zero-temperature. This quantity, which we 
    denote by :math:`A_{\nu;T=0}(\omega)` with :math:`\nu\in\{y, z\}`, was 
    introduced in Eq. :eq:`bath_spectral_density_cmpnt_expr` in the 
    documentation for the class :class:`sbc.bath.SpectralDensityCmpnt`.

    In many applications of interest, :math:`A_{\nu;T=0}(\omega)` will comprise
    of multiple subcomponents:

    .. math ::
        A_{\nu;T=0}(\omega)=\sum_{\varsigma}A_{\nu;T=0;\varsigma}(\omega),
        :label: bath_spectral_density_cmpnt_0T_breakdown_2

    where :math:`A_{\nu;T=0;\varsigma}(\omega)` is the 
    :math:`\varsigma^{\mathrm{th}}` subcomponent. As an example, 
    :math:`A_{z;T=0}(\omega)` may be strongly peaked at multiple frequencies, in
    which case it naturally decomposes into multiple subcomponents. If
    :math:`A_{\nu;T=0}(\omega)` naturally decomposes into multiple 
    subcomponents, it is best to treat each subcomponent separately in our 
    QUAPI+TN approach rather than treat :math:`A_{\nu;T=0}(\omega)` as a single 
    entity. For further discussion on this matter, see our detailed exposition 
    of our QUAPI+TN approach found :manual:`here <>`. 

    The quantity :math:`A_{\nu;T=0;\varsigma}(\omega)` is represented by the
    :class:`sbc.bath.SpectralDensitySubcmpnt0T` class. See the documentation for
    the aforementioned class for more details on 
    :math:`A_{\nu;T=0;\varsigma}(\omega)`.

    Parameters
    ----------
    subcmpnts : `array_like` (:class:`sbc.bath.SpectralDensitySubcmpnt0T`, ndim=1)
        The subcomponents of :math:`A_{\nu;T=0}(\omega)`, i.e. the
        :math:`A_{\nu;T=0;\varsigma}(\omega)`.

    Attributes
    ----------
    subcmpnts : `array_like` (:class:`sbc.bath.SpectralDensitySubcmpnt0T`, ndim=1), read-only
        The subcomponents of :math:`A_{\nu;T=0}(\omega)`, i.e. the
        :math:`A_{\nu;T=0;\varsigma}(\omega)`.
    """
    def __init__(self, subcmpnts):
        self.subcmpnts = subcmpnts

        return None



    def eval(self, omega):
        r"""Evaluate :math:`A_{\nu;T=0}(\omega)` at frequency ``omega``.

        Parameters
        ----------
        omega : `float`
            Frequency.

        Returns
        -------
        result : `float`
            The value of :math:`A_{\nu;T=0}(\omega)` at frequency ``omega``.
        """
        result = 0.0
        for subcmpnt in self.subcmpnts:
            result += subcmpnt.eval(omega)

        return result



class SpectralDensitySubcmpnt():
    r"""A subcomponent of some finite-temperature spectral density of the noise.

    For background information on spectral densities, see the documentation for
    the module :mod:`sbc.bath`, and the class 
    :class:`sbc.bath.SpectralDensityCmpnt`. The latter introduces the concept
    of breaking down a given finite-temperature spectral density of some 
    component of the noise [e.g. the :math:`y`-component] into subcomponents. We
    denote these subcomponents by :math:`A_{\nu;T;\varsigma}(\omega)`, with
    :math:`\nu\in\{y, z\}` and :math:`\varsigma` indicating the subcomponent.
    
    This class represents a single :math:`A_{\nu;T;\varsigma}(\omega)`.

    As briefly discussed in the documentation for the class 
    :class:`sbc.bath.SpectralDensityCmpnt`, :math:`A_{\nu;T;\varsigma}(\omega)`
    can be expressed as

    .. math ::
        A_{\nu;T;\varsigma}(\omega)=\text{sign}(\omega)
        \frac{A_{\nu;T=0;\varsigma}\left(\left|\omega\right|\right)}
        {1-e^{-\beta\omega}},
        :label: bath_spectral_density_subcmpt_expr_2

    where

    .. math ::
        \text{sign}\left(\omega\right)=\begin{cases}
        1 & \text{if }\omega>0,\\
        -1 & \text{if }\omega<0,
        \end{cases}
        :label: bath_spectral_density_subcmpnt_sign_function

    :math:`A_{\nu;T=0;\varsigma}(\omega)` is the zero-temperature limit of
    :math:`A_{\nu;T;\varsigma}(\omega)`, and :math:`\beta=1/(k_B T)` with 
    :math:`k_B` being the Boltzmann constant and :math:`T` being the 
    temperature. 

    The quantity :math:`A_{\nu;T=0;\varsigma}(\omega)` is
    represented by the :class:`sbc.bath.SpectralDensitySubcmpnt0T` class.

    Parameters
    ----------
    limit_0T : :class:`sbc.bath.SpectralDensitySubcmpnt0T`
        The zero-temperature limit of :math:`A_{\nu;T;\varsigma}(\omega)`, i.e.
        :math:`A_{\nu;T=0;\varsigma}(\omega)`.
    beta : `float`
        The inverse temperature, :math:`\beta=1/(k_B T)`, with :math:`k_B` 
        being the Boltzmann constant and :math:`T` being the temperature.

    Attributes
    ----------
    limit_0T : :class:`sbc.bath.SpectralDensitySubcmpnt0T`, read-only
        The zero-temperature limit of :math:`A_{\nu;T;\varsigma}(\omega)`, i.e.
        :math:`A_{\nu;T=0;\varsigma}(\omega)`.
    beta : `float`, read-only
        The inverse temperature, :math:`\beta=1/(k_B T)`, with :math:`k_B` 
        being the Boltzmann constant and :math:`T` being the temperature.
    """
    def __init__(self, limit_0T, beta):
        self.limit_0T = limit_0T
        self.beta = beta

        return None



    def eval(self, omega):
        r"""Evaluate :math:`A_{\nu;T;\varsigma}(\omega)` at frequency ``omega``.

        Parameters
        ----------
        omega : `float`
            Frequency.

        Returns
        -------
        result : `float`
            The value of :math:`A_{\nu;T;\varsigma}(\omega)` at frequency 
            ``omega``.
        """
        if np.abs(omega) > self.limit_0T.hard_cutoff_freq:
            result = 0.0
        else:
            result = self._eval(omega)

        return result



    def _eval(self, omega):
        abs_omega = np.abs(omega)
        beta_omega = self.beta * omega

        # See Appendix D.2 of detailed manuscript on our QUAPI-TN approach for
        # the rational behind the following implementation. In particular, see
        # the discussion starting just above Eq. (636) to Eq. (641).
        if omega == 0.0:
            result = self.limit_0T.zero_pt_derivative / self.beta
        elif 0 < abs_omega * self.beta < 1.0e-3:
            result = (np.sign(omega)
                      * (self.limit_0T._eval(abs_omega) / beta_omega)
                      * (1.0 + beta_omega/2.0
                         + (beta_omega**2)/12.0 - (beta_omega**4) / 720.0))
        elif beta_omega <= 1.0e-3:
            result = -(np.exp(beta_omega) * self.limit_0T._eval(abs_omega)
                       / (np.exp(beta_omega) - 1.0))
        else:
            result = (self.limit_0T._eval(abs_omega)
                      / (1.0 - np.exp(-beta_omega)))

        return result



class SpectralDensitySubcmpnt0T():
    r"""A subcomponent of some zero-temperature spectral density of the noise.

    For background information on spectral densities, see the documentation for
    the module :mod:`sbc.bath`, and the class 
    :class:`sbc.bath.SpectralDensityCmpnt0T`. The latter discusses the concept
    of breaking down a given zero-temperature spectral density of some component
    of the noise [e.g. the :math:`y`-component] into subcomponents. We denote
    these subcomponents by :math:`A_{\nu;T=0;\varsigma}(\omega)`, with
    :math:`\nu\in\{y, z\}` and :math:`\varsigma` indicating the subcomponent.
    
    This class represents a single :math:`A_{\nu;T=0;\varsigma}(\omega)`.

    In the continuum limit, :math:`A_{\nu;T=0;\varsigma}(\omega)` becomes a 
    continuous function of :math:`\omega` satisfying:

    .. math ::
        A_{\nu;T=0;\varsigma}(\omega) \ge 0,
        :label: bath_spectral_density_subcmpnt_0T_properties_1

    and

    .. math ::
        A_{\nu;T=0;\varsigma}(\omega \le 0) = 0,
        :label: bath_spectral_density_subcmpnt_0T_properties_2

    In order for the open-system dynamics to be "well-behaved", 
    :math:`A_{\nu;T=0;\varsigma}(\omega)` must satisfy:

    .. math ::
        \left|\lim_{\omega \to 0} 
        \frac{A_{\nu;T=0;\varsigma}}{\omega}\right| < \infty.
        :label: bath_spectral_density_subcmpnt_0T_requirement

    In our QUAPI+TN approach, frequency integrals are performed with integrands
    containing the :math:`A_{\nu;T=0;\varsigma}(\omega)`. Each integrand
    contains a removable singularity at :math:`\omega=0`. In order to handle
    these integrands properly in our approach, we require the limit of the
    derivative of each :math:`A_{\nu;T=0;\varsigma}(\omega)` as :math:`\omega`
    approaches zero from the right.

    Parameters
    ----------
    func_form : `func` (`float`, `**kwargs`)
        The functional form of :math:`A_{\nu;T=0;\varsigma}(\omega)`, where the
        first function argument of ``func_form`` is the frequency
        :math:`\omega`. ``func_form`` needs to be well-defined for non-negative 
        frequencies and must satisfy 
        Eqs. :eq:`bath_spectral_density_subcmpnt_0T_properties_1`  and
        :eq:`bath_spectral_density_subcmpnt_0T_requirement`. Note that 
        ``func_form`` is effectively ignored for negative frequencies as
        :math:`A_{\nu;T=0;\varsigma}(\omega)` is assumed to be zero for those
        frequencies.
    func_kwargs : `dict`
        A dictionary specifying specific values of the keyword arguments of
        ``func_form``. If there are no keyword arguments, then an empty
        dictionary should be passed.
    hard_cutoff_freq : `float`
        A hard cutoff frequency. For frequencies ``omega`` satisfying
        ``omega >= hard_cutoff_freq``, :math:`A_{\nu;T=0;\varsigma}(\omega)`
        evaluates to zero. ``hard_cutoff_freq`` is expected to be positive.
    zero_pt_derivative : `float` | `None`, optional
        The limit of the derivative of :math:`A_{\nu;T=0;\varsigma}(\omega)` as
        :math:`\omega` approaches zero from the right. If ``zero_pt_derivative``
        is set to `None` [i.e. the default value], then it will be estimated
        automatically by ``sbc``.

    Attributes
    ----------
    func_form : `func` (`float`, `**kwargs`), read-only
        The functional form of :math:`A_{\nu;T=0;\varsigma}(\omega)`, where the
        first function argument of ``func_form`` is the frequency
        :math:`\omega`. Note that ``func_form`` is effectively ignored for 
        negative frequencies as :math:`A_{\nu;T=0;\varsigma}(\omega)` is 
        assumed to be zero for those frequencies.
    func_kwargs : `dict`, read-only
        A dictionary specifying specific values of the keyword arguments of
        ``func_form``.
    hard_cutoff_freq : `float`, read-only
        A hard cutoff frequency. For frequencies ``omega`` satisfying
        ``omega >= hard_cutoff_freq``, :math:`A_{\nu;T=0;\varsigma}(\omega)`
        evaluates to zero.
    zero_pt_derivative : `float` | `None`, read-only
        The limit of the derivative of :math:`A_{\nu;T=0;\varsigma}(\omega)` as
        :math:`\omega` approaches zero from the right. 
    """
    def __init__(self,
                 func_form,
                 func_kwargs,
                 hard_cutoff_freq,
                 zero_pt_derivative=None):
        omega = 0
        
        try:
            func_form(omega, **func_kwargs)  # Check TypeErrors.
        except:
            raise TypeError("The given dictionary `func_kwargs` that is "
                            "suppose to specify the keyword arguments of the "
                            "given function `func_form`, used to construct an "
                            "instance of the "
                            "`sbc.bath.SpectralDensitySubcmpnt0T` class, "
                            "is not compatible with `func_form`.")
        
        self.func_form = func_form
        self.func_kwargs = copy.deepcopy(func_kwargs)
        self.hard_cutoff_freq = abs(hard_cutoff_freq)

        self.zero_pt_derivative = zero_pt_derivative
        if zero_pt_derivative == None:
            self.zero_pt_derivative = self._eval(1.0e-30) / 1.0e-30

        return None



    def eval(self, omega):
        r"""Evaluate :math:`A_{\nu;T=0;\varsigma}(\omega)` at frequency 
        ``omega``.

        Parameters
        ----------
        omega : `float`
            Frequency.

        Returns
        -------
        result : `float`
            The value of :math:`A_{\nu;T=0;\varsigma}(\omega)` at frequency 
            ``omega``.
        """
        result = self._eval(omega)
        result *= np.heaviside(self.hard_cutoff_freq - omega, 0)
        result *= np.heaviside(omega, 0)

        return result



    def _eval(self, omega):
        result = self.func_form(omega, **self.func_kwargs)

        return result



class Model():
    r"""The bath's model components.

    For background information on spectral densities and system memory, see the 
    documentation for the module :mod:`sbc.bath`, and the class 
    :class:`sbc.bath.SpectralDensityCmpnt`. 

    Parameters
    ----------
    spectral_density_y_cmpnt_0T : :class:`sbc.bath.SpectralDensityCmpnt0T`
        The zero-temperature limit of the spectral density of the 
        :math:`y^{\mathrm{th}}` component of the noise, 
        :math:`A_{y;T=0}(\omega)`.
    spectral_density_z_cmpnt_0T : :class:`sbc.bath.SpectralDensityCmpnt0T`
        The zero-temperature limit of the spectral density of the 
        :math:`z^{\mathrm{th}}` component of the noise, 
        :math:`A_{z;T=0}(\omega)`.
    beta : `float`
        The inverse temperature, :math:`\beta=1/(k_B T)`, with :math:`k_B` 
        being the Boltzmann constant and :math:`T` being the temperature.
    tau : `float`
        The bath correlation time, also known as the system's memory 
        :math:`\tau`. ``tau`` is expected to be non-negative.

    Attributes
    ----------
    spectral_density_y_cmpnt : :class:`sbc.bath.SpectralDensityCmpnt`, read-only
        The finite-temperature spectral density of the :math:`y^{\mathrm{th}}` 
        component of the noise, :math:`A_{y;T}(\omega)`.
    spectral_density_z_cmpnt : :class:`sbc.bath.SpectralDensityCmpnt`, read-only
        The finite-temperature spectral density of the :math:`z^{\mathrm{th}}` 
        component of the noise, :math:`A_{z;T}(\omega)`.
    tau : `float`, read-only
        The bath correlation time, also known as the system's memory 
        :math:`\tau`.
    """
    def __init__(self,
                 spectral_density_y_cmpnt_0T,
                 spectral_density_z_cmpnt_0T,
                 beta,
                 tau):
        self.spectral_density_y_cmpnt = \
            SpectralDensityCmpnt(spectral_density_y_cmpnt_0T, beta)
        self.spectral_density_z_cmpnt = \
            SpectralDensityCmpnt(spectral_density_z_cmpnt_0T, beta)
        self.tau = tau

        return None



def _calc_K_tau(tau, dt):
    r"""This function implements Eq. (67) of the detailed manuscript (DM) on our
    QUAPI-TN approach. See Sec. 2.5 of DM for further context.
    """
    K_tau = max(0, ceil((tau - 7.0*dt/4.0) / dt)) + 3

    return K_tau
