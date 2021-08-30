Single qubit subject to y-noise
===============================

Introduction
------------

In this example, we implement a simulation of the dynamics of a single qubit
subject to time-dependent transverse and longitudinal fields, and environmental
noise that is coupled to the :math:`y`--component of the qubit's spin. The
spectral density of noise comprises of two Dirac delta-like peak. The
system-environment coupling energy scale is also time-dependent. The simulation
tracks :math:`\left\langle\hat{\sigma}_{x;r}\left(t\right)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{y;r}\left(t\right)\right\rangle`, and
:math:`\left\langle\hat{\sigma}_{z;r}\left(t\right)\right\rangle`.

This example tests the algorithm implemented in the ``spinbosonchain`` library
that is used to calculate local influence paths for the case where the system is
subject to multi-component noise (in this case with a double-peak spectrum), and
the system's bath correlation time, or "memory", is larger than the total
simulation time.

Hamiltonian
-----------

For background information on the generalized spin-boson chain model considered
in ``spinbosonchain``, see the documentation for the modules
:mod:`spinbosonchain.system` and :mod:`spinbosonchain.bath`.

In this example, we consider the following single-site model:

.. math ::
    \hat{H}\left(t\right)=\hat{H}^{\left(A\right)}\left(t\right)
    +\hat{H}^{\left(B\right)}+\hat{H}^{\left(AB\right)}\left(t\right),
    :label: single_qubit_y_noise_total_hamiltonian

where :math:`\hat{H}^{\left(A\right)}\left(t\right)` is the system Hamiltonian:

.. math ::
    \hat{H}^{\left(A\right)}\left(t\right)\equiv\sum_{r=0}^{L-1}
    \left\{ h_{x;r}\left(t\right)\hat{\sigma}_{x;r}+h_{z;r}\left(t\right)
    \hat{\sigma}_{z;r}\right\},
    :label: single_qubit_y_noise_H_A

with :math:`L` being the number of qubits:

.. math ::
    L=1,
    :label: single_qubit_y_noise_L

:math:`h_{x;r}\left(t\right)` and :math:`h_{z;r}\left(t\right)` being the
transverse and longitudinal field energy scales respectively applied to site
:math:`r` at time :math:`t` :

.. math ::
    h_{x;r}\left(t\right)=-\frac{A\left(t\right)}{2},
    :label: single_qubit_y_noise_h_x_r

.. math ::
    h_{z;r}\left(t\right)=\frac{B\left(t\right)}{2}h_{r},
    :label: single_qubit_y_noise_h_z_r

.. math ::
    A\left(t\right)=2\left\{ 1-s\left(t\right)\right\} ,
    :label: single_qubit_y_noise_A

.. math ::
    B\left(t\right)=2s\left(t\right),
    :label: single_qubit_y_noise_B

.. math ::
    s\left(t\right)=\frac{t}{t_{a}},
    :label: single_qubit_y_noise_normalized_anneal_fraction

.. math ::
    h_{r}=\frac{1}{10},
    :label: single_qubit_y_noise_h_r

:math:`t_{a}` being the total simulation time, and :math:`\hat{\sigma}_{\nu;r}`
being the :math:`\nu^{\text{th}}` Pauli operator acting on site :math:`r`;
:math:`\hat{H}^{\left(B\right)}` is the bath Hamiltonian, which describes a
collection of decoupled harmonic oscillators:

.. math ::
    \hat{H}^{\left(B\right)}\equiv\sum_{\nu\in\left\{ y\right\} }
    \sum_{r=0}^{L-1}\sum_{\epsilon}\omega_{\nu;\epsilon}
    \hat{b}_{\nu;r;\epsilon}^{\dagger}
    \hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}},
    :label: single_qubit_y_noise_H_B

with :math:`\epsilon` being the oscillator mode index,
:math:`\hat{b}_{\nu;r;\epsilon}^{\dagger}` and
:math:`\hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}}` being the bosonic creation
and annihilation operators respectively for the harmonic oscillator at site
:math:`r` in mode :math:`\epsilon` with angular frequency
:math:`\omega_{\nu;\epsilon}`, coupled to the :math:`\nu^{\text{th}}` component
of the spin at the same site; :math:`\hat{H}^{\left(AB\right)}\left(t\right)`
describes the coupling between the system and the bath:

.. math ::
    \hat{H}^{\left(AB\right)}\left(t\right)=-\sum_{\nu\in\left\{ y\right\} }
    \sum_{r=0}^{L-1}\hat{\sigma}_{\nu;r}\hat{\mathcal{Q}}_{\nu;r}\left(t\right),
    :label: single_qubit_y_noise_H_AB

with :math:`\hat{\mathcal{Q}}_{\nu;r}\left(t\right)` being the generalized
reservoir force operator at site :math:`r` that acts on the
:math:`\nu^{\text{th}}` component of the spin at the same site at time
:math:`t`:

.. math ::
    \hat{\mathcal{Q}}_{\nu;r}\left(t\right)=
    \mathcal{E}_{\nu;r}^{\left(\lambda\right)}\left(t\right)
    \hat{Q}_{\nu;r},
    :label: single_qubit_y_noise_mathcal_Q

with :math:`\mathcal{E}_{v;r}^{\left(\lambda\right)}\left(t\right)` being a
time-dependent energy scale:

.. math ::
    \mathcal{E}_{y;r=0}^{\left(\lambda\right)}\left(t\right)=
    \frac{1}{2}\sqrt{A\left(t\right)},
    :label: single_qubit_y_noise_mathcal_E_y

:math:`\hat{Q}_{\nu;r}` being a rescaled generalized reservoir force:

.. math ::
    \hat{Q}_{\nu;r}=-\sum_{\epsilon}\lambda_{\nu;r;\epsilon}
    \left\{ \hat{b}_{\nu;r;\epsilon}^{\dagger}
    +\hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}}\right\},
    :label: single_qubit_y_noise_Q_v_r

:math:`\lambda_{\nu;r;\epsilon}` being the coupling strength between the
:math:`\nu^{\text{th}}` component of the spin at site :math:`r` and the harmonic
oscillator at the same site in mode :math:`\epsilon`:

.. math ::
    \lambda_{\nu;r;\epsilon}=
    \sum_{\varsigma = 0}^{1}
    \delta_{\epsilon,\epsilon_{\varsigma}}\sqrt{\eta_{\nu;r;\varsigma}},
    :label: single_qubit_y_noise_lambda

.. math ::
    \eta_{y;r=0;\varsigma=0}=\frac{4}{5};
    \quad\eta_{y;r=0;\varsigma=1}=\frac{1}{5};
    :label: single_qubit_y_noise_etas

.. math ::
    \omega_{y ;\epsilon=\epsilon_{0}}=5;
    \quad\omega_{y ;\epsilon=\epsilon_{1}}=1.
    :label: single_qubit_y_noise_peak_frequencies

Initial state
-------------

We assume that the system and bath together are initially prepared in a state at
time :math:`t=0` corresponding to the state operator:

.. math ::
    \hat{\rho}^{\left(i\right)}=\hat{\rho}^{\left(i,A\right)}\otimes
    \hat{\rho}^{\left(i,B\right)},
    :label: single_qubit_y_noise_rho_i

where :math:`\hat{\rho}^{\left(i,A\right)}` is the system's reduced state
operator at :math:`t=0`, and :math:`\hat{\rho}^{\left(i,B\right)}` is the bath's
reduced state operator at :math:`t=0`:

.. math ::
    \hat{\rho}^{\left(i,B\right)}\equiv
    \frac{e^{-\beta\hat{H}^{\left(B\right)}}}{\text{Tr}^{\left(B\right)}
    \left\{ e^{-\beta\hat{H}^{\left(B\right)}}\right\} },
    :label: single_qubit_y_noise_rho_i_B

with :math:`\beta=1/\left(k_{B}T\right)`, :math:`k_{B}` being the Boltzmann
constant, :math:`T` being the temperature, and :math:`\text{Tr}_{B}\left\{
\cdots\right\}` being the partial trace with respect to the bath degrees of
freedom. In this example, we assume that the system is prepared in a state
corresponding to a state operator of the form:

.. math ::
    \hat{\rho}^{\left(i,A\right)}=\left|\Psi^{\left(i\right)}\right\rangle
    \left\langle \Psi^{\left(i\right)}\right|,
    :label: single_qubit_y_noise_rho_i_A

where :math:`\left|\Psi^{\left(i\right)}\right\rangle`

.. math ::
    \left|\Psi^{\left(i\right)}\right\rangle =\bigotimes_{r=0}^{L-1}
    \left|+x\right\rangle _{r},
    :label: single_qubit_y_noise_psi_i

with

.. math ::
    \hat{\sigma}_{\nu;r}\left|\pm\nu\right\rangle _{r}=
    \pm\left|\pm\nu\right\rangle _{r}.
    :label: single_qubit_y_noise_defining_nu_eigenstates

Spectral densities of noise
---------------------------

To determine the noisy dynamics of the single-qubit system, it suffices to
specify the the "spin" part of the Hamiltonian :math:`\hat{H}^{\left(A\right)}`
Eq. :eq:`single_qubit_y_noise_H_A`, and the spectral densitiy of noise at
temperature :math:`T` in the continuum limit in bosonic space:

.. math ::
    A_{\nu;r;T}\left(\omega\right)=\int_{-\infty}^{\infty}dt\,
    e^{i\omega t}\text{Tr}^{\left(B\right)}\left\{ \hat{\rho}^{\left(i,B\right)}
    \hat{Q}_{\nu;r}^{\left(B\right)}\left(t\right)
    \hat{Q}_{\nu;r}^{\left(B\right)}\left(0\right)\right\},
    :label: single_qubit_y_noise_introducing_spectral_densitities

where

.. math ::
    \hat{Q}_{z;r}^{\left(B\right)}\left(t\right)=e^{i\hat{H}^{\left(B\right)}t}
    \hat{Q}_{z;r}e^{-i\hat{H}^{\left(B\right)}t};
    :label: single_qubit_y_noise_Q_operator_in_heisenberg_picture

and

.. math ::
    \hat{\rho}^{\left(i,B\right)}\equiv
    \frac{e^{-\beta\hat{H}^{\left(B\right)}}}{\text{Tr}^{\left(B\right)}
    \left\{ e^{-\beta\hat{H}^{\left(B\right)}}\right\} },
    :label: single_qubit_y_noise_initial_state_operator

with :math:`\beta=1/\left(k_{B}T\right)`:

.. math ::
    \beta = 1,
    :label: single_qubit_y_noise_beta

and :math:`k_{B}` being the Boltzmann constant. As discussed in the
documentation of the class :class:`spinbosonchain.bath.SpectralDensity`, we
express :math:`A_{\nu;r;T}\left(\omega\right)` as

.. math ::
    A_{\nu;r;T}\left(\omega\right)=\text{sign}\left(\omega\right)
    \frac{A_{\nu;r;T=0}\left(\left|\omega\right|\right)}{1-e^{-\beta\omega}},
    :label: single_qubit_y_noise_A_v_r_T

where :math:`A_{\nu;r;T=0}\left(\omega\right)` is the zero-temperature limit:

.. math ::
    A_{\nu;r;T=0}\left(\omega\right)=\sum_{\varsigma = 0}^{1}
    A_{\nu;r;T=0;\varsigma}\left(\omega\right),
    :label: single_qubit_y_noise_A_v_r_0T

with :math:`A_{\nu;r;T=0;\varsigma}\left(\omega\right)` being the
:math:`\varsigma^{\text{th}}`-component of
:math:`A_{\nu;r;T=0}\left(\omega\right)`. From
Eqs. :eq:`single_qubit_y_noise_Q_v_r`-:eq:`single_qubit_y_noise_peak_frequencies`,
and
:eq:`single_qubit_y_noise_introducing_spectral_densitities`-:eq:`single_qubit_y_noise_initial_state_operator`,
we see that the :math:`A_{\nu;r;T=0;\varsigma}\left(\omega\right)` are of the
form:

.. math ::
    A_{\nu;r;T=0;\varsigma}\left(\omega\right)=
    2\pi\eta_{\nu;r;\varsigma}\delta
    \left(\omega-\omega_{\nu;\epsilon_{\varsigma}}\right).
    :label: single_qubit_y_noise_A_v_r_0T_varsigma

We choose a model with a discrete spectral density of noise so that we can
alternatively calculate the dynamics using exact diagonalization. By comparing
the two approaches, we can verify the correctness of the algorithm implemented
in the ``spinbosonchain`` library that is used to calculate local influence
paths for the case where the system is subject to multi-component noise (in this
case with a double-peak spectrum), and the system's bath correlation time, or
"memory", is larger than the total simulation time. For background information
on local influence paths, see our detailed exposition on our QUAPI+TN approach
found :manual:`here <>`.

In order to implement the discrete spectral density of noise in ``Python``, we
approximate the Dirac delta functions by normal distributions with arbitrarily
small variance:

.. math ::
    A_{\nu;r;T=0;\varsigma}\left(\omega\right)
    \approx 2\pi\eta_{\nu;r;\varsigma}\delta_{w}
    \left(\omega-\omega_{\nu;\epsilon_{\varsigma}}\right)
    \left\{ \Theta\left(\omega-
    \omega_{\varsigma}^{\left(\text{IR}\right)}\right)-
    \Theta\left(\omega-\omega_{\varsigma}^{\left(\text{UV}\right)}\right)
    \right\},
    :label: single_qubit_y_noise_A_v_r_0T_varsigma_rewritten

where :math:`\Theta\left(\omega\right)` is the Heaviside step function,

.. math ::
    \delta_{w}\left(\omega-\omega_{\nu;\epsilon_{\varsigma}}\right)=
    \frac{1}{\sqrt{w^{2}\pi}}\exp\left(-\left\{
    \frac{\omega-\omega_{\nu;\epsilon_{\varsigma}}}{w}\right\} ^{2}\right),
    :label: single_qubit_y_noise_dirac_delta_sequence

:math:`\omega_{\varsigma}^{\left(\text{IR}\right)}` and
:math:`\omega_{\varsigma}^{\left(\text{UV}\right)}` are infrared and ultraviolet
cutoff frequencies respectively:

.. math ::
    \omega_{\varsigma}^{\left(\text{IR}\right)}
    =\omega_{\nu;\epsilon_{\varsigma}}-8w,
    :label: single_qubit_y_noise_IR_cutoff
	    
.. math ::	    
    \omega_{\varsigma}^{\left(\text{UV}\right)}
    =\omega_{\nu;\epsilon_{\varsigma}}+8w,
    :label: single_qubit_y_noise_UV_cutoff

and

.. math ::
    w=10^{-6}.
    :label: single_qubit_y_noise_w_2

Quantities tracked
------------------

The following quantities are tracked in the ``spinbosonchain`` simulation:

.. math ::
    \left\langle \hat{\sigma}_{x;r}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}
    \left(t\right)\hat{\sigma}_{x;r}\right\},
    :label: single_qubit_y_noise_ev_of_sx

with :math:`\hat{\rho}^{\left(A\right)}\left(t\right)` being the system's
reduced state operator at time :math:`t`;

.. math ::
    \left\langle \hat{\sigma}_{y;r}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}
    \left(t\right)\hat{\sigma}_{y;r}\right\};
    :label: single_qubit_y_noise_ev_of_sy

and

.. math ::
    \left\langle \hat{\sigma}_{z;r}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \hat{\sigma}_{z;r}\right\}.
    :label: single_qubit_y_noise_ev_of_sz

If ``quspin`` has been installed, then the quantities given by
Eqs. :eq:`single_qubit_y_noise_ev_of_sx`-:eq:`single_qubit_y_noise_ev_of_sz`
will also be calculated by exact diagonalization for comparison sake.

Code
----

Below is the code that implements the simulation described above. You can also
find the same code in the file `examples/single-qubit/y-noise/example.py` of
the repository.

.. literalinclude:: ../examples/single-qubit/y-noise/example.py
