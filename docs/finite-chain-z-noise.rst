Finite chain subject to z-noise
===============================

Introduction
------------

In this example, we implement a simulation of the dynamics of a 3-qubit chain
subject to time-dependent transverse and longitudinal fields, and environmental
noise that is coupled to the :math:`z`-component of the center qubit's spin. The
spectral density of noise comprises of a single Dirac delta like peak. The
system-environment coupling energy scale is also time-dependent. The simulation
tracks the expectation value of the system's energy :math:`\left\langle
E\left(t\right)\right\rangle`, :math:`\left\langle
\hat{\sigma}_{x;r}\left(t\right)\right\rangle`, :math:`\left\langle
\hat{\sigma}_{z;r}\left(t\right)\right\rangle`, :math:`\left\langle
\hat{\sigma}_{z;r}\left(t\right)
\hat{\sigma}_{z;r+1}\left(t\right)\right\rangle`, :math:`\left\langle
\hat{\sigma}_{y;r}\left(t\right)
\hat{\sigma}_{z;r+1}\left(t\right)\right\rangle`, :math:`\left\langle
\hat{\sigma}_{x;0}\left(t\right)\hat{\sigma}_{z;1}\left(t\right)
\hat{\sigma}_{x;2}\left(t\right)\right\rangle`, :math:`\left\langle
\hat{\sigma}_{x;0}\left(t\right)\hat{\sigma}_{x;1}\left(t\right)
\hat{\sigma}_{x;2}\left(t\right)\right\rangle`, the probability of measuring
:math:`\sigma_{z;r}\left(t\right)=+1` at each site :math:`r`, and the
probability of measuring :math:`\sigma_{z;r}\left(t\right)=-1` at each site
:math:`r`. The simulation also checks whether the system is entangled using the
realignment criterion.

This example tests the algorithm implemented in the ``spinbosonchain`` library
that is used to calculate the dynamics of a 3-qubit system subject to
multi-component :math:`z`-noise, where the system's bath correlation time, or
"memory", is larger than the total simulation time.

Hamiltonian
-----------

For background information on the generalized spin-boson chain model considered
in ``spinbosonchain``, see the documentation for the modules
:mod:`spinbosonchain.system` and :mod:`spinbosonchain.bath`.

In this example, we consider the following 3-site model:

.. math ::
    \hat{H}\left(t\right)=\hat{H}^{\left(A\right)}\left(t\right)
    +\hat{H}^{\left(B\right)}+\hat{H}^{\left(AB\right)}\left(t\right),
    :label: finite_chain_z_noise_total_hamiltonian

where :math:`\hat{H}^{\left(A\right)}\left(t\right)` is the system Hamiltonian:

.. math ::
    \hat{H}^{\left(A\right)}\left(t\right)\equiv\sum_{r=0}^{L-1}
    \left\{ h_{x;r}\left(t\right)\hat{\sigma}_{x;r}+h_{z;r}\left(t\right)
    \hat{\sigma}_{z;r}\right\} +\sum_{r=0}^{L-2}J_{z,z;r,r+1}\left(t\right)
    \hat{\sigma}_{z;r}\hat{\sigma}_{z;r+1},
    :label: finite_chain_z_noise_H_A

with :math:`L` being the number of qubits:

.. math ::
    L=3,
    :label: finite_chain_z_noise_L

:math:`h_{x;r}\left(t\right)` and :math:`h_{z;r}\left(t\right)` being the
transverse and longitudinal field energy scales respectively applied to site
:math:`r` at time :math:`t` :

.. math ::
    h_{x;r}\left(t\right)=-\frac{A\left(t\right)}{2},
    :label: finite_chain_z_noise_h_x_r

.. math ::
    h_{z;r}\left(t\right)=\frac{B\left(t\right)}{2}h_{r},
    :label: finite_chain_z_noise_h_z_r

.. math ::
    A\left(t\right)=2\left\{ 1-s\left(t\right)\right\} ,
    :label: finite_chain_z_noise_A

.. math ::
    B\left(t\right)=2s\left(t\right),
    :label: finite_chain_z_noise_B

.. math ::
    s\left(t\right)=\frac{t}{t_{a}},
    :label: finite_chain_z_noise_normalized_anneal_fraction

.. math ::
    h_{r}=\begin{cases}
    \frac{1}{5}, & \text{if }r=0,\\
    0, & \text{if }r=1,\\
    -\frac{1}{10}, & \text{if }r=2,
    \end{cases}
    :label: finite_chain_z_noise_h_r

:math:`t_{a}` being the total simulation time,
:math:`J_{z,z;r,r+1}\left(t\right)` being the longitudinal coupling energy scale
between sites :math:`r` and :math:`r+1` at time :math:`t`:

.. math ::
    J_{z,z;r,r+1}\left(t\right)=\frac{B\left(t\right)}{2}J_{r,r+1},
    :label: finite_chain_z_noise_J_zz

.. math ::
    J_{r,r+1}=\begin{cases}
    -1, & \text{if }r=0,\\
    -\frac{9}{10}, & \text{if }r=1,
    \end{cases}
    :label: finite_chain_z_noise_J

and :math:`\hat{\sigma}_{\nu;r}` being the :math:`\nu^{\text{th}}` Pauli
operator acting on site :math:`r`; :math:`\hat{H}^{\left(B\right)}` is the bath
Hamiltonian, which describes a collection of decoupled harmonic oscillators:

.. math ::
    \hat{H}^{\left(B\right)}\equiv\sum_{\nu\in\left\{z\right\} }
    \sum_{r=0}^{L-1}\sum_{\epsilon}\omega_{\nu;\epsilon}
    \hat{b}_{\nu;r;\epsilon}^{\dagger}
    \hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}},
    :label: finite_chain_z_noise_H_B

with :math:`\epsilon` being the oscillator mode index,
:math:`\hat{b}_{\nu;r;\epsilon}^{\dagger}` and
:math:`\hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}}` being the bosonic creation
and annihilation operators respectively for the harmonic oscillator at site
:math:`r` in mode :math:`\epsilon` with angular frequency
:math:`\omega_{\nu;\epsilon}`, coupled to the :math:`\nu^{\text{th}}` component
of the spin at the same site; :math:`\hat{H}^{\left(AB\right)}\left(t\right)`
describes the coupling between the system and the bath:

.. math ::
    \hat{H}^{\left(AB\right)}\left(t\right)=-\sum_{\nu\in\left\{z\right\} }
    \sum_{r=0}^{L-1}\hat{\sigma}_{\nu;r}\hat{\mathcal{Q}}_{\nu;r}\left(t\right),
    :label: finite_chain_z_noise_H_AB

with :math:`\hat{\mathcal{Q}}_{\nu;r}\left(t\right)` being the generalized
reservoir force operator at site :math:`r` that acts on the
:math:`\nu^{\text{th}}` component of the spin at the same site at time
:math:`t`:

.. math ::
    \hat{\mathcal{Q}}_{\nu;r}\left(t\right)=
    \mathcal{E}_{\nu;r}^{\left(\lambda\right)}\left(t\right)
    \hat{Q}_{\nu;r},
    :label: finite_chain_z_noise_mathcal_Q

with :math:`\mathcal{E}_{v;r}^{\left(\lambda\right)}\left(t\right)` being a
time-dependent energy scale:

.. math ::
    \mathcal{E}_{z;r}^{\left(\lambda\right)}\left(t\right)=\begin{cases}
    0, & \text{if }r=0,\\
    \frac{3}{4}\sqrt{B\left(t\right)}, & \text{if }r=1,\\
    0, & \text{if }r=2,
    \end{cases}
    :label: finite_chain_z_noise_mathcal_E_z

:math:`\hat{Q}_{\nu;r}` being a rescaled generalized reservoir force:

.. math ::
    \hat{Q}_{\nu;r}=-\sum_{\epsilon}\lambda_{\nu;r;\epsilon}
    \left\{ \hat{b}_{\nu;r;\epsilon}^{\dagger}
    +\hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}}\right\},
    :label: finite_chain_z_noise_Q_v_r

:math:`\lambda_{\nu;r;\epsilon}` being the coupling strength between the
:math:`\nu^{\text{th}}` component of the spin at site :math:`r` and the harmonic
oscillator at the same site in mode :math:`\epsilon`:

.. math ::
    \lambda_{\nu;r;\epsilon}=\sum_{\varsigma=0}^{1}
    \delta_{\epsilon,\epsilon_{\varsigma}}\sqrt{\eta_{\nu;r;\varsigma}},
    :label: finite_chain_z_noise_lambda

.. math ::
    \eta_{z;0;0}=0; \quad \eta_{z;1;1}=\frac{4}{5}; \quad \eta_{z;1;2}=0;
    :label: finite_chain_z_noise_eta_z

.. math ::
    \omega_{z ;\epsilon=\epsilon_{0}}=1;
    :label: finite_chain_z_noise_peak_frequencies

Initial state
-------------

We assume that the system and bath together are initially prepared in a state at
time :math:`t=0` corresponding to the state operator:

.. math ::
    \hat{\rho}^{\left(i\right)}=\hat{\rho}^{\left(i,A\right)}\otimes
    \hat{\rho}^{\left(i,B\right)},
    :label: finite_chain_z_noise_rho_i

where :math:`\hat{\rho}^{\left(i,A\right)}` is the system's reduced state
operator at :math:`t=0`, and :math:`\hat{\rho}^{\left(i,B\right)}` is the bath's
reduced state operator at :math:`t=0`:

.. math ::
    \hat{\rho}^{\left(i,B\right)}\equiv
    \frac{e^{-\beta\hat{H}^{\left(B\right)}}}{\text{Tr}^{\left(B\right)}
    \left\{ e^{-\beta\hat{H}^{\left(B\right)}}\right\} },
    :label: finite_chain_z_noise_rho_i_B

with :math:`\beta=1/\left(k_{B}T\right)`, :math:`k_{B}` being the Boltzmann
constant, :math:`T` being the temperature, and :math:`\text{Tr}_{B}\left\{
\cdots\right\}` being the partial trace with respect to the bath degrees of
freedom. In this example, we assume that the system is prepared in a state
corresponding to a state operator of the form:

.. math ::
    \hat{\rho}^{\left(i,A\right)}=\left|\Psi^{\left(i\right)}\right\rangle
    \left\langle \Psi^{\left(i\right)}\right|,
    :label: finite_chain_z_noise_rho_i_A

where :math:`\left|\Psi^{\left(i\right)}\right\rangle`

.. math ::
    \left|\Psi^{\left(i\right)}\right\rangle =\bigotimes_{r=0}^{L-1}
    \left|+x\right\rangle _{r},
    :label: finite_chain_z_noise_psi_i

with

.. math ::
    \hat{\sigma}_{\nu;r}\left|\pm\nu\right\rangle _{r}=
    \pm\left|\pm\nu\right\rangle _{r}.
    :label: finite_chain_z_noise_defining_nu_eigenstates

Spectral densities of noise
---------------------------

To determine the noisy dynamics of the multi-qubit system, it suffices to
specify the the "spin" part of the Hamiltonian :math:`\hat{H}^{\left(A\right)}`
Eq. :eq:`finite_chain_z_noise_H_A`, and the spectral density of noise at
temperature :math:`T` in the continuum limit in bosonic space:

.. math ::
    A_{\nu;r;T}\left(\omega\right)=\int_{-\infty}^{\infty}dt\,
    e^{i\omega t}\text{Tr}^{\left(B\right)}\left\{ \hat{\rho}^{\left(i,B\right)}
    \hat{Q}_{\nu;r}^{\left(B\right)}\left(t\right)
    \hat{Q}_{\nu;r}^{\left(B\right)}\left(0\right)\right\},
    :label: finite_chain_z_noise_introducing_spectral_densitities

where

.. math ::
    \hat{Q}_{z;r}^{\left(B\right)}\left(t\right)=e^{i\hat{H}^{\left(B\right)}t}
    \hat{Q}_{z;r}e^{-i\hat{H}^{\left(B\right)}t};
    :label: finite_chain_z_noise_Q_operator_in_heisenberg_picture

and

.. math ::
    \hat{\rho}^{\left(i,B\right)}\equiv
    \frac{e^{-\beta\hat{H}^{\left(B\right)}}}{\text{Tr}^{\left(B\right)}
    \left\{ e^{-\beta\hat{H}^{\left(B\right)}}\right\} },
    :label: finite_chain_z_noise_initial_state_operator

with :math:`\beta=1/\left(k_{B}T\right)`:

.. math ::
    \beta = 1,
    :label: finite_chain_z_noise_beta

and :math:`k_{B}` being the Boltzmann constant. As discussed in the
documentation of the class :class:`spinbosonchain.bath.SpectralDensity`, we
express :math:`A_{\nu;r;T}\left(\omega\right)` as

.. math ::
    A_{\nu;r;T}\left(\omega\right)=\text{sign}\left(\omega\right)
    \frac{A_{\nu;r;T=0}\left(\left|\omega\right|\right)}{1-e^{-\beta\omega}},
    :label: finite_chain_z_noise_A_v_r_T

where :math:`A_{\nu;r;T=0}\left(\omega\right)` is the zero-temperature limit:

.. math ::
    A_{\nu;r;T=0}\left(\omega\right)=
    A_{\nu;r;T=0;\varsigma=0}\left(\omega\right),
    :label: finite_chain_z_noise_A_v_r_0T

with :math:`A_{\nu;r;T=0;\varsigma}\left(\omega\right)` being the
:math:`\varsigma^{\text{th}}`-component of
:math:`A_{\nu;r;T=0}\left(\omega\right)`. In this example, the spectral density
of noise trivially has one component. From
Eqs. :eq:`finite_chain_z_noise_Q_v_r`-:eq:`finite_chain_z_noise_peak_frequencies`,
and
:eq:`finite_chain_z_noise_introducing_spectral_densitities`-:eq:`finite_chain_z_noise_initial_state_operator`,
we see that the :math:`A_{\nu;r;T=0;\varsigma}\left(\omega\right)` are of the
form:

.. math ::
    A_{\nu;r;T=0;\varsigma}\left(\omega\right)=
    2\pi\eta_{\nu;r;\varsigma}\delta
    \left(\omega-\omega_{\nu;\epsilon_{\varsigma}}\right).
    :label: finite_chain_z_noise_A_v_r_0T_varsigma

We choose a model with a discrete spectral density of noise so that we can
alternatively calculate the dynamics using exact diagonalization. By comparing
the two approaches, we can verify the correctness of the algorithm implemented
in the ``spinbosonchain`` library that is used to calculate the dynamics of a
3-qubit system subject to multi-component :math:`z`-noise, where the system's
bath correlation time, or "memory", is larger than the total simulation time.

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
    :label: finite_chain_z_noise_A_v_r_0T_varsigma_rewritten

where

.. math ::
    \delta_{w}\left(\omega-\omega_{\nu;\epsilon_{\varsigma}}\right)=
    \frac{1}{\sqrt{w^{2}\pi}}\exp\left(-\left\{
    \frac{\omega-\omega_{\nu;\epsilon_{\varsigma}}}{w}\right\} ^{2}\right),
    :label: finite_chain_z_noise_dirac_delta_sequence

:math:`\omega_{\varsigma}^{\left(\text{IR}\right)}` and
:math:`\omega_{\varsigma}^{\left(\text{UV}\right)}` are infrared and ultraviolet
cutoff frequencies respectively:

.. math ::
    \omega_{\varsigma}^{\left(\text{IR}\right)}
    =\omega_{\nu;\epsilon_{\varsigma}}-8w,
    :label: finite_chain_z_noise_IR_cutoff
	    
.. math ::	    
    \omega_{\varsigma}^{\left(\text{UV}\right)}
    =\omega_{\nu;\epsilon_{\varsigma}}+8w,
    :label: finite_chain_z_noise_UV_cutoff

and
	    
.. math ::
    w=10^{-6}.
    :label: finite_chain_z_noise_w_2

Quantities tracked
------------------

The following quantities are tracked in the ``spinbosonchain`` simulation:

.. math ::
    \left\langle \hat{\sigma}_{x;r}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}
    \left(t\right)\hat{\sigma}_{x;r}\right\} ,
    \quad\text{for }r\in\left\{ 0,1,2\right\},
    :label: finite_chain_z_noise_ev_of_sx

with :math:`\hat{\rho}^{\left(A\right)}\left(t\right)` being the system's
reduced state operator at time :math:`t`;

.. math ::
    \left\langle \hat{\sigma}_{z;r}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \hat{\sigma}_{z;r}\right\} ,\quad\text{for }r\in\left\{ 0,1,2\right\};
    :label: finite_chain_z_noise_ev_of_sz

.. math ::
    \left\langle \hat{\sigma}_{z;r}\left(t\right)
    \hat{\sigma}_{z;r+1}\left(t\right)\right\rangle =\text{Tr}^{\left(A\right)}
    \left\{ \hat{\rho}^{\left(A\right)}\left(t\right)\hat{\sigma}_{z;r}
    \hat{\sigma}_{z;r+1}\right\} ,\quad\text{for }r\in\left\{ 0,1\right\};
    :label: finite_chain_z_noise_ev_of_szsz

.. math ::
    \left\langle \hat{\sigma}_{y;r}\left(t\right)
    \hat{\sigma}_{z;r+1}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \hat{\sigma}_{y;r}\hat{\sigma}_{z;r+1}\right\},
    \quad\text{for }r\in\left\{ 0,1\right\};
    :label: finite_chain_z_noise_ev_of_sysz

.. math ::
    \left\langle \hat{\sigma}_{x;0}\left(t\right)
    \hat{\sigma}_{z;1}\left(t\right)
    \hat{\sigma}_{x;2}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \hat{\sigma}_{x;0}\hat{\sigma}_{z;1}\hat{\sigma}_{x;2}\right\};
    :label: finite_chain_z_noise_ev_of_sxszsx

.. math ::
    \left\langle \hat{\sigma}_{x;0}\left(t\right)
    \hat{\sigma}_{x;1}\left(t\right)
    \hat{\sigma}_{x;2}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \hat{\sigma}_{x;0}\hat{\sigma}_{x;1}\hat{\sigma}_{x;2}\right\};
    :label: finite_chain_z_noise_ev_of_sxsxsx

.. math ::
    \text{Prob}\left(\boldsymbol{\sigma}_{z}=\pm1;t\right)=
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \bigotimes_{r=0}^{L-1}\left[\left|\pm z\right\rangle _{r}\left\langle
    \pm z\right|_{r}\right]\right\};
    :label: finite_chain_z_noise_probabilities

and

.. math ::
    \left\langle E\left(t\right)\right\rangle =\text{Tr}^{\left(A\right)}
    \left\{ \hat{\rho}^{\left(A\right)}\left(t\right)\hat{H}^{\left(A\right)}
    \left(t\right)\right\}.
    :label: finite_chain_z_noise_ev_of_energy

We also apply the realignment criterion at each time step to detect entanglement
[see the documentation for the function
:func:`spinbosonchain.state.realignment_criterion` for a discussion on the
realignment criterion].

If ``quspin`` has been installed, then the quantities given by
Eqs. :eq:`finite_chain_z_noise_ev_of_sx`-:eq:`finite_chain_z_noise_ev_of_energy`
will also be calculated by exact diagonalization for comparison sake.

Code
----

Below is the code that implements the simulation described above. You can also
find the same code in the file `examples/finite-chain/z-noise/example.py` of
the repository.

.. literalinclude:: ../examples/finite-chain/z-noise/example.py
