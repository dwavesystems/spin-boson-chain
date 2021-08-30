Single qubit subject to z-noise
===============================

Introduction
------------

In this example, we implement a simulation of the dynamics of a single qubit
subject to a time-independent transverse field, and high-frequency ohmic noise
that is coupled to the :math:`z`-component of the qubit's spin. The
system-environment coupling energy scale is also time-independent. The
simulation tracks
:math:`\left\langle\hat{\sigma}_{x;r}\left(t\right)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{y;r}\left(t\right)\right\rangle`, and
:math:`\left\langle\hat{\sigma}_{z;r}\left(t\right)\right\rangle`.

This example tests the algorithm implemented in the ``spinbosonchain`` library
that is used to calculate local influence paths for the case where the system is
subject to :math:`z`-noise only, and the system's bath correlation time, or
"memory", is smaller than the total simulation time.

Hamiltonian
-----------

For background information on the generalized spin-boson chain model considered
in ``spinbosonchain``, see the documentation for the modules
:mod:`spinbosonchain.system` and :mod:`spinbosonchain.bath`.

In this example, we consider the following single-site model:

.. math ::
    \hat{H}=\hat{H}^{\left(A\right)}
    +\hat{H}^{\left(B\right)}+\hat{H}^{\left(AB\right)},
    :label: single_qubit_z_noise_total_hamiltonian

where :math:`\hat{H}^{\left(A\right)}\left(t\right)` is the system Hamiltonian:

.. math ::
    \hat{H}^{\left(A\right)}\equiv\sum_{r=0}^{L-1} h_{x;r}\hat{\sigma}_{x;r},
    :label: single_qubit_z_noise_H_A

with :math:`L` being the number of qubits:

.. math ::
    L=1,
    :label: single_qubit_z_noise_L

:math:`h_{x;r}\left(t\right)` being the transverse field energy scale applied to
site :math:`r`:

.. math ::
    h_{x;r}=-\frac{1}{2},
    :label: single_qubit_z_noise_h_x_r

and :math:`\hat{\sigma}_{\nu;r}` being the :math:`\nu^{\text{th}}` Pauli
operator acting on site :math:`r`; :math:`\hat{H}^{\left(B\right)}` is the bath
Hamiltonian, which describes a collection of decoupled harmonic oscillators:

.. math ::
    \hat{H}^{\left(B\right)}\equiv\sum_{\nu\in\left\{z\right\} }
    \sum_{r=0}^{L-1}\sum_{\epsilon}\omega_{\nu;\epsilon}
    \hat{b}_{\nu;r;\epsilon}^{\dagger}
    \hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}},
    :label: single_qubit_z_noise_H_B

with :math:`\epsilon` being the oscillator mode index,
:math:`\hat{b}_{\nu;r;\epsilon}^{\dagger}` and
:math:`\hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}}` being the bosonic creation
and annihilation operators respectively for the harmonic oscillator at site
:math:`r` in mode :math:`\epsilon` with angular frequency
:math:`\omega_{\nu;\epsilon}`, coupled to the :math:`\nu^{\text{th}}` component
of the spin at the same site; :math:`\hat{H}^{\left(AB\right)}` describes the
coupling between the system and the bath:

.. math ::
    \hat{H}^{\left(AB\right)}=-\sum_{\nu\in\left\{ z\right\} }
    \sum_{r=0}^{L-1}\hat{\sigma}_{\nu;r}\hat{\mathcal{Q}}_{\nu;r},
    :label: single_qubit_z_noise_H_AB

with :math:`\hat{\mathcal{Q}}_{\nu;r}` being the generalized reservoir force
operator at site :math:`r` that acts on the :math:`\nu^{\text{th}}` component of
the spin at the same site:

.. math ::
    \hat{\mathcal{Q}}_{\nu;r}=
    \mathcal{E}_{\nu;r}^{\left(\lambda\right)}
    \hat{Q}_{\nu;r},
    :label: single_qubit_z_noise_mathcal_Q

with :math:`\mathcal{E}_{v;r}^{\left(\lambda\right)}` being a time-independent
energy scale:

.. math ::
    \mathcal{E}_{z;r=0}^{\left(\lambda\right)}=1,
    :label: single_qubit_z_noise_mathcal_E_z

:math:`\hat{Q}_{\nu;r}` being a rescaled generalized reservoir force:

.. math ::
    \hat{Q}_{\nu;r}=-\sum_{\epsilon}\lambda_{\nu;r;\epsilon}
    \left\{ \hat{b}_{\nu;r;\epsilon}^{\dagger}
    +\hat{b}_{\nu;r;\epsilon}^{\vphantom{\dagger}}\right\},
    :label: single_qubit_z_noise_Q_v_r

:math:`\lambda_{\nu;r;\epsilon}` being the coupling strength between the
:math:`\nu^{\text{th}}` component of the spin at site :math:`r` and the harmonic
oscillator at the same site in mode :math:`\epsilon`.

Initial state
-------------

We assume that the system and bath together are initially prepared in a state at
time :math:`t=0` corresponding to the state operator:

.. math ::
    \hat{\rho}^{\left(i\right)}=\hat{\rho}^{\left(i,A\right)}\otimes
    \hat{\rho}^{\left(i,B\right)},
    :label: single_qubit_z_noise_rho_i

where :math:`\hat{\rho}^{\left(i,A\right)}` is the system's reduced state
operator at :math:`t=0`, and :math:`\hat{\rho}^{\left(i,B\right)}` is the bath's
reduced state operator at :math:`t=0`:

.. math ::
    \hat{\rho}^{\left(i,B\right)}\equiv
    \frac{e^{-\beta\hat{H}^{\left(B\right)}}}{\text{Tr}^{\left(B\right)}
    \left\{ e^{-\beta\hat{H}^{\left(B\right)}}\right\} },
    :label: single_qubit_z_noise_rho_i_B

with :math:`\beta=1/\left(k_{B}T\right)`, :math:`k_{B}` being the Boltzmann
constant, :math:`T` being the temperature, and :math:`\text{Tr}_{B}\left\{
\cdots\right\}` being the partial trace with respect to the bath degrees of
freedom. In this example, we assume that the system is prepared in a state
corresponding to a state operator of the form:

.. math ::
    \hat{\rho}^{\left(i,A\right)}=\left|\Psi^{\left(i\right)}\right\rangle
    \left\langle \Psi^{\left(i\right)}\right|,
    :label: single_qubit_z_noise_rho_i_A

where :math:`\left|\Psi^{\left(i\right)}\right\rangle`

.. math ::
    \left|\Psi^{\left(i\right)}\right\rangle =\bigotimes_{r=0}^{L-1}
    \left|+x\right\rangle _{r},
    :label: single_qubit_z_noise_psi_i

with

.. math ::
    \hat{\sigma}_{\nu;r}\left|\pm\nu\right\rangle _{r}=
    \pm\left|\pm\nu\right\rangle _{r}.
    :label: single_qubit_z_noise_defining_nu_eigenstates

Spectral densities of noise
---------------------------

To determine the noisy dynamics of the single-qubit system, it suffices to
specify the the "spin" part of the Hamiltonian :math:`\hat{H}^{\left(A\right)}`
Eq. :eq:`single_qubit_z_noise_H_A`, and the spectral density of noise at
temperature :math:`T` in the continuum limit in bosonic space:

.. math ::
    A_{\nu;r;T}\left(\omega\right)=\int_{-\infty}^{\infty}dt\,
    e^{i\omega t}\text{Tr}^{\left(B\right)}\left\{ \hat{\rho}^{\left(i,B\right)}
    \hat{Q}_{\nu;r}^{\left(B\right)}\left(t\right)
    \hat{Q}_{\nu;r}^{\left(B\right)}\left(0\right)\right\},
    :label: single_qubit_z_noise_introducing_spectral_densitities

where

.. math ::
    \hat{Q}_{z;r}^{\left(B\right)}\left(t\right)=e^{i\hat{H}^{\left(B\right)}t}
    \hat{Q}_{z;r}e^{-i\hat{H}^{\left(B\right)}t};
    :label: single_qubit_z_noise_Q_operator_in_heisenberg_picture

and

.. math ::
    \hat{\rho}^{\left(i,B\right)}\equiv
    \frac{e^{-\beta\hat{H}^{\left(B\right)}}}{\text{Tr}^{\left(B\right)}
    \left\{ e^{-\beta\hat{H}^{\left(B\right)}}\right\} },
    :label: single_qubit_z_noise_initial_state_operator

with :math:`\beta=1/\left(k_{B}T\right)`:

.. math ::
    \beta = 5 \times 10^{-3},
    :label: single_qubit_z_noise_beta

and :math:`k_{B}` being the Boltzmann constant. As discussed in the
documentation of the class :class:`spinbosonchain.bath.SpectralDensity`, we
express :math:`A_{\nu;r;T}\left(\omega\right)` as

.. math ::
    A_{\nu;r;T}\left(\omega\right)=\text{sign}\left(\omega\right)
    \frac{A_{\nu;r;T=0}\left(\left|\omega\right|\right)}{1-e^{-\beta\omega}},
    :label: single_qubit_z_noise_A_v_r_T

where :math:`A_{\nu;r;T=0}\left(\omega\right)` is the zero-temperature limit:

.. math ::
    A_{\nu;r;T=0}\left(\omega\right)=
    A_{\nu;r;T=0;\varsigma=0}\left(\omega\right),
    :label: single_qubit_z_noise_A_v_r_0T

with :math:`A_{\nu;r;T=0;\varsigma}\left(\omega\right)` being the
:math:`\varsigma^{\text{th}}`-component of
:math:`A_{\nu;r;T=0}\left(\omega\right)`. In this example, the spectral density
of noise :math:`A_{z;r;T}\left(\omega\right)` trivially has one component, which
we assume to be ohmic:

.. math ::
    A_{z;r=0;T=0;\varsigma}\left(\omega\right)=
    \left\{ \Theta\left(\omega\right)
    -\Theta\left(\omega-\omega^{\left(\text{UV}\right)}\right)\right\}
    \eta\omega e^{-\omega/\omega_{c}},
    :label: single_qubit_z_noise_A_v_r_0T_varsigma

where :math:`\Theta\left(\omega\right)` is the Heaviside step funcion,
:math:`\eta` is a system-bath coupling constant:

.. math ::
    \eta=6.11\times10^{-4},
    :label: single_qubit_z_noise_eta

:math:`\omega_{c}` is a soft frequency cutoff:

.. math ::
    \omega_{c}=2\times10^{3},
    :label: single_qubit_z_noise_soft_frequency_cutoff

and :math:`\omega^{\left(\text{UV}\right)}` is a hard frequency cutoff:

.. math ::
    \omega^{\left(\text{UV}\right)}=40\omega_{c}.
    :label: single_qubit_z_noise_UV_cutoff

Polarization vector and pointer basis
-------------------------------------

The system's reduced state operator
:math:`\hat{\rho}^{\left(A\right)}\left(t\right)` can be expressed as
[see e.g. Ref. [Ballentine1]_]:

.. math ::
    \hat{\rho}^{\left(A\right)}\left(t\right)=
    \frac{1}{2}\left(\hat{1}+\hat{\sigma}_{\mathbf{a}\left(t\right);r=0}\right),
    :label: single_qubit_z_noise_rho_A_in_terms_of_polarization

where :math:`\mathbf{a}\left(t\right)` is the polarization vector:

.. math ::
    \mathbf{a}\left(t\right)=\sum_{i=x,y,z}a_{i}\left(t\right)\mathbf{e}_{i}.
    :label: single_qubit_z_noise_polarization_vector

with :math:`\left\{ \mathbf{e}_{i}\right\} _{i=x,y,z}` being the standard basis
of :math:`\mathbb{R}^{3}`, and

.. math ::
    a_{i=x,y,z}\left(t\right)=\left\langle
    \hat{\sigma}_{i;r=0}\left(t\right)\right\rangle;
    :label: single_qubit_z_noise_a_i

.. math ::
    \hat{\sigma}_{\mathbf{v};r}=\mathbf{v}\cdot\hat{\boldsymbol{\sigma}}_{r},
    :label: single_qubit_z_noise_sigma_v_0

with

.. math ::
    \hat{\boldsymbol{\sigma}}_{r}=\hat{\sigma}_{x;r}\mathbf{e}_{x}
    +\hat{\sigma}_{y;r}\mathbf{e}_{y}+\hat{\sigma}_{z;r}\mathbf{e}_{z}.
    :label: single_qubit_z_noise_sigma_0

If we denote the eigenstates of :math:`\hat{\sigma}_{\mathbf{v};r}` by
:math:`\left|\pm\mathbf{v}\right\rangle _{r}` such that:

.. math ::
    \hat{\sigma}_{\mathbf{v};r}\left|\pm\mathbf{v}\right\rangle _{r}
    =\pm\left|\pm\mathbf{v}\right\rangle _{r},
    :label: single_qubit_z_noise_eigenstates_of_sigma_v_r

then :math:`\left|\pm\mathbf{a}\left(t\right)\right\rangle` is the instantaneous
pointer basis. Hence, the polarization vector :math:`\mathbf{a}\left(t\right)`
is a convenient way to parameterize the pointer basis, which itself can be
expressed in spherical coordinates:

.. math ::
    r_{\mathbf{a}\left(t\right)}=\left|\mathbf{a}\left(t\right)\right|,
    :label: single_qubit_z_noise_r_a

.. math ::
    \theta_{\mathbf{a}\left(t\right)}=
    \text{atan2}\left(\sqrt{a_{x}^{2}\left(t\right)+a_{y}^{2}\left(t\right)},
    a_{z}\left(t\right)\right),
    :label: single_qubit_z_noise_theta_a

.. math ::
    \varphi_{\mathbf{a}\left(t\right)}=
    \text{atan2}\left(a_{y}\left(t\right),a_{x}\left(t\right)\right),
    :label: single_qubit_z_noise_varphi_a

where atan2 is the 2-argument arctangent.

Quantities tracked
------------------

The following quantities are tracked in the ``spinbosonchain`` simulation:

.. math ::
    \left\langle \hat{\sigma}_{x;r}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}
    \left(t\right)\hat{\sigma}_{x;r}\right\},
    :label: single_qubit_z_noise_ev_of_sx

.. math ::
    \left\langle \hat{\sigma}_{y;r}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}
    \left(t\right)\hat{\sigma}_{y;r}\right\},
    :label: single_qubit_z_noise_ev_of_sy

and

.. math ::
    \left\langle \hat{\sigma}_{z;r}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \hat{\sigma}_{z;r}\right\}.
    :label: single_qubit_z_noise_ev_of_sz

As discussed in the previous section, we can see that by tracking
:math:`\left\langle \hat{\sigma}_{x;r=0}\left(t\right)\right\rangle`,
:math:`\left\langle \hat{\sigma}_{y;r=0}\left(t\right)\right\rangle`, and
:math:`\left\langle \hat{\sigma}_{z;r=0}\left(t\right)\right\rangle`, we are
tracking all the information encoded in
:math:`\hat{\rho}^{\left(A\right)}\left(t\right)`.

Expected result
---------------

Because we are considering high frequency ohmic noise with :math:`\omega_{c}\gg
h_{x;r=0}`, the dynamics are expected to be Markovian, i.e. that the dynamics of
the present time do not depend on the system's history. Moreover, the parameters
:math:`\eta`, :math:`\omega_{c}`, and :math:`\beta` have been chosen such that
the system is strongly coupled to its environment.  Using the Lindblad formalism
Albash and Lidar [see Ref. [Albash1]_] have shown that for such strong Markovian
noise, the spherical coordinates of the polarization vector satisfy:

.. math ::
    r_{\mathbf{a}\left(t\right)}=e^{-t/T_{2}^{\left(c\right)}},
    :label: single_qubit_z_noise_r_a_expected_result

.. math ::
    \theta_{\mathbf{a}\left(t\right)}=\frac{\pi}{2},
    :label: single_qubit_z_noise_theta_a_expected_result

.. math ::
    \varphi_{\mathbf{a}\left(t\right)}=0,
    :label: single_qubit_z_noise_varphi_a_expected_result

where :math:`T_{2}^{\left(c\right)}` is the dephasing time in the computational
basis:

.. math ::
    T_{2}^{\left(c\right)}=\frac{1}{2A_{z;r=0;T}\left(0^{+}\right)}.
    :label: single_qubit_z_noise_CB_thermal_relaxation_time

Code
----

Below is the code that implements the simulation described above. You can also
find the same code in the file `examples/single-qubit/z-noise/example.py` of
the repository.

.. literalinclude:: ../examples/single-qubit/z-noise/example.py
