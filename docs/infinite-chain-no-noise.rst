Infinite chain subject to no noise
==================================

Introduction
------------

In this example, we implement a simulation of the dynamics of an infinite
ferromagnetic qubit chain subject to time-dependent transverse and longitudinal
fields, and no environmental noise. The simulation tracks the expectation value
of the system's energy per unit cell :math:`\left\langle
E\left(t\right)\right\rangle`, the first three correlation lengths [see the
documentation for the attribute
:attr:`spinbosonchain.state.SystemState.correlation_lengths` for a discussion on
correlation lengths], :math:`\left\langle
\hat{\sigma}_{x;0}\left(t\right)\right\rangle`, :math:`\left\langle
\hat{\sigma}_{y;0}\left(t\right) \hat{\sigma}_{z;1}\left(t\right)\right\rangle`,
and the probability of measuring :math:`\sigma_{z;r}\left(t\right)=+1` at sites
:math:`r=0` and :math:`r=1`, where :math:`r=0` is the center spin site of the
infinite chain. We also show how one can periodically backup the simulation data
to file in case of a crash and how to recover a simulation.

This example tests the algorithm implemented in the ``spinbosonchain`` library
that is used to calculate the dynamics of an infinite system subject to no
noise.

Hamiltonian
-----------

For background information on the generalized spin-boson chain model considered
in ``spinbosonchain``, see the documentation for the modules
:mod:`spinbosonchain.system` and :mod:`spinbosonchain.bath`.

In this example, we consider the following infinite model with no noise:

.. math ::
    \hat{H}\left(t\right)\equiv\sum_{u=-N}^{N}
    \hat{H}_{u}^{\left(A\right)}\left(t\right),
    :label: infinite_chain_no_noise_total_hamiltonian_1

where :math:`N\to\infty`, :math:`\hat{H}_{u}\left(t\right)` is the Hamiltonian
of the :math:`u^{\mathrm{th}}` 'unit cell' of the model, and
:math:`\hat{H}_{u}^{\left(A\right)}\left(t\right)` is the system Hamiltonian:

.. math ::
    \hat{H}_{u}^{\left(A\right)}\left(t\right)\equiv\sum_{r=0}^{L-1}
    \left\{ h_{x;r}\left(t\right)\hat{\sigma}_{x;r+uL}+h_{z;r}\left(t\right)
    \hat{\sigma}_{z;r+uL}\right\} +\sum_{r=0}^{L-2}J_{z,z;r,r+1}\left(t\right)
    \hat{\sigma}_{z;r+uL}\hat{\sigma}_{z;r+uL+1},
    :label: infinite_chain_no_noise_H_A

with :math:`L` being the number of qubits per 'unit cell':

.. math ::
    L=1,
    :label: infinite_chain_no_noise_L

:math:`h_{x;r}\left(t\right)` and :math:`h_{z;r}\left(t\right)` being the
transverse and longitudinal field energy scales respectively applied to site
:math:`r` at time :math:`t` :

.. math ::
    h_{x;r}\left(t\right)=-\frac{A\left(t\right)}{2},
    :label: infinite_chain_no_noise_h_x_r

.. math ::
    h_{z;r}\left(t\right)=\frac{B\left(t\right)}{2}h_{r},
    :label: infinite_chain_no_noise_h_z_r

.. math ::
    A\left(t\right)=2\left\{ 1-s\left(t\right)\right\} ,
    :label: infinite_chain_no_noise_A

.. math ::
    B\left(t\right)=2s\left(t\right),
    :label: infinite_chain_no_noise_B

.. math ::
    s\left(t\right)=\frac{t}{t_{a}},
    :label: infinite_chain_no_noise_normalized_anneal_fraction

.. math ::
    h_{0}=\frac{1}{5},
    :label: infinite_chain_no_noise_h_r

:math:`t_{a}` being the total simulation time,
:math:`J_{z,z;r,r+1}\left(t\right)` being the longitudinal coupling energy scale
between sites :math:`r` and :math:`r+1` at time :math:`t`:

.. math ::
    J_{z,z;r,r+1}\left(t\right)=\frac{B\left(t\right)}{2}J_{r,r+1},
    :label: infinite_chain_no_noise_J_zz

.. math ::
    J_{0,1}=-1,
    :label: infinite_chain_no_noise_J

and :math:`\hat{\sigma}_{\nu;r}` being the :math:`\nu^{\text{th}}` Pauli
operator acting on site :math:`r`.

We choose an infinite model with no environmental noise so that we can
alternatively calculate the dynamics using time evolving block decimation (TEBD)
method. By comparing the two approaches, we can verify the correctness of the
algorithm implemented in the ``spinbosonchain`` library that is used to
calculate the dynamics of a chain of qubits that is infinitely long. The
algorithms used to calculate the effects of :math:`J_{zz}`-coupling between
qubits are effectively independent of the algorithms used to calculate the
effects of environmental noise on the qubits. Therefore, while we de not
consider environmental noise in this example of an infinite chain, we have
tested extensively the algorithms used to calculate the effects of environmental
noise in the other examples included in this repository. As far as we know,
there are no practical examples of infinite chains subject to noise that we can
simulate using an alternative method that is essentially exact that we can
compare to the ``spinbosonchain`` library.

Initial state
-------------

We assume that the system and bath together are initially prepared in a state at
time :math:`t=0` corresponding to the state operator:

.. math ::
    \hat{\rho}^{\left(i\right)}=\hat{\rho}^{\left(i,A\right)}\otimes
    \hat{\rho}^{\left(i,B\right)},
    :label: infinite_chain_no_noise_rho_i

where :math:`\hat{\rho}^{\left(i,A\right)}` is the system's reduced state
operator at :math:`t=0`, and :math:`\hat{\rho}^{\left(i,B\right)}` is the bath's
reduced state operator at :math:`t=0`:

.. math ::
    \hat{\rho}^{\left(i,B\right)}\equiv
    \frac{e^{-\beta\hat{H}^{\left(B\right)}}}{\text{Tr}^{\left(B\right)}
    \left\{ e^{-\beta\hat{H}^{\left(B\right)}}\right\} },
    :label: infinite_chain_no_noise_rho_i_B

with :math:`\beta=1/\left(k_{B}T\right)`, :math:`k_{B}` being the Boltzmann
constant, :math:`T` being the temperature, and :math:`\text{Tr}_{B}\left\{
\cdots\right\}` being the partial trace with respect to the bath degrees of
freedom. In this example, we assume that the system is prepared in a state
corresponding to a state operator of the form:

.. math ::
    \hat{\rho}^{\left(i,A\right)}=\left|\Psi^{\left(i\right)}\right\rangle
    \left\langle \Psi^{\left(i\right)}\right|,
    :label: infinite_chain_no_noise_rho_i_A

where :math:`\left|\Psi^{\left(i\right)}\right\rangle`

.. math ::
    \left|\Psi^{\left(i\right)}\right\rangle =
    \bigotimes_{u=-N}^{N}\left[\bigotimes_{r=0}^{L-1}
    \left\{\left|+x\right\rangle _{r+uL}\right\}\right],
    :label: infinite_chain_no_noise_psi_i

with

.. math ::
    \hat{\sigma}_{\nu;r}\left|\pm\nu\right\rangle _{r}=
    \pm\left|\pm\nu\right\rangle _{r}.
    :label: infinite_chain_no_noise_defining_nu_eigenstates

Quantities tracked
------------------

The following quantities are tracked in the ``spinbosonchain`` simulation:

.. math ::
    \left\langle \hat{\sigma}_{x;r=0}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}
    \left(t\right)\hat{\sigma}_{x;r}\right\},
    :label: infinite_chain_no_noise_ev_of_sx

with :math:`\hat{\rho}^{\left(A\right)}\left(t\right)` being the system's
reduced state operator at time :math:`t`;

.. math ::
    \left\langle \hat{\sigma}_{y;0}\left(t\right)
    \hat{\sigma}_{z;1}\left(t\right)\right\rangle =
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \hat{\sigma}_{y;0}\hat{\sigma}_{z;1}\right\};
    :label: infinite_chain_no_noise_ev_of_sysz

.. math ::
    \left\langle E\left(t\right)\right\rangle =\text{Tr}^{\left(A\right)}
    \left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \hat{H}_{u=0}^{\left(A\right)}\left(t\right)\right\};
    :label: infinite_chain_no_noise_ev_of_energy

and

.. math ::
    \text{Prob}\left(\boldsymbol{\sigma}_{z}=+1;t\right)=
    \text{Tr}^{\left(A\right)}\left\{ \hat{\rho}^{\left(A\right)}\left(t\right)
    \bigotimes_{r=0}^{L-1}\left[\left|+z\right\rangle _{r}\left\langle
     +z\right|_{r}\right]\right\}.
    :label: infinite_chain_no_noise_probabilities

We also calculate the first three correlation lengths [see the documentation for
the attribute :attr:`spinbosonchain.state.SystemState.correlation_lengths` for a
discussion on correlation lengths].
	    
If ``tenpy`` has been installed, then the quantities given by
Eqs. :eq:`infinite_chain_no_noise_ev_of_sx`-:eq:`infinite_chain_no_noise_ev_of_energy`
will also be calculated by TEBD for comparison sake.

Code
----

Below is the code that implements the simulation described above. You can also
find the same code in the file `examples/infinite-chain/no-noise/example.py` of
the repository.

.. literalinclude:: ../examples/infinite-chain/no-noise/example.py
