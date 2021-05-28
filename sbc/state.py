#!/usr/bin/env python
r"""For calculating the system's state and various related state properties.

``sbc`` is a library for simulating the dynamics of a generalized
one-dimensional spin-boson chain model, where both the :math:`z`- and 
:math:`y`-components of the spins are coupled to bosonic baths, rather than 
only the :math:`z`-components. A convenient way to discuss both finite and
infinite chains is to express the Hamiltonian of the aforementioned spin-boson
model as a sum of :math:`2N+1` 'unit cell' Hamiltonians:

.. math ::
    \hat{H}\left(t\right)\equiv\sum_{u=-N}^{N}\hat{H}_{u}\left(t\right),
    :label: state_total_Hamiltonian

where :math:`N` is a non-negative integer, and :math:`\hat{H}_{u}\left(t\right)`
is the Hamiltonian of the :math:`u^{\mathrm{th}}` 'unit cell' of the model:

.. math ::
    \hat{H}_{u}\left(t\right)=\hat{H}_{u}^{\left(A\right)}\left(t\right)
    +\hat{H}_{u}^{\left(B\right)}+\hat{H}_{u}^{\left(AB\right)},
    :label: state_unit_cell_Hamiltonian

with :math:`\hat{H}_{u}^{\left(A\right)}\left(t\right)` being the system part of
:math:`\hat{H}_{u}\left(t\right)`, which encodes all information regarding
energies associated exclusively with the spins;
:math:`\hat{H}_{u}^{\left(B\right)}` being the bath part of
:math:`\hat{H}_{u}\left(t\right)`, which encodes all information regarding
energies associated with the components of the bosonic environment; and
:math:`\hat{H}_{u}^{\left(AB\right)}` is the system-bath coupling part of
:math:`\hat{H}_{u}\left(t\right)`, which describes all energies associated with
the coupling between the system and the environment.

For finite chains, we set :math:`N=0`, whereas for infinite chains, we take the 
limit of :math:`N\to\infty`.

The full state operator at time :math:`t` can be expressed as:

.. math ::
    \hat{\rho}(t) = \hat{U}(t, 0) \hat{\rho}^{(i)} \hat{U}(0, t),
    :label: state_full_state_operator

where :math:`\hat{\rho}^{(i)}` is the state operator corresponding to the 
initial state of the system at time :math:`t=0`; and 
:math:`\hat{U}\left(t, t^{\prime}\right)` is the evolution operator:

.. math ::
    \hat{U}\left(t,t^{\prime}\right) & \equiv\begin{cases}
    T\left\{ e^{-i\int_{t^{\prime}}^{t}dt^{\prime\prime}\hat{H}
    \left(t^{\prime\prime}\right)}\right\} , & \text{if }t\ge t^{\prime},\\
    \tilde{T}\left\{ e^{i\int_{t}^{t^{\prime}}dt^{\prime\prime}\hat{H}
    \left(t^{\prime\prime}\right)}\right\} , & \text{if }t<t^{\prime},
    \end{cases}\nonumber \\
     & \equiv\begin{cases}
    \sum_{n=0}^{\infty}\frac{\left(-i\right)^{n}}{n!}\prod_{m=1}^{n}
    \left\{ \int_{t^{\prime}}^{t}dt^{\left(m\right)}\right\} T
    \left\{ \prod_{m=1}^{n}\left[\hat{H}\left(t^{\left(m\right)}\right)\right]
    \right\} , & \text{if }t\ge t^{\prime},\\
    \sum_{n=0}^{\infty}\frac{\left(i\right)^{n}}{n!}\prod_{m=1}^{n}
    \left\{ \int_{t}^{t^{\prime}}dt^{\left(m\right)}\right\} \tilde{T}
    \left\{ \prod_{m=1}^{n}\left[\hat{H}\left(t^{\left(m\right)}\right)\right]
    \right\} , & \text{if }t<t^{\prime},
    \end{cases}
    :label: state_evolution_operator

with :math:`T\left\{\cdots\right\}` is the time ordering symbol, which specifies
that the string of time-dependent operators contained within 
:math:`T\left\{\cdots\right\}` be rearranged as a time-descending sequence, and 
:math:`\tilde{T}\left\{\cdots\right\}` is the anti-time ordering symbol, which
orders strings of time-dependent operators in the reverse order to that
specified by :math:`T\left\{\cdots\right\}`.

For all simulations in ``sbc``, :math:`\hat{\rho}^{(i)}` is assumed to be of the
form:

.. math ::
    \hat{\rho}^{(i)} = \hat{\rho}^{(i, A)} \otimes \hat{\rho}^{(i, B)},
    :label: state_initial_state_operator

where :math:`\hat{\rho}^{(i, A)}` is the system's reduced state operator at
time :math:`t=0`:

.. math ::
    \hat{\rho}^{(i,A)}\equiv\left|\Psi^{(i,A)}\right\rangle 
    \left\langle \Psi^{(i,A)}\right|,
    :label: state_initial_state_operator_A

with :math:`\left|\Psi^{(i,A)}\right\rangle` being a pure state; 
:math:`\hat{\rho}^{(i,B)}` is the bath's reduced state operator at time 
:math`t=0`:

.. math ::
    \hat{\rho}^{(i,B)}\equiv\frac{e^{-\beta\hat{H}^{(B)}}}{\mathcal{Z}^{(B)}},
    :label: state_initial_state_operator_B

with :math:`\beta=1/\left(k_{B}T\right)`, :math:`k_{B}` being the Boltzmann
constant, :math:`T` being the temperature, :math:`\mathcal{Z}^{(B)}`
being the partition function of the bath at :math:`t=0`:

.. math ::
    \mathcal{Z}^{(B)}=\text{Tr}^{(B)}\left\{ e^{-\beta\hat{H}^{(B)}}\right\},
    :label: state_bath_partition_function

and :math:`\text{Tr}^{(B)}\left\{ \cdots\right\}` being the partial trace with
respect to the bath degrees of freedom.

The system's reduced state operator at time :math:`t` is given by:

.. math ::
    \hat{\rho}^{(A)}(t)=\text{Tr}^{(B)}\left\{ \hat{\rho}(t)\right\}.
    :label: state_system_reduced_state_operator

``sbc`` adopts the quasi-adiabatic path integral (QUAPI) formalism to express
the spin system's reduced density matrix/operator as a time-discretized path
integral, comprising of a series of influence functionals that encode the
non-Markovian dynamics of the system. The path integral is decomposed into a
series of components that can be represented by tensor networks. 

This module contains a class which represents the system's reduced density
matrix, which contains a method for evolving the system's state. Moreover, this
module contains functions that calculate various properties of the system's 
state.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# Import a few math functions.
from math import ceil

# For explicitly releasing memory.
import gc



# For creating arrays to be used to construct tensor nodes and networks.
import numpy as np

# For calculating the eigenspectra of complex square matrices.
import scipy.linalg

# For creating tensor networks and performing contractions.
import tensornetwork as tn



# For creating nodes relating to the phase factors in the QUAPI path integral.
from sbc._phasefactor import tensorfactory

# For creating influence paths/functionals.
from sbc import _influence

# For performing SVD truncation sweeps.
from sbc import _svd



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Non-Production"



##############################################
## Define classes, functions, and instances ##
##############################################

# List of public objects in objects.
__all__ = ["SystemState",
           "trace",
           "schmidt_spectrum_sum",
           "realignment_criterion",
           "spin_config_prob"]



class SystemState():
    r"""The system's reduced density matrix.

    The documentation for this class makes reference to the concept of a 
    'system', 'bath', and 'unit cell', which is introduced in the documentation 
    for the module :mod:`sbc.state`.

    In addition to representing the system's state, this class also encodes the
    dynamics of the system.

    Parameters
    ----------
    system_model : :class:`sbc.system.Model`
        The system's model parameter set.
    bath_model : :class:`sbc.bath.Model`
        The bath's model components.
    alg_params : :class:`sbc.alg.Params`
        The simulation parameters relating to the tensor network algorithm.
    initial_state_nodes : `array_like` (:class:`tensornetwork.Node`, shape=(``L``,))
        The nodes making up the MPS that represents the initial state of the 
        :math:`u=0` unit cell of the system. Note that in the case of a 
        finite chain there is only one unit cell (i.e. the :math:`u=0` unit 
        cell), whereas for an infinite chain there is an arbitrarily large 
        number of unit cells. The MPS can represent a pure state vector, or a 
        density matrix of the equivalent state. Should the MPS represent a pure 
        state vector, then the ``r`` th node should have the shape 
        ``(chi[r], 2, chi[r+1])`` where ``chi[0]=chi[L]=1`` for finite chains 
        and ``chi[0<r<L]>=1`` for both finite and infinite chains. In ``sbc``, 
        states are expressed in the eigenbasis of :math:`\hat{\sigma}_{z}`:
        
        .. math ::
            \hat{\sigma}_{z} \left|\sigma_z z\right\rangle = 
            \sigma_{z} \left|\sigma_z z\right\rangle

        where the :math:`\left|\sigma_z z\right\rangle` are the eigenstates with
        :math:`\sigma_z=\pm 1`. If we denote the physical index, i.e. the 
        middle tensor index, of any given node by ``q``, then ``q=0`` 
        corresponds to :math:`\sigma_z=1` and ``q=1`` corresponds to
        :math:`\sigma_z=-1`. Should the MPS represent a density matrix, then 
        the ``r`` th node should have the shape ``(chi[r], 4, chi[r+1])``, 
        where ``chi[0]=chi[L]=1`` and ``chi[0<r<L]>=1``. Denoting the physical
        index by ``j``: ``j=0`` corresponds to the :math:`\sigma_z`-pair
        :math:`(1, 1)`; ``j=1`` corresponds to :math:`(1, -1)`; ``j=2``
        corresponds to :math:`(-1, 1)`; and ``j=3`` corresponds to 
        :math:`(-1, -1)`.

    Attributes
    ----------
    system_model : :class:`sbc.system.Model`, read-only
        The system's model parameter set.
    bath_model : :class:`sbc.bath.Model`, read-only
        The bath's model components.
    alg_params : :class:`sbc.alg.Params`, read-only
        The simulation parameters relating to the tensor network algorithm.
    t : `float`, read-only
        The current time in the simulation.
    nodes : `array_like` (:class:`tensornetwork.Node`, shape=(``L``,))
        The nodes making up the MPS that represents the current state of the 
        :math:`u=0` unit cell of the system (i.e. the state at time ``t=n*dt``).
    correlation_length : `None` | `float`, read-only
        Set to `None` for finite chains, otherwise ``correlation_length`` is the
        correlation length of the system.
    """
    def __init__(self,
                 system_model,
                 bath_model,
                 alg_params,
                 initial_state_nodes):
        if system_model.L != bath_model.L:
            raise ValueError(_system_state_init_err_msg_1)

        self._n = 0  # Time step index.
        self.t = 0.0
        self.system_model = system_model
        self.bath_model = bath_model
        self.alg_params = alg_params
        self._num_bonds = len(system_model.zz_couplers)
        
        self._set_nodes_from_initial_state_nodes(initial_state_nodes)
        self._Xi_rho_vdash = self.nodes
        self._initialize_influence_paths()

        dt = alg_params.dt
        self._z_field_phase_factor_node_rank_2_factory = \
            tensorfactory.ZFieldPhaseFactorNodeRank2(system_model, dt)
        self._zz_coupler_phase_factor_node_rank_2_factory = \
            tensorfactory.ZZCouplerPhaseFactorNodeRank2(system_model, dt)

        dt = alg_params.dt
        tau = bath_model.memory
        K_tau = max(0, ceil((tau - 7.0*dt/4.0) / dt)) + 3
        self._max_k_in_first_iteration_procedure = lambda n: (n-K_tau)-1
        self._max_k_in_second_iteration_procedure = lambda n: n

        self._alg = \
            "yz-noise" if bath_model.y_spectral_densities != None else "z-noise"

        self._trace = None  # For caching the instantaneous trace.
        
        if self.system_model.is_infinite:
            self._update_infinite_chain_alg_attrs()
        else:
            self.correlation_length = None

        return None



    def _set_nodes_from_initial_state_nodes(self, initial_state_nodes):
        num_nodes = len(initial_state_nodes)
        L = self.system_model.L
        
        if num_nodes != L:
            msg = _system_state_set_nodes_from_initial_state_nodes_err_msg_1
            raise ValueError(msg)

        d = initial_state_nodes[0].shape[1]
        if (d != 2) and (d != 4):
            msg = _system_state_set_nodes_from_initial_state_nodes_err_msg_2
            raise ValueError(msg)

        rho_nodes = []
        for node in initial_state_nodes:
            if (node.shape[1] != d) or (len(node.shape) != 3):
                msg = _system_state_set_nodes_from_initial_state_nodes_err_msg_3
                raise ValueError(msg)

            
            if node.shape[1] == 4:
                new_node = node.copy()
            elif node.shape[1] == 2:
                new_node = tn.outer_product(node.copy(), tn.conj(node))
                tn.flatten_edges([new_node[0], new_node[3]])
                tn.flatten_edges([new_node[0], new_node[2]])
                tn.flatten_edges([new_node[0], new_node[1]])
                
            rho_nodes.append(new_node)

        is_infinite = self.system_model.is_infinite
        state_trunc_params = self.alg_params.state_trunc_params
        _svd.left_to_right_svd_sweep_across_mps(rho_nodes,
                                                state_trunc_params,
                                                is_infinite)
        self._schmidt_spectrum = \
            _svd.right_to_left_svd_sweep_across_mps(rho_nodes,
                                                    state_trunc_params,
                                                    is_infinite)
        self.nodes = rho_nodes

        return None



    def _initialize_influence_paths(self):
        self._map_btwn_site_indices_and_unique_influence_paths = \
            self._calc_map_btwn_site_indices_and_unique_influence_paths()
        site_indices_of_unique_influence_paths = \
            set(self._map_btwn_site_indices_and_unique_influence_paths)
        
        system_model = self.system_model
        bath_model = self.bath_model
        L = system_model.L
        dt = self.alg_params.dt
        influence_trunc_params = self.alg_params.influence_trunc_params

        self._unique_influence_paths = \
            {r: _influence.path.Path(r, system_model, bath_model,
                                     dt, influence_trunc_params)
             for r in site_indices_of_unique_influence_paths}
        
        self._influence_paths = [None]*L
        for idx in range(L):
            r = self._map_btwn_site_indices_and_unique_influence_paths[idx]
            self._influence_paths[idx] = self._unique_influence_paths[r]
        
        return None                      



    def _calc_map_btwn_site_indices_and_unique_influence_paths(self):
        system_model = self.system_model
        bath_model = self.bath_model
        
        _map_btwn_site_indices_and_unique_x_fields = \
            system_model._map_btwn_site_indices_and_unique_x_fields
        _map_btwn_site_indices_and_unique_local_bath_model_cmpnt_sets = \
            bath_model._map_btwn_site_indices_and_unique_local_model_cmpnt_sets

        zip_obj = \
            zip(_map_btwn_site_indices_and_unique_x_fields,
                _map_btwn_site_indices_and_unique_local_bath_model_cmpnt_sets)
        pairs = list(zip_obj)

        L = system_model.L
        result = list(range(L))
        for idx1 in range(L):
            for idx2 in range(idx1+1, L):
                if pairs[idx2] == pairs[idx1]:
                    result[idx2] = result[idx1]

        return result



    def evolve(self, num_steps=1):
        r"""Evolve the system state by a given number of time steps.

        Parameters
        ----------
        num_steps : `int`, optional
            The number of times to step-evolve the system state.

        Returns
        -------
        """
        if num_steps < 0:
            raise ValueError(_system_state_evolve_err_msg_1)
        if num_steps == 0:
            return None

        for r, influence_path in self._unique_influence_paths.items():
            influence_path.evolve(num_n_steps=num_steps)

        self._k = max(-1, self._max_k_in_first_iteration_procedure(self._n)+1)
        self._n += num_steps
        self.t += num_steps * self.alg_params.dt
            
        while self._k <= self._max_k_in_first_iteration_procedure(self._n):
            self._k_step()
            gc.collect()

        self._Xi_rho = self._Xi_rho_vdash[:]  # Shallow copy.
        while self._k <= self._max_k_in_second_iteration_procedure(self._n):
            self._k_step()
            gc.collect()

        self.nodes = self._Xi_rho

        if self.system_model.is_infinite:
            self._update_infinite_chain_alg_attrs()

        self._trace = None  # Needs to be recalculated upon evolving state.

        return None



    def _k_step(self):
        k = self._k
        n = self._n
        L = self.system_model.L
        num_couplers = len(self.system_model.zz_couplers)
        state_trunc_params = self.alg_params.state_trunc_params
        is_infinite = self.system_model.is_infinite

        print("n={}; k={}".format(n, k))

        if k <= self._max_k_in_first_iteration_procedure(n):
            rho_nodes = self._Xi_rho_vdash
        else:
            rho_nodes = self._Xi_rho

        restructured_zz_coupler_phase_factor_nodes = \
            self._build_split_and_restructure_zz_coupler_phase_factor_nodes()

        for r in range(0, L):
            self._update_rho_node(r,
                                  rho_nodes,
                                  restructured_zz_coupler_phase_factor_nodes)

        if k < self._max_k_in_second_iteration_procedure(n):
            one_legged_node = tn.Node(np.ones([4]))
            network_struct = [(-1, -2, 1, -3), (1,)]
        else:
            one_legged_node = tn.Node(np.ones([1]))
            network_struct = [(-1, 1, -2, -3), (1,)]
        for r in range(0, L):
            nodes_to_contract = [rho_nodes[r], one_legged_node]
            rho_nodes[r] = tn.ncon(nodes_to_contract, network_struct)

        _svd.left_to_right_svd_sweep_across_mps(rho_nodes,
                                                state_trunc_params,
                                                is_infinite)
        self._schmidt_spectrum = \
            _svd.right_to_left_svd_sweep_across_mps(rho_nodes,
                                                    state_trunc_params,
                                                    is_infinite)

        if k <= self._max_k_in_first_iteration_procedure(n):
            self._Xi_rho_vdash = rho_nodes
            beg = 1 if (k == -1) or (self._alg == "z-noise") else 3
            for r, influence_path in self._unique_influence_paths.items():
                influence_path.Xi_I_1_1_nodes = \
                    influence_path.Xi_I_1_1_nodes[beg:]
        else:
            self._Xi_rho = rho_nodes

        self._k += 1

        return None



    def _build_split_and_restructure_zz_coupler_phase_factor_nodes(self):
        k = self._k
        n = self._n
        L = self.system_model.L
        num_couplers = len(self.system_model.zz_couplers)
        zz_coupler_phase_factor_node_rank_2_factory = \
            self._zz_coupler_phase_factor_node_rank_2_factory

        split_zz_coupler_phase_factor_nodes = [None] * (2*L)
        for r in range(num_couplers):
            node = zz_coupler_phase_factor_node_rank_2_factory.build(r, k+1, n)
            left_node, right_node = _svd.split_node(node=node,
                                                    left_edges=(node[0],),
                                                    right_edges=(node[1],))
            split_zz_coupler_phase_factor_nodes[(2*r+1)%(2*L)] = left_node
            split_zz_coupler_phase_factor_nodes[(2*r+2)%(2*L)] = right_node

        restructured_zz_coupler_phase_factor_nodes = []
        for r in range(L):
            node_1 = split_zz_coupler_phase_factor_nodes[2*r]
            node_3 = split_zz_coupler_phase_factor_nodes[2*r+1]
            if (node_1 is None) and (node_3 is None):  # L=1; finite chain.
                tensor = np.zeros([1, 4, 4, 1], dtype=np.complex128)
                for idx in range(4):
                    tensor[0, idx, idx, 0] = 1
                node = tn.Node(tensor)
                restructured_zz_coupler_phase_factor_nodes.append(node)
                break
            if (node_1 is None) and (node_3 is not None):
                tensor = np.zeros([1, 4, 4, 4], dtype=np.complex128)
                for idx in range(4):
                    tensor[0, idx, idx, idx] = 1
                node_2 = tn.Node(tensor)
                nodes_to_contract = [node_2, node_3]
                network_struct = [(-1, -2, -3, 1), (1, -4)]
            elif (node_1 is not None) and (node_3 is None):
                tensor = np.zeros([4, 4, 4, 1], dtype=np.complex128)
                for idx in range(4):
                    tensor[idx, idx, idx, 0] = 1
                node_2 = tn.Node(tensor)
                nodes_to_contract = [node_1, node_2]
                network_struct = [(-1, 1), (1, -2, -3, -4)]
            else:
                tensor = np.zeros([4, 4, 4, 4], dtype=np.complex128)
                for idx in range(4):
                    tensor[idx, idx, idx, idx] = 1
                node_2 = tn.Node(tensor)
                nodes_to_contract = [node_1, node_2, node_3]
                network_struct = [(-1, 1), (1, -2, -3, 2), (2, -4)]

            node = tn.ncon(nodes_to_contract, network_struct)
            restructured_zz_coupler_phase_factor_nodes.append(node)

        return restructured_zz_coupler_phase_factor_nodes



    def _update_rho_node(self,
                         r,
                         rho_nodes,
                         restructured_zz_coupler_phase_factor_nodes):
        k = self._k
        n = self._n
        L = self.system_model.L

        influence_nodes = self._get_influence_nodes(r, k, n)
        z_field_phase_factor_node = \
            self._z_field_phase_factor_node_rank_2_factory.build(r, k+1, n)

        if (k != -1) and (self._alg == "yz-noise"):
            j_node_1 = tn.Node(np.ones([4]))
            j_node_2 = tn.Node(np.ones([4]))
            nodes_to_contract = [rho_nodes[r],
                                 influence_nodes[0],
                                 j_node_1,
                                 influence_nodes[1],
                                 j_node_2,
                                 influence_nodes[2],
                                 z_field_phase_factor_node,
                                 restructured_zz_coupler_phase_factor_nodes[r]]
            network_struct = [(-2, 6, -6),
                              (6, 1, 4),
                              (1,),
                              (4, 2, 5),
                              (2,),
                              (5, 3, -3),
                              (3, 7),
                              (-1, -4, 7, -5)]
        else:
            nodes_to_contract = [rho_nodes[r],
                                 influence_nodes[0],
                                 z_field_phase_factor_node,
                                 restructured_zz_coupler_phase_factor_nodes[r]]
            network_struct = [(-2, 2, -6),
                              (2, 1, -3),
                              (1, 3),
                              (-1,-4, 3, -5)]

        updated_rho_node = tn.ncon(nodes_to_contract, network_struct)
        tn.flatten_edges([updated_rho_node[0], updated_rho_node[1]])
        tn.flatten_edges([updated_rho_node[2], updated_rho_node[3]])
        updated_rho_node.reorder_edges([updated_rho_node[2],
                                        updated_rho_node[0],
                                        updated_rho_node[1],
                                        updated_rho_node[3]])
        rho_nodes[r] = updated_rho_node

        return None
        


    def _get_influence_nodes(self, r, k, n):
        if k <= self._max_k_in_first_iteration_procedure(n):
            Xi_I_1_1_nodes = self._influence_paths[r].Xi_I_1_1_nodes
            beg = 0
            end = 1 if (k == -1) or (self._alg == "z-noise") else 3
            influence_nodes = Xi_I_1_1_nodes[beg:end]
        else:
            Xi_I_dashv_nodes = self._influence_paths[r].Xi_I_dashv_nodes
            num_nodes = len(Xi_I_dashv_nodes)
            end = (num_nodes - (n+1-k) + 1
                   if self._alg == "z-noise"
                   else num_nodes - 3*(n+1-k) + 2*(num_nodes%3) + 1)
            beg = end-1 if (k == -1) or (self._alg == "z-noise") else end-3
            influence_nodes = Xi_I_dashv_nodes[beg:end]

        return influence_nodes



    def _update_infinite_chain_alg_attrs(self):
        self._update_transfer_matrix()
        w, vl, vr = scipy.linalg.eig(self._transfer_matrix, left=True)

        dominant_eigval_idx = np.argmax(np.abs(w))
        self._dominant_eigval = w[dominant_eigval_idx]
        
        left_eigvec = vl[:, dominant_eigval_idx]
        right_eigvec = vr[:, dominant_eigval_idx]
        norm_const = np.sqrt(np.vdot(left_eigvec, right_eigvec)+0j)
        
        self._dominant_left_eigvec_node = \
            tn.Node(np.conj(left_eigvec) / norm_const)
        self._dominant_right_eigvec_node = \
            tn.Node(right_eigvec / norm_const)

        L = self.system_model.L
        if len(w) > 1:
            self.correlation_length = \
                -L / np.log(np.sort(np.abs(w / self._dominant_eigval))[-2])
        else:
            self.correlation_length = 0

        return None



    def _update_transfer_matrix(self):
        L = self.system_model.L
        
        tensor = np.array([1, 0, 0, 1], dtype=np.complex128)
        physical_1_legged_node = tn.Node(tensor)

        nodes_to_contract = [self.nodes[0], physical_1_legged_node]
        network_struct = [(-1, 1, -2), (1,)]
        result = tn.ncon(nodes_to_contract, network_struct)

        for i in range(1, L):
            nodes_to_contract = [result, self.nodes[i], physical_1_legged_node]
            network_struct = [(-1, 2), (2, 1, -2), (1,)]
            result = tn.ncon(nodes_to_contract, network_struct)

        self._transfer_matrix = np.array(result.tensor)

        return None



def _apply_1_legged_nodes_to_system_state_mps(physical_1_legged_nodes,
                                              system_state):
    r"""Used in some of the public functions in this module and elsewhere."""
    L = system_state.system_model.L

    if system_state.system_model.is_infinite:
        left_1_legged_node = system_state._dominant_left_eigvec_node
        right_1_legged_node = system_state._dominant_right_eigvec_node
        num_unit_cells_required = len(physical_1_legged_nodes) // L
        scale_factor = np.power(system_state._dominant_eigval,
                                num_unit_cells_required)
    else:
        left_1_legged_node = tn.Node(np.array([1], dtype=np.complex128))
        right_1_legged_node = tn.Node(np.array([1], dtype=np.complex128))
        scale_factor = 1
    
    result = left_1_legged_node

    for idx, physical_1_legged_node in enumerate(physical_1_legged_nodes):
        nodes_to_contract = [result,
                             physical_1_legged_node,
                             system_state.nodes[idx % L]]
        network_struct = [(1,), (2,), (1, 2, -1)]
        result = tn.ncon(nodes_to_contract, network_struct)
        
    nodes_to_contract = [result, right_1_legged_node]
    network_struct = [(1,), (1,)]
    result = tn.ncon(nodes_to_contract, network_struct)

    result = np.array(result.tensor) / scale_factor
    
    return result



def trace(system_state):
    r"""Evaluate the trace of the system's reduced density matrix.

    The QUAPI algorithm used in the ``sbc`` library does not preserve the
    unitarity of the time evolution of the system state. As a result, the trace 
    of the system's reduced density matrix that is simulated is not necessarily 
    unity. Strictly speaking, this point only applies to finite chains: for 
    infinite chains, the algorithm explicitly renormalizes the system state such
    that the function under discussion will always yield an evaluated trace of
    unity. In the case of a finite chain, one can use this function to assess 
    the accuracy/error resulting from the simulation.

    Parameters
    ----------
    system_state : :class:`sbc.state.SystemState`
        The system state.
    
    Returns
    -------
    result : `float`
        The trace of the system's reduced density matrix.
    """
    # If trace was already calculated after evolving state.
    if system_state._trace is not None:
        return system_state._trace

    L = system_state.system_model.L

    tensor = np.array([1, 0, 0, 1], dtype=np.complex128)
    physical_1_legged_node = tn.Node(tensor)
    physical_1_legged_nodes = [physical_1_legged_node] * L

    result = \
        _apply_1_legged_nodes_to_system_state_mps(physical_1_legged_nodes,
                                                  system_state)
    result = float(np.real(result))

    # Cache result for future use.
    system_state._trace = result

    return result



def schmidt_spectrum_sum(system_state, bond_indices=None):
    r"""Calculate the Schmidt spectrum sum for a given set of bonds.

    Note that this function can only calculate the Schmidt spectrum sum for
    **finite** chains.

    Suppose we bipartition the system at the :math:`r^{\mathrm{th}}` bond of
    a finite chain. For this bipartition, the system's reduced density matrix 
    :math:`\hat{\rho}^{(A)}` can be expressed in the so-called operator Schmidt 
    decomposition:

    .. math ::
        \hat{\rho}^{(A)} = \sum_{c} S_{r, c} 
        \hat{\rho}_{r, c}^{\left(A, \vdash\right)} \otimes
        \hat{\rho}_{r, c}^{\left(A, \dashv\right)},
        :label: state_schmidt_spectrum_sum_schmidt_decomposition

    where :math:`S_{r, c}` is the Schmidt spectrum for the
    :math:`r^{\mathrm{th}}` bond, and the sets 
    :math:`\hat{\rho}_{r, c}^{\left(A, \vdash\right)}` and
    :math:`\hat{\rho}_{r, c}^{\left(A, \dashv\right)}` form orthonormal bases of
    Hermitian matrices in the Hilbert spaces of the left and right subsystems 
    formed by the bipartition respectively. By orthonormal, we mean that

    .. math ::
        \mathrm{Tr}\left\{
        \left(\hat{\rho}_{r, c_1}^{\left(A, \vdash\right)}
        \right)^{\dagger} \hat{\rho}_{r, c_2}^{\left(A, \vdash\right)}
        \right\} = \delta_{c_1, c_2},
        :label: state_schmidt_spectrum_sum_orthonormal_bases_1

    .. math ::
        \mathrm{Tr}\left\{
        \left(\hat{\rho}_{r, c_1}^{\left(A, \dashv\right)}
        \right)^{\dagger} \hat{\rho}_{r, c_2}^{\left(A, \dashv\right)}
        \right\} = \delta_{c_1, c_2},
        :label: state_schmidt_spectrum_sum_orthonormal_bases_2

    The Schmidt spectrum sum for the :math:`r^{\mathrm{th}}` bond is
    :math:`\sum_{c} S_{r, c}`. 
    
    :func:`sbc.state.schmidt_spectrum_sum` calculates the Schmidt spectrum
    sum for a given set of bonds at the current moment in time :math:`t`. The
    current time is stored in the :obj:`sbc.state.SystemState` object
    ``system_state``.

    Parameters
    ----------
    system_state : :class:`sbc.state.SystemState`
        The system state, where the system must be a **finite** chain.
    bond_indices : `None` | `array_like` (`int`, ndim=1), optional
        The bond indices corresponding to the bonds for which to calculate the
        Schmidt spectrum sum. If set to `None`, then ``bond_indices`` is reset 
        to ``range(system_state.system_model.L-1)``, i.e. the Schmidt spectrum 
        sum is calculated for all bonds.

    Returns
    -------
    result : `array_like` (`float`, shape=(``len(bond_indices)``,))
        For ``0<=r<len(bond_indices)``, ``result[r]`` is the Schmidt spectrum
        sum for the bond ``bond_indices[r]``.
    """
    if system_state.system_model.is_infinite:
        raise ValueError(_schmidt_spectrum_sum_err_msg_1a)
    
    L = system_state.system_model.L
    if bond_indices == None:
        bond_indices = range(L-1)

    result = []

    try:
        schmidt_spectrum = system_state._schmidt_spectrum
        
        for bond_idx in bond_indices:
            S_node = schmidt_spectrum[bond_idx]
            edge = S_node[0] ^ S_node[1]
            S_node_after_taking_trace = tn.contract(edge)
            S_sum = float(np.real(S_node_after_taking_trace.tensor))
            result.append(S_sum)

    except IndexError:
        raise IndexError(_schmidt_spectrum_sum_err_msg_1b)

    return result



def realignment_criterion(system_state):
    r"""Determine whether the system is entangled via the realignment criterion.

    Note that this function can only determine whether or not **finite** chains
    are entangled.

    Let :math:`S_{r, c}` be the Schmidt spectrum for the 
    :math:`r^{\mathrm{th}}` bond [see documentation for the function 
    :func:`sbc.state.realignment_criterion` for a discussion on Schmidt
    spectra]. According to the realignment criterion [see Refs. [Chen1]_ and
    [Rudolph1]_ for more detailed discussions regarding the realignment 
    criterion], if :math:`\sum_{c} S_{r, c} > 1` for any of the bonds, then the 
    system is entangled.

    :func:`sbc.state.realignment_criterion` determines whether the system is
    entangled at the current moment in time :math:`t` using the realignment
    criterion. The current time is stored in the :obj:`sbc.state.SystemState` 
    object ``system_state``.    

    Parameters
    ----------
    system_state : :class:`sbc.state.SystemState`
        The system state, where the system must be a **finite** chain.

    Returns
    -------
    entangled : `bool`
        If ``entangled`` is set to ``True``, then the system is entangled. 
        Otherwise, it is not.
    """
    if system_state.system_model.is_infinite:
        raise ValueError(_realignment_criterion_err_msg_1)
    
    S_sum = np.array(schmidt_spectrum_sum(system_state))
    entangled = np.any(S_sum > 1)

    return entangled



def spin_config_prob(spin_config, system_state):
    r"""Calculate spin configuration probability of a given spin configuration.

    The documentation for this function makes reference to the concept of a 
    'unit cell', which is introduced in the documentation for the module 
    :mod:`sbc.system`.

    This function calculates the probability of measuring a given system in a
    given classical spin configuration :math:`\boldsymbol{\sigma}_z` at the
    current moment in time :math:`t` in the simulation. The current time is 
    stored in the :obj:`sbc.state.SystemState` object ``system_state``.

    The classical spin configuration specifies values for the spins on sites
    :math:`r=0` to :math:`r=M L - 1`, where :math:`L` is the unit cell size,
    and :math:`M` is a positive integer that is less than or equal to the number
    of unit cells in the system. Note that in the case of a finite chain, there 
    is only one unit cell, hence :math:`M=1`. In the case of an infinite chain,
    :math:`M` can be any positive number.

    Parameters
    ----------
    spin_config : `array_like` (``-1`` | ``1``, shape=(``M*L``,))
        The classical spin configuration. If ``spin_config[0<=r<M*L]==-1``, 
        where ``M`` and ``L`` are :math:`M` and :math:`L` from above 
        respectively, then the ``r`` th spin of the spin configuration is in the
        "down" state. Otherwise, ``spin_config[0<=r<M*L]==1``, then the ``r`` th
        spin of the spin configuration is in the "up" state.
    system_state : :class:`sbc.state.SystemState`
        The system state. 

    Returns
    -------
    prob : `float`
        The spin configuration probability.
    """
    L = system_state.system_model.L
    if system_state.system_model.is_infinite:
        if len(spin_config) % L != 0:
            raise ValueError(_spin_config_prob_err_msg_1a)
    else:
        if len(spin_config) != L:
            raise ValueError(_spin_config_prob_err_msg_1b)

    spin_config = np.array(spin_config)
    if not np.all(np.logical_or(spin_config == 1, spin_config == -1)):
        raise ValueError(_spin_config_prob_err_msg_2)

    spin_config = spin_config.astype(np.int)

    physical_1_legged_nodes = []
    for spin in spin_config:
        if spin == 1:
            tensor = np.array([1, 0, 0, 0], dtype=np.complex128)
        else:
            tensor = np.array([0, 0, 0, 1], dtype=np.complex128)

        physical_1_legged_node = tn.Node(tensor)
        physical_1_legged_nodes.append(physical_1_legged_node)

    prob = _apply_1_legged_nodes_to_system_state_mps(physical_1_legged_nodes,
                                                     system_state)
    prob = float(np.real(prob)) / trace(system_state)

    return prob



_system_state_init_err_msg_1 = \
    ("The parameters `system_model` and `bath_model` must encode the same "
     "unit cell size, i.e. we must have that `system_model.L == bath_model.L`.")

_system_state_set_nodes_from_initial_state_nodes_err_msg_1 = \
    ("The number of nodes in the given MPS representing the initial state of "
     "the u=0 unit cell of the system is not equal to the system's unit cell "
     "size (as specified in the given transverse field Ising model).")

_system_state_set_nodes_from_initial_state_nodes_err_msg_2 = \
    ("Every node in the given MPS representing the initial system state needs "
     "to have the same physical dimensions: either physical dimensions equal "
     "to 2 or 4.")

_system_state_set_nodes_from_initial_state_nodes_err_msg_3 = \
    ("Given MPS representing the initial system state is not of the correct "
     "form: each node is expected to have three dangling edges, with the "
     "second edge having dimensions of either 2 or 4.")

_system_state_evolve_err_msg_1 = \
    ("The number of time steps `num_steps` must be a non-negative integer.")

_schmidt_spectrum_sum_err_msg_1a = \
    ("This function can only be applied to finite chains.")
_schmidt_spectrum_sum_err_msg_1b = \
    ("Valid bond indices range from 0 to `L-2`, where `L` is unit cell size.")

_realignment_criterion_err_msg_1 = \
    ("This function can only be applied to finite chains.")

_spin_config_prob_err_msg_1a = \
    ("The number of spins in the given spin configuration should be a positive "
     "multiple of the unit cell size.")
_spin_config_prob_err_msg_1b = \
    ("The number of spins in the given spin configuration does not match the "
     "unit cell size.")

_spin_config_prob_err_msg_2 = \
    ("A valid spin configuration consists of an array with each element equal "
     "to either 1 (signifying an Ising spin pointing 'up'), or -1 (signifying "
     "an Ising spin pointing 'down').")
