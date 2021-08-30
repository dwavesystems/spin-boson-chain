Spin-Boson Chain
================

``spinbosonchain`` is a Python library for simulating the dynamics of a
generalized spin-boson chain model, where both the :math:`z`- and
:math:`y`-components of the spins are coupled to bosonic baths, rather than only
the :math:`z`-components. The library adopts the quasi-adiabatic path integral
(QUAPI) formalism to express the spin system's reduced density matrix as a
time-discretized path integral, comprising of a series of influence functionals
that encode the non-Markovian dynamics of the system. The path integral is
decomposed into a series of components that can be represented by tensor
networks. ``spinbosonchain`` currently relies heavily on Google's TensorNetwork_
package for its implementation of tensor networks and related operations.

Setting up spinbosonchain
-------------------------

For instructions on installing the ``spinbosonchain`` library, see the
:ref:`installation_instructions_sec` page.

.. Note for those reading the raw .rst file: see file 'docs/INSTALL.rst' for
   instructions on installing the spinbosonchain library as well as instructions
   for compiling the documentation of this library.

Learning how to use spinbosonchain
----------------------------------

For those new to the ``spinbosonchain`` library, it is recommended that they
take a look at the :ref:`examples_sec` page, which contains some code examples
that show how one can use the ``spinbosonchain`` library. While going through
the examples, readers can consult the :ref:`reference_guide_sec` to understand
what each line of code is doing in each example.

.. Note for those reading the raw .rst file: see directory 'examples' for the
   aforementioned code examples.

.. _TensorNetwork: https://github.com/google/TensorNetwork
