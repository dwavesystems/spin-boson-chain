Single qubit subject to z-noise
===============================

In this example, we implement a simulation of the dynamics of a single spin
subject to a fixed transverse field, and longitudinal noise (i.e.
:math:`z`-noise) with a spectral density comprising of a single ohmic component.
The coupling between the spin and the environment is constant in time. The
simulation tracks the expectation values of :math:`\hat{\sigma}_{x; r=0}`,
:math:`\hat{\sigma}_{y; r=0}`, and :math:`\hat{\sigma}_{z; r=0}`. You can also
find the same code in the file
`examples/single-qubit/z-noise/example.py` of the repository.

.. literalinclude:: ../examples/single-qubit/z-noise/example.py
