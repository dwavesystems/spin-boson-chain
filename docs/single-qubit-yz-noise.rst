Single qubit subject to y- and z-noise
======================================

In this example, we implement a simulation of the dynamics of a single
flux-qubit subject to quantum annealing, charge noise (modelled by
:math:`y`-noise), and hybrid flux noise (modelled by :math:`z`-noise). The
annealing schedule is taken from a spreadsheet. The simulation tracks the
expectation values of :math:`\hat{\sigma}_{x; r=0}`, :math:`\hat{\sigma}_{y;
r=0}`, and :math:`\hat{\sigma}_{z; r=0}`. You can also find the same code in the
file `examples/single-qubit/yz-noise/example.py` of the repository.

.. literalinclude:: ../examples/single-qubit/yz-noise/example.py
