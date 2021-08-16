Infinite chain subject to z-noise
=================================

In this example, we implement a simulation of they dynamics of an infinite
ferromagnetic spin chain subject to fixed uniform transverse and longitudinal
fields, longitudinal couplings, and longitudinal noise (i.e. :math:`z`-noise)
with a spectral density comprising of a single ohmic component. The coupling
between the system and the environment is constant in time. The simulation
tracks the energy per spin, the first three correlation lengths [see the
documentation for the attribute
:attr:`sbc.state.SystemState.correlation_lengths` for a discussion on
correlation lengths], :math:`\left\langle\hat{\sigma}_{x; r=0}(t)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{z; r=0}(t)\hat{\sigma}_{z;
r+1}(t)\right\rangle`, and :math:`\left\langle\hat{\sigma}_{x; 0}(t)
\hat{\sigma}_{z; 1}(t)\hat{\sigma}_{x; 2}(t)\right\rangle`, where :math:`r=0` is
the center spin site of the infinite chain. You can also find the same code in
the file `examples/infinite-chain/z-noise/example.py` of the repository.

.. literalinclude:: ../examples/infinite-chain/z-noise/example.py
