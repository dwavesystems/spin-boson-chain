Finite chain subject to z-noise
===============================

In this example, we implement a simulation of they dynamics of a three-spin
system subject to fixed transverse and longitudinal fields, longitudinal
couplings, and longitudinal ohmic noise (i.e. :math:`z`-noise). The couplings
between the spins and the environment are constant in time. The simulation
tracks the system energy, :math:`\left\langle\hat{\sigma}_{x;
r}(t)\right\rangle`, :math:`\left\langle\hat{\sigma}_{z; r}(t)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{z; r}(t)\hat{\sigma}_{z; r+1}(t)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{y; r}(t)\hat{\sigma}_{z; r+1}(t)\right\rangle`,
:math:`\left\langle\hat{\sigma}_{x; 0}(t) \hat{\sigma}_{z; 1}(t)\hat{\sigma}_{x;
2}(t)\right\rangle`, :math:`\left\langle\hat{\sigma}_{x; 0}(t) \hat{\sigma}_{x;
1}(t)\hat{\sigma}_{x; 2}(t)\right\rangle`, the trace of the system's reduced
density operator, the probability of measuring :math:`\sigma_{z; r}(t)=+1` at
each site :math:`r`, and the probability of measuring :math:`\sigma_{z;
r}(t)=-1` at each site :math:`r`. The simulation also checks whether the system
is entangled using the realignment criterion. You can also find the same code in
the file `examples/finite-chain/z-noise/example.py` of the repository.

.. literalinclude:: ../examples/finite-chain/z-noise/example.py
