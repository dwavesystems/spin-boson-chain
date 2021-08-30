.. _installation_instructions_sec:

Installation instructions
=========================

Installation using conda and pip (recommended)
----------------------------------------------

The following instructions describe how to install ``spinbosonchain`` in a newly
created virtual environment using the conda package manager from anaconda. You
will need to install anaconda3 in order for these instructions to work. First,
update conda by issuing the following command in the terminal::

    conda update conda

then type and return ``y`` at the prompt to update any packages if necessary.

Next, assuming that you have downloaded/cloned the ``spin-boson-chain`` git
repository, change into the root directory of said repository. To install the
minimal requirements of ``spinbosonchain`` with MKL-optimized versions of
``numpy`` and ``scipy``, issue the following commands (READ THE ENTIRE PARAGRAPH
BEFORE DECIDING TO RUN THE COMMANDS BELOW)::
  
    conda env create -f environment-mkl.yml
    conda activate sbc

This will create a new virtual environment named ``sbc`` with the aforementioned
minimal requirements installed, and activate the new virtual environment. It is
recommended that you install the requirements using the above commands if
performing operations on Intel CPUs. Otherwise, it may be better to install the
minimal requirements with the default versions of ``numpy`` and ``scipy``, which
can be done by issuing the following commands::

    conda env create -f environment.yml
    conda activate sbc

Once you have installed the required packages, you can install
``spinbosonchain`` by issuing the following command::

    pip install .

Note that you must include the period as well. The installation will generate
some directories that can be safely removed to clean up the repository. To
remove these directories, issue the following command::

    python setup.py clean

Installation using pip only
---------------------------

The following instructions describe how to install ``spinbosonchain`` in a newly
created virtual environment using only pip. You will need to install Python 3.X
where X>=5. First, install pip (if you haven't done so already) by typing::

    python3 -m pip install --user --upgrade pip

Install virtualenv, which we will use to create our virtual environment::

    python3 -m pip install --user virtualenv

Create a virtual environment in your home directory. The following command
installs a virtual environment named ``sbc`` with Python 3.6::

    virtualenv -p python3.6 ~/sbc

Of course, the above command can be modified to install a virtual environment
with a different version of Python. Activate the virtual environment::

    source ~/sbc/bin/activate

Upgrade pip inside the virtual environment::

    pip install --upgrade pip

Next, assuming that you have downloaded/cloned the ``spin-boson-chain`` git
repository, change into the root directory of said repository. To install the
minimal requirements of ``spinbosonchain`` with MKL-optimized versions of
``numpy`` and ``scipy``, issue the following commands (READ THE ENTIRE PARAGRAPH
BEFORE DECIDING TO RUN THE COMMANDS BELOW)::

    pip install -r requirements-mkl.txt

It is recommended that you install the requirements using the above command if
performing operations on Intel CPUs. Otherwise, it may be better to install the
minimal requirements with the default versions of ``numpy`` and ``scipy``. In
this case, do not run the above command.

Next, to install ``spinbosonchain``, issue the following command::

    pip install .

Note that you must include the period as well. The installation will generate
some directories that can be safely remove to clean up the repository. To
remove these directories, issue the following command::

    python setup.py clean

Using tensorflow as a backend in spinbosonchain
-----------------------------------------------

In simulating the open-system dynamics of transverse field Ising chains,
``spinbosonchain`` performs various tensor network operations. By default,
``spinbosonchain`` uses ``numpy`` as the backend for performing these tensor
network operations, however one can use ``tensorflow`` as the backend instead if
it is available.  This can be particularly advantageous if the ``tensorflow``
library has been installed with GPU support. It is beyond the scope of this page
to provide installation instructions on ``tensorflow`` since the recommended
method of installation can depend on several factor, e.g. your machine
architecture. Users that are interested in using ``spinbosonchain`` with
``tensorflow`` are recommended to check out this `link
<https://www.tensorflow.org/install>`_ for more information on installing
``tensorflow``. Note that ``spinbosonchain`` only works with version 2 of
``tensorflow``.

Update spinbosonchain
---------------------

If you, or someone else has made changes to this library, you can reinstall it
by issuing the following command::
  
    pip install .

Uninstall spinbosonchain
------------------------

To uninstall ``spinbosonchain``, all you need to type is::

    pip uninstall spinbosonchain

Testing spinbosonchain
----------------------

A set of example scripts can be found in the ``examples`` directory that a user
can run to see how ``spinbosonchain`` can be used to simulate a variety of
different systems. Optionally, for each example that simulates a finite system
(except for that which simulates a single-qubit subject to :math:`z`-noise), the
data obtained from ``spinbosonchain`` can be compared against that obtained by
exact diagonalization (ED) via the QuSpin_ package to verify the correctness of
the algorithms that are implemented in ``spinbosonchain``. For the exceptional
case, the data obtained from ``spinbosonchain`` is compared against that
obtained from the Lindblad formalism (which is essentially exact for said
case). There exists one example that simulates a noise-free infinite chain in
which the data obtained from ``spinbosonchain`` can be compared against that
obtained by the time evolving block decimation (TEBD) method via the TeNPy_
package. Altogether the examples provide sufficient verification of the
correctness of all the algorithms implemented in ``spinbosonchain``.

Comparisons against ED and TEBD, where applicable, will be automatically enabled
in the example scripts if the ``quspin`` and ``tenpy`` libraries are installed
respectively. ``matplotlib`` also needs to be installed in order to generate
comparison plots. If you have installed the ``spinbosonchain`` library in a
conda virtual environment, then you can install ``quspin`` easily with OpenMP
support by issuing the following command in the terminal::

    conda install -c weinbe58 omp quspin=0.3.6

or without OpenMP support by running::

    conda install -c weinbe58 quspin=0.3.6

If you installed ``spinbosonchain`` using pip only, then you will have to
install ``quspin`` manually following the instructions found on the
documentation webpage of the QuSpin_ package. Irrespective of the environment
that you are using, you can install ``tenpy`` by running::

    pip install physics-tenpy==0.8.4

``matplotlib`` can be installed within a conda virtual environment by running
the following command in the root directory of the ``spinbosonchain``
repository::

    conda install --file requirements-plot.txt

or alternatively, if ``spinbosonchain`` was installed using pip only, then
``matplotlib`` can be installed by running::

    pip install -r requirements-plot.txt

Generating documention files
----------------------------

To generate documentation in html format from source files you will also need
the sphinx and numpydoc packages. If you have installed ``spinbosonchain``
within a conda virtual environment, then you can install the aforementioned
packages by typing in the root directory of the repository::

    conda install --file requirements-doc.txt

Otherwise, if you installed ``spinbosonchain`` using pip only, then type in the
root directory of the repository::

    pip install -r requirements-doc.txt

Then, assuming you are in the root directory of the ``spin-boson-chain`` git
repository and that ``spinbosonchain`` is already installed, issue the following
commands to generate html documentation files::

    cd docs
    make html

This will generate a set of html files in ``./_build/html`` containing the
documentation of ``spinbosonchain``. You can then open any of the files using
your favorite web browser to start navigating the documentation within said
browser::

    firefox ./_build/html/index.html &>/dev/null &

If ``spinbosonchain`` has been updated, the documentation has most likely
changed as well. To update the documentation, first remove the ``reference``
directory inside ``docs``::

    rm -r reference

and then issue the following command::

    make clean

Now that we have cleaned everything up, we can simply run::

    make html

to generate the new documentation.



.. _QuSpin: https://weinbe58.github.io/QuSpin/index.html
.. _TeNPy: https://tenpy.readthedocs.io/en/latest/
