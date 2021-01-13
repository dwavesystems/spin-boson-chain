.. _installation_instructions_sec:

Installation instructions
=========================

Installation using conda and pip
--------------------------------

The following instructions describe how to install ``sbc`` in a newly created
virtual environment using the conda package manager from anaconda. You will need
to install anaconda3 in order for these instructions to work. First, update
conda by issuing the following command in the terminal::

    conda update conda

then type and return ``y`` at the prompt to update any packages if necessary.

Next, assuming that you have downloaded/cloned the ``sbc`` git repository,
change into the root directory of said repository. To install the minimal
requirements of ``sbc`` with MKL-optimized versions of ``numpy`` and
``scipy``, issue the following commands (READ THE ENTIRE PARAGRAPH BEFORE
DECIDING TO RUN THE COMMANDS BELOW)::
  
    conda env create -f environment-mkl.yml
    conda activate sbc

This will create a new virtual environment named ``sbc`` with the
aforementioned minimal requirements installed, and activate the new virtual
environment. It is recommended that you install the requirements using the above
commands if performing operations on Intel CPUs. Otherwise, it may be better
to install the minimal requirements with "no-MKL" versions of ``numpy`` and
``scipy``, which can be done by issuing the following commands::

    conda env create -f environment-nomkl.yml
    conda activate sbc

Once you have installed the required packages, you can install ``sbc`` by
issuing the following command::

    pip install .

Note that you must include the period as well. The installation will generate
some directories that can be safely removed to clean up the repository. To
remove these directories, issue the following command::

    python setup.py clean

Installation using pip only
---------------------------

The following instructions describe how to install ``sbc`` in a newly created
virtual environment using only pip. You will need to install Python 3.X where
X>=5. First, install pip (if you haven't done so already) by typing::

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

Next, assuming that you have downloaded/cloned the ``sbc`` git repository,
change into the root directory of said repository. To install the minimal
requirements of ``sbc`` with MKL-optimized versions of ``numpy`` and
``scipy``, issue the following commands (READ THE ENTIRE PARAGRAPH BEFORE
DECIDING TO RUN THE COMMANDS BELOW)::

    pip install -r requirements-mkl.txt

It is recommended that you install the requirements using the above command if
performing operations on Intel CPUs. Otherwise, it may be better
to install the minimal requirements with "no-MKL" versions of ``numpy`` and
``scipy``. In this case, do not run the above command.

Next, to install ``sbc``, issue the following command::

    pip install .

Note that you must include the period as well. The installation will generate
some directories that can be safely remove to clean up the repository. To
remove these directories, issue the following command::

    python setup.py clean

Using tensorflow as a backend in sbc
------------------------------------

In simulating the open-system dynamics of transverse field Ising chains,
``sbc`` performs various tensor network operations. By default, ``sbc``
uses ``numpy`` as the backend for performing these tensor network operations,
however one can use ``tensorflow`` as the backend instead if it is available.
This can be particularly advantageous if the ``tensorflow`` library has been
installed with GPU support. It is beyond the scope of this page to provide
installation instructions on ``tensorflow`` since the recommended method of
installation can depend on several factor, e.g. your machine architecture. Users
that are interested in using ``sbc`` with ``tensorflow`` are recommended to
check out this `link <https://www.tensorflow.org/install>`_ for more information
on installing ``tensorflow``. Note that ``sbc`` only works with version 2 of
``tensorflow``.

Update sbc
----------

If you, or someone else has made changes to this library, you can reinstall it
by issuing the following command::
  
    pip install .

Uninstall sbc
-------------

To uninstall ``sbc``, all you need to type is::

    pip uninstall sbc

Generating documention files
----------------------------

To generate documentation in html format from source files you will also need
the sphinx and numpydoc packages. If you have installed ``sbc`` within a
conda virtual environment, then you can install the aforementioned packages by
typing at the root directory::

    conda install --file requirements-doc.txt

Otherwise, if you installed ``sbc`` using pip only, then type at the root
directory::

    pip install -r requirements-doc.txt

Then, assuming you are in the root directory of ``sbc`` and that ``sbc``
is already installed, issue the following commands to generate html
documentation files::

    cd docs
    make html

This will generate a set of html files in ``./_build/html`` containing the
documentation of ``sbc``. You can then open any of the files using your
favorite web browser to start navigating the documentation within said browser::

    firefox ./_build/html/index.html &>/dev/null &

If ``sbc`` has been updated, the documentation has most likely changed
as well. To update the documentation, first remove the ``reference`` directory
inside ``docs``::

    rm -r reference

and then issue the following command::

    make clean

Now that we have cleaned everything up, we can simply run::

    make html

to generate the new documentation.
