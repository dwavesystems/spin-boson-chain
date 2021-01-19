#!/usr/bin/env python
r"""This subpackage contains classes representing two-point (i.e. two-time)
functions that occur in our QUAPI-TN approach. ``sbc`` is structured so that
there is an algorithm for handling systems with sigma-y and sigma-z noise,
which we refer to as the "yz-noise" algorithm, and an algorithm for handling
systems with sigma-z noise only, which we refer to as the "z-noise" algorithm.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# Import submodules of subpackage.
from . import common



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Non-Production"
