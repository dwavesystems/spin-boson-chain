#!/usr/bin/env python
r"""This subpackage contains classes representing quantities related to
influence functions, and influence path/functionals in our QUAPI formalism, upon
which ``sbc`` is based.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# Import submodules of subpackage.
from . import eta
from . import twopt
from . import tensorfactory
from . import path



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Non-Production"
