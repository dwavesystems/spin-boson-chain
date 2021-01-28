#!/usr/bin/env python
r"""This module contains classes representing two-point functions that are 
used in the "yz-noise".
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# Contains classes and functions related to the model components of the bath.
import bath

# Import the class representing the eta-function.
from sbc._twopt.common import Eta

# Import the generic 'bath influence function' class.
from sbc._twopt.common import BathInfluence

# For converting base-4 variables to Ising spin pairs.
from sbc._base4 import base_4_to_ising_pair



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Non-Production"



##################################
## Define classes and functions ##
##################################

class YBathInfluence(BathInfluence):
    r"""A class representing the bath influence function for y-noise:

    .. math ::
        I_{y; r; n; k_1; k_2}^{(\mathrm{2-pt bath})}
        \left(j_{r; k_1}, j_{r; k_2}\right) = 
        I_{y; r; n; k_1; k_2}^{(\mathrm{bath})}
        \left(g_{1}\left(j_{r; k_1}\right), g_{-1}\left(j_{r; k_1}\right)
        g_{1}\left(j_{r; k_2}\right), g_{-1}\left(j_{r; k_2}\right)\right),

    where :math:`I_{y; r; n; k_1; k_2}^{(\mathrm{bath})}\left(\cdots\right)`
    is introduced in Eq. (47) of the detailed manuscript (DM) on our QUAPI-TN
    approach, :math:`g_{\alpha}(j)` is introduced in Eq. (73) of the DM, and
    :math:`j_{r; k_1}` and :math:`j_{r; k_2}` are base-4 variables [see Sec. 3.2
    of the DM for a discussion of base-4 variables].
    
    Parameters
    ----------
    bath_model : :class:`sbc.bath.Model`
        The bath model components.
    dt : `float`
        The simulation time step size.
    """
    def __init__(self, bath_model, dt):
        A_y_T = bath_model.spectral_density_y_cmpnt
        if A_y_T == None:
            A_y_T = bath._trivial_spectral_density_cmpnt  # i.e. no y-noise.

        # The eta function is introduced in Eqs. (49)-(59) of the DM.
        eta_y = Eta(A_v_T=A_y_T,
                    dt=dt,
                    tilde_w_set=[0.25, 0.25, 0.25, 0.50],
                    eval_k_v_n=lambda n: 2*n+1)

        # Calculate Eq. (67), where tau is the system's 'memory'.
        K_tau = _calc_K_tau(tau=bath_model.memory, dt=dt)

        # Parent class does most of the work.
        super().__init__(eta=eta_y, K_tau=K_tau, K_v_tau=2*K_tau)

        return None



class ZBathInfluence(BathInfluence):
    r"""A class representing the bath influence function for z-noise:

    .. math ::
        I_{z; r; n; k_1; k_2}^{(\mathrm{2-pt bath})}
        \left(j_{r; k_1}, j_{r; k_2}\right) = 
        I_{z; r; n; k_1; k_2}^{(\mathrm{bath})}
        \left(g_{1}\left(j_{r; k_1}\right), g_{-1}\left(j_{r; k_1}\right)
        g_{1}\left(j_{r; k_2}\right), g_{-1}\left(j_{r; k_2}\right)\right),

    where :math:`I_{z; r; n; k_1; k_2}^{(\mathrm{bath})}\left(\cdots\right)`
    is introduced in Eq. (47) of the detailed manuscript (DM) on our QUAPI-TN
    approach, :math:`g_{\alpha}(j)` is introduced in Eq. (73) of the DM, and
    :math:`j_{r; k_1}` and :math:`j_{r; k_2}` are base-4 variables [see Sec. 3.2
    of the DM for a discussion of base-4 variables].
    
    Parameters
    ----------
    bath_model : :class:`sbc.bath.Model`
        The bath model components.
    dt : `float`
        The simulation time step size.
    """
    def __init__(self, bath_model, dt):
        A_z_T = bath_model.spectral_density_z_cmpnt
        if A_z_T == None:
            A_z_T = bath._trivial_spectral_density_cmpnt  # i.e. no z-noise.

        # The eta function is introduced in Eqs. (49)-(59) of the DM.
        eta_z = Eta(A_v_T=A_z_T,
                    dt=dt,
                    tilde_w_set=[0.50, 0.25, 0.75, 1.00],
                    eval_k_v_n=lambda n: n+1)

        # Calculate Eq. (67), where tau is the system's 'memory'.
        K_tau = _calc_K_tau(tau=bath_model.memory, dt=dt)

        # Parent class does most of the work.
        super().__init__(eta=eta_z, K_tau=K_tau, K_v_tau=K_tau)

        return None



def z2y(sigma_y_r_alpha_k1, sigma_z_r_alpha_k2):
    r"""This function implements the inverse of Eq. (40) of the detailed
    manuscript (DM) on our QUAPI-TN approach. 
    """
    result = (1 - 1j*sigma_y_r_alpha_k1 + sigma_z_r_alpha_k2
              + 1j*sigma_y_r_alpha_k1*sigma_z_r_alpha_k2) / 2.0 / np.sqrt(2)

    return result



def y2z(sigma_z_r_alpha_k2, sigma_y_r_alpha_k1):
    r"""This function implements the inverse of Eq. (41) of the detailed
    manuscript (DM) on our QUAPI-TN approach. 
    """
    result = np.conj(z2y(sigma_y_r_alpha_k1, sigma_z_r_alpha_k2))

    return result



class YZInfluence1():
    r"""A class representing the "basis change" influence function:

    .. math ::
        I^{(\mathrm{2-pt y<->z}, 1)}
        \left(j_{r; k_1}, j_{r; k_2}\right) = 
        I^{(y \leftrightarrow z, 1)}
        \left(g_{1}\left(j_{r; k_1}\right), g_{-1}\left(j_{r; k_1}\right)
        g_{1}\left(j_{r; k_2}\right), g_{-1}\left(j_{r; k_2}\right)\right),

    where :math:`I^{(y \leftrightarrow z, 1)}\left(\cdots\right)`
    is introduced in Eq. (38) of the detailed manuscript (DM) on our QUAPI-TN
    approach, :math:`g_{\alpha}(j)` is introduced in Eq. (73) of the DM, and
    :math:`j_{r; k_1}` and :math:`j_{r; k_2}` are base-4 variables [see Sec. 3.2
    of the DM for a discussion of base-4 variables].
    """
    def __init__(self):
        return None

    

    def eval(self, j_r_k1, j_r_k2):
        r"""Evaluate the "basis change" influence function
        I^{(\mathrm{2-pt y<->z}, 1)}\left(j_{r; k_1}, j_{r; k_2}\right) [see 
        constructor documentation for the definition of the bath influence 
        function].

        Parameters
        ----------
        j_r_k1 : ``0`` | ``1`` | ``2`` | ``3``
            The base-4 variable :math:`j_{r; k_1}`.
        j_r_k2 : ``0`` | ``1`` | ``2`` | ``3``
            The base-4 variable :math:`j_{r; k_2}`.
        """
        # Convert base-4 variables to Ising spin pairs. See Sec. 3.2 of DM for
        # a discussion on such conversions.
        sigma_r_pos1_k1, sigma_r_neg1_k1 = base_4_to_ising_pair(j_r_k1)
        sigma_r_pos1_k2, sigma_r_neg1_k2 = base_4_to_ising_pair(j_r_k2)

        result = (y2z(sigma_r_pos1_k2, sigma_r_pos1_k1)
                  * z2y(sigma_r_neg1_k1, sigma_r_neg1_k2))

        return result



class YZInfluence2():
    r"""A class representing the "basis change" influence function:

    .. math ::
        I^{(\mathrm{2-pt y<->z}, 2)}
        \left(j_{r; k_1}, j_{r; k_2}\right) = 
        I^{(y \leftrightarrow z, 2)}
        \left(g_{1}\left(j_{r; k_1}\right), g_{-1}\left(j_{r; k_1}\right)
        g_{1}\left(j_{r; k_2}\right), g_{-1}\left(j_{r; k_2}\right)\right),

    where :math:`I^{(y \leftrightarrow z, 1)}\left(\cdots\right)`
    is introduced in Eq. (39) of the detailed manuscript (DM) on our QUAPI-TN
    approach, :math:`g_{\alpha}(j)` is introduced in Eq. (73) of the DM, and
    :math:`j_{r; k_1}` and :math:`j_{r; k_2}` are base-4 variables [see Sec. 3.2
    of the DM for a discussion of base-4 variables].
    """
    def __init__(self):
        return None

    

    def eval(self, j_r_k1, j_r_k2):
        r"""Evaluate the "basis change" influence function
        I^{(\mathrm{2-pt y<->z}, 2)}\left(j_{r; k_1}, j_{r; k_2}\right) [see 
        constructor documentation for the definition of the bath influence 
        function].

        Parameters
        ----------
        j_r_k1 : ``0`` | ``1`` | ``2`` | ``3``
            The base-4 variable :math:`j_{r; k_1}`.
        j_r_k2 : ``0`` | ``1`` | ``2`` | ``3``
            The base-4 variable :math:`j_{r; k_2}`.
        """
        # Convert base-4 variables to Ising spin pairs. See Sec. 3.2 of DM for
        # a discussion on such conversions.
        sigma_r_pos1_k1, sigma_r_neg1_k1 = base_4_to_ising_pair(j_r_k1)
        sigma_r_pos1_k2, sigma_r_neg1_k2 = base_4_to_ising_pair(j_r_k2)

        result = (z2y(sigma_r_pos1_k1, sigma_r_pos1_k2)
                  * y2z(sigma_r_neg1_k2, sigma_r_neg1_k1))

        return result
