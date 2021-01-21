#!/usr/bin/env python
r"""This module contains classes representing two-point (i.e. two-time) 
functions that are common to both the "yz-noise" and "z-noise" algorithms.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For defining abstract classes.
from abc import ABC, abstractmethod

# Import factorial function.
from math import factorial



# For evaluating special math functions and general array handling.
import numpy as np

# For evaluating numerically integrals.
from scipy.integrate import quad



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

class Eta():
    r"""A class representing the eta-function, 
    :math:`\eta_{\nu; r; n; k_1; k_2}`, which is introduced in Eqs. (49)-(59) 
    of the detailed manuscript (DM) on our QUAPI-TN approach. See Appendix D 
    for details on how we evaluate the eta-function numerically.
    
    Parameters
    ----------
    A_v_T : :class:`sbc.bath.SpectralDensityCmpnt`
        The finite-temperature spectral density of some component of noise of
        interest. The quantity is denoted by :math:`A_{\nu;T}(\omega)` in the
        DM, where :math:`\nu\in\{y,z\}` indicates the component of noise of
        interest.
    dt : `float`
        The time step size.
    tilde_w_set : `array_like` (`float`, shape=(4,))
        This is the array :math:`\left(\tilde{w}_{\nu}, \tilde{w}_{\nu; 0},
        \tilde{w}_{\nu; 1}, \tilde{w}_{\nu; 2}\right)`, which is introduced in
        Eqs. (548)-(554) of the DM.
    eval_k_v_n : `func` (`int`)
        The functional form of :math:`k_{\nu; n}`, which appears in Eq. (549)
        of the DM. The function's argument is :math:`n`, i.e. :math:`\nu` is
        treated as fixed.
    """
    def __init__(self, A_v_T, dt, tilde_w_set, eval_k_v_n):
        self.A_v_T = A_v_T
        self.dt = dt
        self.tilde_w_v = tilde_w_set[0]
        self.tilde_w_v_0 = tilde_w_set[1]
        self.tilde_w_v_1 = tilde_w_set[2]
        self.tilde_w_v_2 = tilde_w_set[3]
        self.eval_k_v_n = eval_k_v_n

        return None



    def eval(self, k1, k2, n):
        r"""'DM' refers to the detailed manuscript on our QUAPI+TN approach.
        """
        self.update_W_vars(k1, k2, n)

        result = 0.0j
        for A_v_T_subcmpnt in self.A_v_T.subcmpnts:
            # For each ``A_v_T_subcmpnt`` we are evaluating Eq. (642) of DM.
            self.update_integration_pts(A_v_T_subcmpnt)
            for a in range(0, 4):
                result += self.eval_real_cmpnt(A_v_T_subcmpnt, a)
                result += 1j*self.eval_imag_cmpnt(k1, k2, A_v_T_subcmpnt, a)
        
        return result



    def update_W_vars(self, k1, k2, n):
        r"""The ``W_vars`` are the :math:`W_{\nu; n; k_1, k_2}^{(l=0,1,2,3)}`
        which are introduced in Eqs. (618)-(612) of the detailed manuscript (DM)
        on our QUAPI-TN approach.
        """
        self.update_ST_weights(k1, k2, n)

        dt = self.dt
        
        self.W_vars = [0.0]*4
        self.W_vars[0] = dt * sum(self.ST_weights[k2:(k1-1)+1])
        self.W_vars[1] = dt * sum(self.ST_weights[k2+1:k1+1])
        self.W_vars[2] = dt * sum(self.ST_weights[k2+1:(k1-1)+1])
        self.W_vars[3] = dt * sum(self.ST_weights[k2:k1+1])

        return None
        
        

    def update_ST_weights(self, k1, k2, n):
        r"""The ``ST_weights`` are the :math:`w_{\nu; n; k}` which are 
        introduced in Eqs. (548) of the detailed manuscript (DM) on our 
        QUAPI-TN approach.
        """
        k_v_n = self.eval_k_v_n(n)

        # Eq. (548) of DM.
        self.ST_weights = [0.0]*(k_v_n+1)
        self.ST_weights[0] = self.tilde_w_v_0
        self.ST_weights[1] = self.tilde_w_v if n==1 else self.tilde_w_v_1
        for idx in range(2, k_v_n-1):
            self.ST_weights[idx] = self.tilde_w_v_2
        self.ST_weights[k_v_n-1] = self.ST_weights[1]
        self.ST_weights[k_v_n] = self.ST_weights[0]

        return None



    def update_integration_pts(self, A_v_T_subcmpnt):
        r"""The ``integration_pts`` are the 
        :math:`\omega_{\nu; n; k_1; k_2; \varsigma; a}`
        which are introduced in Eq. (647) of the detailed manuscript (DM)
        on our QUAPI-TN approach.
        """
        W_var_max = max(self.W_vars)
        
        self.integration_pts = [0.0]*5
        self.integration_pts[0] = -A_v_T_subcmpnt.limit_0T.hard_cutoff_freq

        if W_var_max != 0.0:
            self.integration_pts[1] = \
                max(self.integration_pts[0], -np.pi / W_var_max)
        else:
            self.integration_pts[1] = self.integration_pts[0]
            
        self.integration_pts[2] = 0.0
        self.integration_pts[3] = -self.integration_pts[1]
        self.integration_pts[4] = -self.integration_pts[0]

        return None



    def eval_real_cmpnt(self, A_v_T_subcmpnt, a):
        r"""'DM' refers to the detailed manuscript on our QUAPI+TN approach.
        """
        if (a == 1) or (a == 2):
            # Evaluate Eq. (643) of DM.
            result = self.eval_integral_type_R1(A_v_T_subcmpnt, a)
        else:
            # Evaluate Eq. (645) of DM.
            result = 0.0
            for l in range(0, 4):
                result += self.eval_integral_type_R2(A_v_T_subcmpnt, a, l)

        return result



    def eval_imag_cmpnt(self, k1, k2, A_v_T_subcmpnt, a):
        r"""'DM' refers to the detailed manuscript on our QUAPI+TN approach.
        """
        if (a == 1) or (a == 2):
            # Evaluate Eq. (644) of DM.
            result = self.eval_integral_type_I1(k1, k2, A_v_T_subcmpnt, a)
        else:
            # Evaluate Eq. (646) of DM.
            result = 0.0
            for l in range(0, 4):
                result += self.eval_integral_type_I2(k1, k2,
                                                     A_v_T_subcmpnt, a, l)

        return result



    def eval_integral_type_R1(self, A_v_T_subcmpnt, a):
        r"""This method evaluates Eq. (643) of the detailed manuscript (DM) on 
        our QUAPI+TN approach.
        """
        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        if pt2 <= pt1:
            result = 0.0
            return result
        
        W_vars = self.W_vars
        W_var_max = max(W_vars)
        pi = np.pi

        def summand(omega, m):
            return ((-1)**m * omega**(2*m-2) / factorial(2*m)
                    * (W_vars[0]**(2*m) + W_vars[1]**(2*m)
                       - W_vars[2]**(2*m) - W_vars[3]**(2*m)))
        
        # Implement Eqs. (648) and (649) of DM.
        def F(omega):
            if abs(omega * W_var_max) < 1.0e-3:
                return sum([summand(omega, m) for m in range(1, 4)])
            else:
                return ((np.cos(W_vars[0]*omega) + np.cos(W_vars[1]*omega)
                         - np.cos(W_vars[2]*omega) - np.cos(W_vars[3]*omega))
                        / omega / omega)

        # See discussion below Eq. (654) of DM.
        integrand = lambda omega: (A_v_T_subcmpnt._eval(omega)
                                   * F(omega) / 2.0 / pi)
        result = quad(integrand, a=pt1, b=pt2, limit=2000)[0]

        return result
            


    def eval_integral_type_R2(self, A_v_T_subcmpnt, a, l):
        r"""This method evaluates Eq. (652) of the detailed manuscript (DM) on 
        our QUAPI+TN approach.
        """
        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        
        if pt2 <= pt1:
            result = 0.0
            return result
        
        W_var = self.W_vars[l]
        sign_prefactor = (-1)**(l//2)
        pi = np.pi

        # See discussion below Eq. (654) of DM.
        integrand = lambda omega: (sign_prefactor * A_v_T_subcmpnt._eval(omega)
                                   / omega / omega / 2.0 / pi)
        if W_var == 0:
            result = quad(integrand, a=pt1, b=pt2, limit=2000)[0]
        else:
            result = quad(integrand, a=pt1, b=pt2, weight="cos", wvar=W_var,
                          limit=2000*int(abs((pt2-pt1)/pt1)+1))[0]

        return result



    def eval_integral_type_I1(self, k1, k2, A_v_T_subcmpnt, a):
        r"""This method evaluates Eq. (644) of the detailed manuscript (DM) on 
        our QUAPI+TN approach.
        """
        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        
        if (k1 == k2) or (pt2 <= pt1):
            result = 0.0
            return result
        
        W_vars = self.W_vars
        W_var_max = max(W_vars)
        pi = np.pi

        def summand(omega, m):
            return ((-1)**m * omega**(2*m-1) / factorial(2*m+1)
                    * (W_vars[0]**(2*m+1) + W_vars[1]**(2*m+1)
                       - W_vars[2]**(2*m+1) - W_vars[3]**(2*m+1)))
        
        # Implement Eqs. (650) and (651) of DM.
        def F(omega):
            if abs(omega * W_var_max) < 1.0e-3:
                return -sum([summand(omega, m) for m in range(1, 4)])
            else:
                return ((-np.sin(W_vars[0]*omega) - np.sin(W_vars[1]*omega)
                         + np.sin(W_vars[2]*omega) + np.sin(W_vars[3]*omega))
                        / omega / omega)

        # See discussion below Eq. (654) of DM.
        integrand = lambda omega: (A_v_T_subcmpnt._eval(omega)
                                   * F(omega) / 2.0 / pi)
        result = quad(integrand, a=pt1, b=pt2, limit=2000)[0]

        return result
            


    def eval_integral_type_I2(self, k1, k2, A_v_T_subcmpnt, a, l):
        r"""This method evaluates Eq. (653) of the detailed manuscript (DM) on 
        our QUAPI+TN approach.
        """
        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        W_var = self.W_vars[l]
        
        if (k1 == k2) or (W_var == 0) or (pt2 <= pt1):
            result = 0.0
            return result
        
        sign_prefactor = -(-1)**(l//2)
        pi = np.pi

        # See discussion below Eq. (654) of DM.
        integrand = lambda omega: (sign_prefactor * A_v_T_subcmpnt._eval(omega)
                                   / omega / omega / 2.0 / pi)
        result = quad(integrand, a=pt1, b=pt2, weight="sin", wvar=W_var,
                      limit=2000*int(abs((pt2-pt1)/pt1)+1))[0]

        return result



class BathInfluence():
    r"""A class representing the bath influence function

    .. math ::
        I_{\nu; r; n; k_1; k_2}^{(\mathrm{2-pt bath})}
        \left(j_{r; k_1}, j_{r; k_2}\right) = 
        I_{\nu; r; n; k_1; k_2}^{(\mathrm{bath})}
        \left(g_{1}\left(j_{r; k_1}\right), g_{-1}\left(j_{r; k_1}\right)
        g_{1}\left(j_{r; k_2}\right), g_{-1}\left(j_{r; k_2}\right)\right),

    where :math:`I_{\nu; r; n; k_1; k_2}^{(\mathrm{bath})}\left(\cdots\right)`
    is introduced in Eq. (47) of the detailed manuscript (DM) on our QUAPI-TN
    approach, :math:`g_{\alpha}(j)` is introduced in Eq. (73) of the DM, and
    :math:`j_{r; k_1}` and :math:`j_{r; k_2}` are base-4 variables [see Sec. 3.2
    of the DM for a discussion of base-4 variables].
    
    Parameters
    ----------
    eta : :class:`sbc._twopt.common.Eta`
        The eta-function, :math:`\eta_{\nu; r; n; k_1; k_2}`, which is 
        introduced in Eqs. (49)-(59) of the DM. See Appendix D for details on 
        how we evaluate the eta-function numerically.
    K_tau : `int`
        This is the quantity :math:`K_{\tau}` that appears throughtout the DM,
        where :math:`\tau` indicates the bath correlation time, or the system's 
        "memory". :math:`K_{\tau}` is given by Eq. (67).
    K_v_tau : `int`
        This is the quantity :math:`K_{\nu; \tau}` that appears throughtout the
        DM, where :math:`\nu` indicates the component of noise of interest, and 
        :math:`\tau` indicates the bath correlation time, or the system's 
        "memory". The :math:`K_{\nu; \tau}` are given by Eqs. (65) and (66).
    """
    def __init__(self, eta, K_tau, K_v_tau):
        self.K_tau = K_tau
        self.K_v_tau = K_v_tau
        self.eval_k_v_n = eta.eval_k_v_n

        self.calc_eta_caches(eta)

        self.set_k1_k2_n(0, 0, 1)

        return None



    def calc_eta_caches(self, eta):
        r"""In Sec. D.1 of the detailed manuscript (DM) of our QUAPI-TN
        approach, we discuss why is it computationally beneficial to calculate
        only the :math:`\eta_{\nu; r; n; k_1; k_2}` that are required for a full
        simulation, cache them, and then reuse the results wherever possible.
        This method calculates the :math:`\eta_{\nu; r; n; k_1; k_2}` that are
        required and caches them.
        """
        K_tau = self.K_tau
        K_v_tau = self.K_v_tau
        eval_k_v_n = self.eval_k_v_n

        # Implementing Eq. (588) of the DM.
        lf = K_tau - 2
        self.eta_cache_1 = [0.0j]*(lf+1)
        for l in range(0, lf+1):
            k = eval_k_v_n(n=l+1)
            k1 = k - 1
            k2 = 0
            n = l+1
            self.eta_cache_1[l] = eta.eval(k1, k2, n)

        # Implementing Eq. (589) of the DM.
        lf = K_tau - 3 + K_v_tau // (2*K_tau)
        self.eta_cache_2 = [0.0j]*(lf+1)
        for l in range(0, lf+1):
            k = eval_k_v_n(n=l+1)
            k1 = k
            k2 = 0
            n = l+1
            self.eta_cache_2[l] = eta.eval(k1, k2, n)

        # Implementing Eq. (590) of the DM.
        lf = K_tau - 1
        self.eta_cache_3 = [0.0j]*(lf+1)
        for l in range(0, lf+1):
            k = eval_k_v_n(n=l+1)
            k1 = k - 1
            k2 = 1
            n = l+1
            self.eta_cache_3[l] = eta.eval(k1, k2, n)

        # Implementing Eq. (591) of the DM.
        lf = K_v_tau - 1
        self.eta_cache_4 = [0.0j]*(lf+1)
        for l in range(0, lf+1):
            k = eval_k_v_n(n=K_tau+2)
            k1 = k - 2
            k2 = k - K_v_tau - 1 + l
            n = K_tau + 2
            self.eta_cache_4[l] = eta.eval(k1, k2, n)

        # Implementing Eq. (592) of the DM.
        lf = K_v_tau - 1
        self.eta_cache_5 = [0.0j]*(lf+1)
        for l in range(0, lf+1):
            k = eval_k_v_n(n=K_tau+1)
            k1 = k - 1
            k2 = k - K_v_tau + l
            n = K_tau + 1
            self.eta_cache_5[l] = eta.eval(k1, k2, n)

        # Implementing Eq. (593) of the DM.
        lf = K_v_tau - 1
        self.eta_cache_6 = [0.0j]*(lf+1)
        for l in range(0, lf+1):
            k = eval_k_v_n(n=K_tau)
            k1 = k
            k2 = k - K_v_tau + 1 + l
            n = K_tau
            self.eta_cache_6[l] = eta.eval(k1, k2, n)

        return None



    def eval(self, j_r_k1, j_r_k2):
        r"""Evaluate the bath influence function
        :math:`I_{\nu; r; n; k_1; k_2}^{(\mathrm{2-pt bath})}
        \left(j_{r; k_1}, j_{r; k_2}\right)` [see constructor documentation for
        the definition of the bath influence function].

        Parameters
        ----------
        j_r_k1 : ``0`` | ``1`` | ``2`` | ``3``
            The base-4 variable :math:`j_{r; k_1}`.
        j_r_k2 : ``0`` | ``1`` | ``2`` | ``3``
            The base-4 variable :math:`j_{r; k_2}`.
        """
        # The "eta cache" is selected in the call to the method
        # :meth:`sbc._twopt.common.BathInfluence.set_k1_k2_n`, which first
        # sets :math:`k_1`, :math:`k_2`, and :math:`n`.
        eta_real_part = self.selected_eta_cache_real_part
        eta_imag_part = self.selected_eta_cache_imag_part
        
        # Convert base-4 variables to Ising spin pairs. See Sec. 3.2 of DM for
        # a discussion on such conversions.
        sigma_r_pos1_k1, sigma_r_neg1_k1 = base_4_to_ising_pair(j_r_k1)
        sigma_r_pos1_k2, sigma_r_neg1_k2 = base_4_to_ising_pair(j_r_k2)

        # Implement Eq. (48) of DM.
        gamma = ((sigma_r_pos1_k2-sigma_r_neg1_k2)
                 * ((sigma_r_pos1_k1-sigma_r_neg1_k1) * eta_real_part
                    + 1.0j * (sigma_r_pos1_k1+sigma_r_neg1_k1) * eta_imag_part))

        # Implement Eq. (47) of DM.
        result = np.exp(-gamma)

        return result



    def set_k1_k2_n(self, k1, k2, n):
        r"""Set the :math:`n`, :math:`k_1`, and :math:`k_2` in the bath
        influence function 
        :math:`I_{\nu; r; n; k_1; k_2}^{(\mathrm{2-pt bath})}
        \left(j_{r; k_1}, j_{r; k_2}\right)` [see constructor documentation for
        the definition of the bath influence function].
        """
        K_tau = self.K_tau
        K_v_tau = self.K_v_tau
        k_v_n = self.eval_k_v_n(n)

        if n == 1:
            if (k2 == 0) and (k1 == 0):
                # Using Eq. (594) of DM.
                selected_eta_cache = self.eta_cache_6[K_v_tau-1]
            elif (k2 == 1) and (k1 == 0):
                # Using Eq. (595) of DM.
                if k_v_n == 2:
                    selected_eta_cache = self.eta_cache_1[0]
                else:
                    selected_eta_cache = self.eta_cache_6[K_v_tau-2]
            elif (k2 == 1) and (k1 == 1):
                # Using Eq. (596) of DM.
                if k_v_n == 2:
                    selected_eta_cache = self.eta_cache_3[0]
                else:
                    selected_eta_cache = self.eta_cache_6[K_v_tau-1]
            elif (k2 == 2) and (k1 == 0):
                # Using Eq. (597) of DM.
                if k_v_n == 2:
                    selected_eta_cache = self.eta_cache_2[0]
                else:
                    selected_eta_cache = self.eta_cache_1[0]
            elif (k2 == 2) and (k1 == 1):
                # Using Eq. (598) of DM.
                if k_v_n == 2:
                    selected_eta_cache = self.eta_cache_1[0]
                else:
                    selected_eta_cache = self.eta_cache_6[K_v_tau-2]
            elif (k2 == k_v_n-1) and (k1 == k_v_n-1):
                # Using Eq. (599) of DM.
                if k_v_n == 2:
                    selected_eta_cache = self.eta_cache_3[0]
                else:
                    selected_eta_cache = self.eta_cache_6[K_v_tau-1]
            elif (k2 == k_v_n) and (k1 == 0):
                # Using Eq. (600) of DM.
                selected_eta_cache = self.eta_cache_2[0]
            elif (k2 == k_v_n) and (k1 == 1):
                # Using Eq. (601) of DM.
                selected_eta_cache = self.eta_cache_1[0]
            elif (k2 == k_v_n) and (k1 == k_v_n-1):
                # Using Eq. (602) of DM.
                if k_v_n == 2:
                    selected_eta_cache = self.eta_cache_1[0]
                else:
                    selected_eta_cache = self.eta_cache_6[K_v_tau-2]
            elif (k2 == k_v_n) and (k1 == k_v_n):
                # Using Eq. (603) of DM.
                selected_eta_cache = self.eta_cache_6[K_v_tau-1]
        elif ((2 <= n) and (0 <= k2 <= min(K_v_tau-1, k_v_n-2))
              and (k1 == 0)):
            # Using Eq. (604) of DM.
            selected_eta_cache = self.eta_cache_6[K_v_tau-1-k2]
        elif ((2 <= n <= K_tau-1)
              and (k2 == k_v_n-1) and (k1 == 0)):
            # Using Eq. (605) of DM.
            selected_eta_cache = self.eta_cache_1[n-1]
        elif ((2 <= n <= K_tau-2+(K_v_tau/(2*K_tau)))
              and (k2 == k_v_n) and (k1 == 0)):
            # Using Eq. (606) of DM.
            selected_eta_cache = self.eta_cache_2[n-1]
        elif ((2 <= n) and (1 <= k2 <= min(K_v_tau, k_v_n-2))
              and (k1 == 1)):
            # Using Eq. (607) of DM.
            selected_eta_cache = self.eta_cache_5[K_v_tau-k2]
        elif ((2 <= n <= K_tau)
              and (k2 == k_v_n-1) and (k1 == 1)):
            # Using Eq. (608) of DM.
            selected_eta_cache = self.eta_cache_3[n-1]
        elif ((2 <= n <= K_tau-1)
              and (k2 == k_v_n) and (k1 == 1)):
            # Using Eq. (609) of DM.
            selected_eta_cache = self.eta_cache_1[n-1]
        elif ((2 <= n) and (2 <= k2 <= k_v_n-2)
              and (k2-k1 <= K_v_tau-1)):
            # Using Eq. (610) of DM.
            selected_eta_cache = \
                self.eta_cache_4[K_v_tau-1-k2+k1]
        elif ((2 <= n) and (k2 == k_v_n-1)
              and (max(2, k_v_n-K_v_tau) <= k1 <= k_v_n-1)):
            # Using Eq. (611) of DM.
            selected_eta_cache = self.eta_cache_5[K_v_tau-k_v_n+k1]
        elif ((2 <= n) and (k2 == k_v_n)
              and (max(0,k_v_n-K_v_tau+1) <= k1 <= k_v_n)):
            # Using Eq. (612) of DM.
            selected_eta_cache = self.eta_cache_6[K_v_tau-k_v_n-1+k1]

        self.selected_eta_cache_real_part = selected_eta_cache.real
        self.selected_eta_cache_imag_part = selected_eta_cache.imag

        return None



def eval_tilde_k_m(m):
    r"""This function implements :math:`\tilde{k}_m`, which is introduced in
    Eq. (70) of the detailed manuscript of our QUAPI-TN approach.
    """
    if m % 3 == 0:
        result = m//3
    elif m % 3 == 1:
        result = 2*(m//3)
    else:
        result = 2*(m//3)+1

    return result
