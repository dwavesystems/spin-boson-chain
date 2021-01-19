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
    r"""A class representing the eta-function, :math:`\eta_{\nu; n; k_1; k_2}`, 
    which is introduced in Eqs. (49)-(59) of the detailed manuscript (DM) on our
    QUAPI-TN approach. See Appendix D for details on how we evaluate the 
    eta-function numerically.
    
    Parameters
    ----------
    A_v_T : :class:`sbc.bath.SpectralDensityCmpnt`
        This is the quantity :math:`A_{\nu;T}(\omega)`.
    dt : `float`
        The time step size.
    tilde_w_set : `array_like` (`float`, shape=(4,))
        This is the array :math:`\left(\tilde{w}_{\nu}, \tilde{w}_{\nu; 0},
        \tilde{w}_{\nu; 1}, \tilde{w}_{\nu; 2}\right)`, which is introduced in
        Eqs. (548)-(554) of the DM.
    func_form_of_k_v_n : `func` (`int`)
        The functional form of :math:`k_{\nu; n}`, which appears in Eq. (549)
        of the DM.
    """
    def __init__(self, A_v_T, dt, tilde_w_set, func_form_of_k_v_n):
        self.A_v_T = A_v_T
        self.dt = dt
        self.tilde_w_v = tilde_w_set[0]
        self.tilde_w_v_0 = tilde_w_set[1]
        self.tilde_w_v_1 = tilde_w_set[2]
        self.tilde_w_v_2 = tilde_w_set[3]
        self.func_form_of_k_v_n = func_form_of_k_v_n

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
        k_v_n = self.func_form_of_k_v_n(n)

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
        self.integration_pts[1] = -np.pi / W_var_max
        self.integration_pts[2] = 0.0
        self.integration_pts[3] = np.pi / W_var_max
        self.integration_pts[4] = A_v_T_subcmpnt.limit_0T.hard_cutoff_freq

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
