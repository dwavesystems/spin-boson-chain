#!/usr/bin/env python
r"""This module contains a class representing the eta-function that appears in
our QUAPI formalism, upon which ``sbc`` is based.
"""



#####################################
## Load libraries/packages/modules ##
#####################################

# Import factorial function.
from math import factorial



# For evaluating special math functions and general array handling.
import numpy as np

# For evaluating numerically integrals.
import scipy.integrate



# For importing trivial instances of the ``sbc.bath.SpectralDensity`` class that
# always evaluates to zero.
import sbc.bath



############################
## Authorship information ##
############################

__author__ = "Matthew Fitzpatrick"
__copyright__ = "Copyright 2021"
__credits__ = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__ = "mfitzpatrick@dwavesys.com"
__status__ = "Development"



##################################
## Define classes and functions ##
##################################

class Eta():
    def __init__(self, r, bath_model, dt, spin_basis):
        if spin_basis == "y":
            if bath_model.y_spectral_densities == None:
                self.A_v_r_T = sbc.bath._trivial_spectral_density
            else:
                self.A_v_r_T = bath_model.y_spectral_densities[r]
        elif spin_basis == "z":
            if bath_model.z_spectral_densities == None:
                self.A_v_r_T = sbc.bath._trivial_spectral_density
            else:
                self.A_v_r_T = bath_model.z_spectral_densities[r]
        self.dt = dt
        self.min_limit = 2000  # Smallest `limit` used in `quad` in this module.

        return None



    def eval(self, l1, l2, n):
        # DM: Detailed manuscript.

        # quad_weights is given by Eq. (590) of DM; self.W_vars is given by
        # Eq. (637)-(640) of DM.
        quad_weights = [0.25, 0.25] + [0.50]*(2*n-2) + [0.25, 0.25]
        self.W_vars = [self.dt * sum(quad_weights[l2:(l1-1)+1]),
                       self.dt * sum(quad_weights[l2+1:l1+1]),
                       self.dt * sum(quad_weights[l2+1:(l1-1)+1]),
                       self.dt * sum(quad_weights[l2:l1+1])]

        # Evaluating Eq. (654) of DM.
        result = 0.0j
        for A_v_r_T_cmpnt in self.A_v_r_T.cmpnts:
            self.update_integration_pts(A_v_r_T_cmpnt)
            for a in (0, 1, 2, 3, 5, 6, 7, 8):
                result += self.eval_real_cmpnt(A_v_r_T_cmpnt, a)
                result += 1j*self.eval_imag_cmpnt(l1, l2, A_v_r_T_cmpnt, a)
        
        return result



    def update_integration_pts(self, A_v_r_T_cmpnt):
        # DM: Detailed manuscript.

        # W_var_max is given by Eq. (650) of DM.
        W_var_max = max(self.W_vars)
        uv_cutoff = A_v_r_T_cmpnt.limit_0T.uv_cutoff
        ir_cutoff = A_v_r_T_cmpnt.limit_0T.ir_cutoff
        beta = A_v_r_T_cmpnt.beta

        # pt_A and pt_B are given by Eqs. (665) and (666) of DM respectively.
        pt_A = -np.pi / W_var_max if W_var_max != 0 else -np.inf
        pt_B = -25 / beta

        # pts is given by Eqs. (659)-(664) of DM respectively.
        pts = [0.0] * 10
        pts[0] = -uv_cutoff
        pts[2] = min(max(-uv_cutoff, pt_A), -ir_cutoff)
        pts[4] = -ir_cutoff
        pts[1] = pt_B if pts[0] < pt_B < pts[2] else pts[0]
        pts[3] = pt_B if pts[2] < pt_B < pts[4] else pts[4]
        pts[5] = -pts[4]
        pts[6] = -pts[3]
        pts[7] = -pts[2]
        pts[8] = -pts[1]
        pts[9] = -pts[0]
        self.integration_pts = pts
                
        return None



    def eval_real_cmpnt(self, A_v_r_T_cmpnt, a):
        # DM: Detailed manuscript.
        
        if a in (2, 3, 5, 6):
            # Evaluate Eq. (655) of DM for given a.
            result = self.eval_integral_type_R1(A_v_r_T_cmpnt, a)
        else:
            # Evaluate Eq. (657) of DM for given a.
            result = 0.0
            for j in range(0, 4):
                # Evaluate Eq. (671) of DM for given a and j.
                result += self.eval_integral_type_R2(A_v_r_T_cmpnt, a, j)

        return result



    def eval_imag_cmpnt(self, l1, l2, A_v_r_T_cmpnt, a):
        # DM: Detailed manuscript.
        
        if a in (2, 3, 5, 6):
            # Evaluate Eq. (656) of DM for given a.
            result = self.eval_integral_type_I1(l1, l2, A_v_r_T_cmpnt, a)
        else:
            # Evaluate Eq. (658) of DM for given a.
            result = 0.0
            for j in range(0, 4):
                # Evaluate Eq. (672) of DM for given a and j.
                result += self.eval_integral_type_I2(l1, l2,
                                                     A_v_r_T_cmpnt, a, j)

        return result



    def eval_integral_type_R1(self, A_v_r_T_cmpnt, a):
        # DM: Detailed manuscript.

        # W_vars is given by Eqs. (637)-(640) of DM; W_var_max is given by
        # Eq. (650) of DM.
        W_vars = self.W_vars
        W_var_max = max(W_vars)
        pi = np.pi

        def summand(omega, m):
            return ((-1)**m / factorial(2*m)
                    * (W_vars[0]**2 * (omega*W_vars[0])**(2*m-2)
                       + W_vars[1]**2 * (omega*W_vars[1])**(2*m-2)
                       - W_vars[2]**2 * (omega*W_vars[2])**(2*m-2)
                       - W_vars[3]**2 * (omega*W_vars[3])**(2*m-2)))

        def F(omega):
            if abs(omega * W_var_max) < 1.0e-3:
                # Evaluate Eq. (667) of DM.
                return sum([summand(omega, m) for m in range(1, 5)])
            else:
                # Evaluate Eq. (668) of DM.
                return ((np.cos(W_vars[0]*omega) + np.cos(W_vars[1]*omega)
                         - np.cos(W_vars[2]*omega) - np.cos(W_vars[3]*omega))
                        / omega / omega)

        # Evaluate Eq. (655) of DM; see also paragraph above Eq. (673) of DM.
        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        integrand = lambda omega: (A_v_r_T_cmpnt._eval(omega)
                                   * F(omega) / 2.0 / pi)
        quad = scipy.integrate.quad
        result = quad(integrand, a=pt1, b=pt2, limit=self.min_limit)[0]

        return result
            


    def eval_integral_type_R2(self, A_v_r_T_cmpnt, a, j):
        # DM: Detailed manuscript.

        # W_vars is given by Eqs. (637)-(640) of DM.
        W_var = self.W_vars[j]
        sign_prefactor = (-1)**(j//2)
        pi = np.pi

        # Evaluate Eq. (671) of DM; see also paragraph above Eq. (673) of DM.
        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        integrand = lambda omega: (sign_prefactor * A_v_r_T_cmpnt._eval(omega)
                                   / omega / omega / 2.0 / pi)
        quad = scipy.integrate.quad
        if W_var == 0:
            result = quad(integrand, a=pt1, b=pt2, limit=self.min_limit)[0]
        else:
            result = quad(integrand, a=pt1, b=pt2, weight="cos", wvar=W_var,
                          limit=self.min_limit*int(abs((pt2-pt1)/pt1)+1))[0]

        return result



    def eval_integral_type_I1(self, l1, l2, A_v_r_T_cmpnt, a):
        # DM: Detailed manuscript.
        
        if (l1 == l2):
            # See Eqs. (656), (669) and (670) of DM.
            result = 0.0
            return result

        # W_vars is given by Eqs. (637)-(640) of DM; W_var_max is given by
        # Eq. (650) of DM.
        W_vars = self.W_vars
        W_var_max = max(W_vars)
        pi = np.pi

        def summand(omega, m):
            return ((-1)**m / factorial(2*m+1)
                    * (W_vars[0]**2 * (omega*W_vars[0])**(2*m-1)
                       + W_vars[1]**2 * (omega*W_vars[1])**(2*m-1)
                       - W_vars[2]**2 * (omega*W_vars[2])**(2*m-1)
                       - W_vars[3]**2 * (omega*W_vars[3])**(2*m-1)))
        
        def F(omega):
            if abs(omega * W_var_max) < 1.0e-3:
                # Evaluate Eq. (669) of DM.
                return -sum([summand(omega, m) for m in range(1, 5)])
            else:
                # Evaluate Eq. (670) of DM.
                return ((-np.sin(W_vars[0]*omega) - np.sin(W_vars[1]*omega)
                         + np.sin(W_vars[2]*omega) + np.sin(W_vars[3]*omega))
                        / omega / omega)

        # Evaluate Eq. (656) of DM; see also paragraph above Eq. (673) of DM.
        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        integrand = lambda omega: (A_v_r_T_cmpnt._eval(omega)
                                   * F(omega) / 2.0 / pi)
        quad = scipy.integrate.quad
        result = quad(integrand, a=pt1, b=pt2, limit=self.min_limit)[0]

        return result
            


    def eval_integral_type_I2(self, l1, l2, A_v_r_T_cmpnt, a, j):
        # DM: Detailed manuscript.

        # W_vars is given by Eqs. (637)-(640) of DM.
        W_var = self.W_vars[j]
        
        if (l1 == l2) or (W_var == 0):
            result = 0.0
            return result
        
        sign_prefactor = -(-1)**(j//2)
        pi = np.pi

        # Evaluate Eq. (672) of DM; see also paragraph above Eq. (673) of DM.
        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        integrand = lambda omega: (sign_prefactor * A_v_r_T_cmpnt._eval(omega)
                                   / omega / omega / 2.0 / pi)
        quad = scipy.integrate.quad
        result = quad(integrand, a=pt1, b=pt2, weight="sin", wvar=W_var,
                      limit=self.min_limit*int(abs((pt2-pt1)/pt1)+1))[0]

        return result
