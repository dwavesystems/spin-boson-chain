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
from scipy.integrate import quad



# Import a trivial instance of the ``sbc.bath.SpectralDensity`` class that
# always evaluates to zero.
from sbc.bath import _trivial_spectral_density



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
    def __init__(self, r, bath_model, dt, spin_basis):
        if spin_basis == "y":
            if bath_model.y_spectral_densities == None:
                self.A_v_r_T = _trivial_spectral_density
            else:
                self.A_v_r_T = bath_model.y_spectral_densities[r]
        elif spin_basis == "z":
            if bath_model.z_spectral_densities == None:
                self.A_v_r_T = _trivial_spectral_density
            else:
                self.A_v_r_T = bath_model.z_spectral_densities[r]
        self.dt = dt
        self.min_limit = 2000  # Smallest `limit` used in `quad` in this module.

        return None



    def eval(self, l1, l2, n):
        quad_weights = [0.25, 0.25] + [0.50]*(2*n-2) + [0.25, 0.25]
        self.W_vars = [self.dt * sum(quad_weights[l2:(l1-1)+1]),
                       self.dt * sum(quad_weights[l2+1:l1+1]),
                       self.dt * sum(quad_weights[l2+1:(l1-1)+1]),
                       self.dt * sum(quad_weights[l2:l1+1])]

        result = 0.0j
        for A_v_r_T_cmpnt in self.A_v_r_T.cmpnts:
            self.update_integration_pts(A_v_r_T_cmpnt)
            for a in range(0, 4):
                result += self.eval_real_cmpnt(A_v_r_T_cmpnt, a)
                result += 1j*self.eval_imag_cmpnt(l1, l2, A_v_r_T_cmpnt, a)
        
        return result



    def update_integration_pts(self, A_v_r_T_cmpnt):
        W_var_max = max(self.W_vars)
        
        self.integration_pts = [0.0]*5
        self.integration_pts[0] = -A_v_r_T_cmpnt.limit_0T.hard_cutoff_freq

        if W_var_max != 0.0:
            self.integration_pts[1] = \
                max(self.integration_pts[0], -np.pi / W_var_max)
        else:
            self.integration_pts[1] = self.integration_pts[0]
            
        self.integration_pts[2] = 0.0
        self.integration_pts[3] = -self.integration_pts[1]
        self.integration_pts[4] = -self.integration_pts[0]

        return None



    def eval_real_cmpnt(self, A_v_r_T_cmpnt, a):
        if (a == 1) or (a == 2):
            result = self.eval_integral_type_R1(A_v_r_T_cmpnt, a)
        else:
            result = 0.0
            for u in range(0, 4):
                result += self.eval_integral_type_R2(A_v_r_T_cmpnt, a, u)

        return result



    def eval_imag_cmpnt(self, l1, l2, A_v_r_T_cmpnt, a):
        if (a == 1) or (a == 2):
            result = self.eval_integral_type_I1(l1, l2, A_v_r_T_cmpnt, a)
        else:
            result = 0.0
            for u in range(0, 4):
                result += self.eval_integral_type_I2(l1, l2,
                                                     A_v_r_T_cmpnt, a, u)

        return result



    def eval_integral_type_R1(self, A_v_r_T_cmpnt, a):
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
                return sum([summand(omega, m) for m in range(1, 5)])
            else:
                return ((np.cos(W_vars[0]*omega) + np.cos(W_vars[1]*omega)
                         - np.cos(W_vars[2]*omega) - np.cos(W_vars[3]*omega))
                        / omega / omega)

        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        integrand = lambda omega: (A_v_r_T_cmpnt._eval(omega)
                                   * F(omega) / 2.0 / pi)
        result = quad(integrand, a=pt1, b=pt2, limit=self.min_limit)[0]

        return result
            


    def eval_integral_type_R2(self, A_v_r_T_cmpnt, a, u):
        W_var = self.W_vars[u]
        sign_prefactor = (-1)**(u//2)
        pi = np.pi
        
        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        integrand = lambda omega: (sign_prefactor * A_v_r_T_cmpnt._eval(omega)
                                   / omega / omega / 2.0 / pi)
        if W_var == 0:
            result = quad(integrand, a=pt1, b=pt2, limit=self.min_limit)[0]
        else:
            result = quad(integrand, a=pt1, b=pt2, weight="cos", wvar=W_var,
                          limit=self.min_limit*int(abs((pt2-pt1)/pt1)+1))[0]

        return result



    def eval_integral_type_I1(self, l1, l2, A_v_r_T_cmpnt, a):
        if (l1 == l2):
            result = 0.0
            return result
        
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
                return -sum([summand(omega, m) for m in range(1, 5)])
            else:
                return ((-np.sin(W_vars[0]*omega) - np.sin(W_vars[1]*omega)
                         + np.sin(W_vars[2]*omega) + np.sin(W_vars[3]*omega))
                        / omega / omega)

        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        integrand = lambda omega: (A_v_r_T_cmpnt._eval(omega)
                                   * F(omega) / 2.0 / pi)
        result = quad(integrand, a=pt1, b=pt2, limit=self.min_limit)[0]

        return result
            


    def eval_integral_type_I2(self, l1, l2, A_v_r_T_cmpnt, a, u):
        W_var = self.W_vars[u]
        
        if (l1 == l2) or (W_var == 0):
            result = 0.0
            return result
        
        sign_prefactor = -(-1)**(u//2)
        pi = np.pi

        pt1 = self.integration_pts[a]
        pt2 = self.integration_pts[a+1]
        integrand = lambda omega: (sign_prefactor * A_v_r_T_cmpnt._eval(omega)
                                   / omega / omega / 2.0 / pi)
        result = quad(integrand, a=pt1, b=pt2, weight="sin", wvar=W_var,
                      limit=self.min_limit*int(abs((pt2-pt1)/pt1)+1))[0]

        return result
