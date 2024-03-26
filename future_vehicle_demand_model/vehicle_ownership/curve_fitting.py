import pandas as pd
import numpy as np
from numba import jit, types
from numba.typed import Dict
import numba
import time
from scipy.optimize import least_squares


def vehicle_ownership(gamma_max,D_coeff,U_coeff,UD_coeff,theta_R,theta_F,alpha,beta,D,U,UD,R,F,GDP,V):
    return (gamma_max + D_coeff*D + U_coeff*U + UD_coeff*UD)*(theta_R*R + theta_F*F)*np.exp(alpha*np.exp(beta*GDP))+(1-theta_R*R - theta_F*F)*V


@jit(nopython=True)
def residuals_multicountry_multistep(x,beta_idx,beta_0_idx,CC,year,D,U,UD,R,F,GDP,V_last_in,V):
    '''This function calculates residuals of vehicle ownership for multiple countries at once. 
    Multi-step prediction:
    For consecutive years of the same country, the predictions are used in the next time step rather than the reference value.'''
    beta = x[beta_0_idx+beta_idx]
    res = np.zeros(len(CC),np.float64)
    last_year = np.int64(0.0)
    last_beta_idx = np.int64(-1)
    last_V_pred = np.float64(0.0)

    for i in range(len(CC)):
        

        if last_year > year[i] :
            last_V_pred = V_last_in[i]
        elif last_beta_idx!=beta_idx[i]:
            last_V_pred = V_last_in[i]
            
        V_pred = (x[0] + x[1] * D[i] + x[2]*U[i] + x[3]*UD[i]) * (x[4]*R[i] + x[5]*F[i])*np.exp(x[6]*np.exp(beta[i]*GDP[i]))+(1-x[4]*R[i] - x[5]*F[i])*last_V_pred
        
        last_V_pred = V_pred
        last_year = year[i]
        last_beta_idx = beta_idx[i]


        res[i] = V_pred - V[i]
        

        
    return res


@jit(nopython=True)
def residuals_multicountry_singlestep(x,beta_idx,beta_0_idx,CC,D,U,UD,R,F,GDP,V_last_in,V):
    '''This function calculates residuals of vehicle ownership for multiple countries at once. 
    Single-step prediction:
    For consecutive years of the same country, the reference value is used in the next time step rather than the predicted value.'''
    beta = x[beta_0_idx+beta_idx]
    res = np.zeros(len(CC),np.float64)

    for i in range(len(CC)):

            
        V_pred = (x[0] + x[1] * D[i] + x[2]*U[i] + x[3]*UD[i]) * (x[4]*R[i] + x[5]*F[i])*np.exp(x[6]*np.exp(beta[i]*GDP[i]))+(1-x[4]*R[i] - x[5]*F[i])*V_last_in[i]


        res[i] = V_pred - V[i]  
        
    return res

@jit(nopython=True)
def residuals_singlecountry_multistep(beta,gamma_max,D_coeff,U_coeff,UD_coeff,theta_R,theta_F,alpha,D,U,UD,R,F,GDP,V_last_in,V):
    '''This function calculates residuals of vehicle ownership for one country.
    Multi-step prediction:
    For consecutive years of the same country, the predictions are used in the next time step rather than the reference value.'''
    res = np.zeros(len(D),np.float64)
    V_last = np.float64(0.0)

    for i in range(len(D)):
        if V_last < 1e-8:
            V_last = V_last_in[i]

        V_pred = ((gamma_max + D_coeff * D[i] + U_coeff*U[i] + UD_coeff*UD[i]) 
                  * (theta_R*R[i] + theta_F*F[i])*np.exp(alpha*np.exp(beta[0]*GDP[i]))
                  +(1-theta_R*R[i] - theta_F*F[i])*V_last)

        V_last = V_pred

        res[i] = V_pred - V[i]
        

        
    return res

@jit(nopython=True)
def residuals_singlecountry_singlestep(beta,gamma_max,D_coeff,U_coeff,UD_coeff,theta_R,theta_F,alpha,D,U,UD,R,F,GDP,V_last_in,V):
    res = np.zeros(len(D),np.float64)

    for i in range(len(D)):

        V_pred = ((gamma_max + D_coeff * D[i] + U_coeff*U[i] + UD_coeff*UD[i]) 
                  * (theta_R*R[i] + theta_F*F[i])*np.exp(alpha*np.exp(beta[0]*GDP[i]))
                  +(1-theta_R*R[i] - theta_F*F[i])*V_last_in[i])


        res[i] = V_pred - V[i]
        

        
    return res