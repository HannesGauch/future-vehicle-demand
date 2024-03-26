import numpy as np
import pickle
import pandas as pd

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

def vehicle_ownership(gamma_max,D_coeff,U_coeff,UD_coeff,theta_R,theta_F,alpha,beta,D,U,UD,R,F,GDP,V):
    return ((gamma_max + D_coeff*D + U_coeff*U + UD_coeff*UD)*(theta_R*R + theta_F*F)
            *np.exp(alpha*np.exp(beta*GDP))+(1-theta_R*R - theta_F*F)*V)

def vehicle_ownership_backward(gamma_max,D_coeff,U_coeff,UD_coeff,theta_R,theta_F,alpha,beta,D,U,UD,R,F,GDP,V_next):
    return ((V_next - ((gamma_max + D_coeff*D + U_coeff*U + UD_coeff*UD)*(theta_R*R + theta_F*F)
            *np.exp(alpha*np.exp(beta*GDP))))/(1-theta_R*R - theta_F*F))

def draw_residual_from_percentiles(residuals):
    x = np.random.rand(1)[0]
    x0 = int(x*100)    
    return residuals[x0] + (residuals[x0+1]-residuals[x0]) * (x*100-x0)

with open(dir_path+"/results/bootstrap_fits_one_step.pickle", "rb") as fp:  
    bootstrap_fits = pickle.load(fp)

with open(dir_path+"/results/prediction_residuals_one_step.pickle", "rb") as fp:   
    residuals = pickle.load(fp)

data = pd.read_csv(dir_path+"/data/consolidated/consolidated_data.csv")

betas_order = {}
counter = 0
for cc in data["Country Code"].sort_values().unique():
    betas_order[cc] = counter
    counter += 1

beta_0_idx = 6

def get_vehicle_ownership_bootstrap_predictions(vehicle_stock_0,country_code,pop_dens,urban_pop_perc,urban_pop_dens,gdp_pc):
    bootstrap_preds = np.zeros((len(gdp_pc),len(bootstrap_fits)))
    bootstrap_preds[0,:] = vehicle_stock_0
    for i in range(1,len(gdp_pc)):
        for j,bf in enumerate(bootstrap_fits):
            bootstrap_preds[i,j] = max(0,vehicle_ownership(*bf[0:7],bf[beta_0_idx+betas_order[country_code]],pop_dens[i],
                                                    urban_pop_perc[i],urban_pop_dens[i],
                                                    gdp_pc[i]>=gdp_pc[i-1],gdp_pc[i]<gdp_pc[i-1],gdp_pc[i],
                                                    bootstrap_preds[i-1,j])+draw_residual_from_percentiles(residuals))
    return bootstrap_preds
            
def get_vehicle_ownership_bootstrap_predictions_backward(vehicle_stock_0,country_code,pop_dens,urban_pop_perc,urban_pop_dens,gdp_pc):
    bootstrap_preds = np.zeros((len(gdp_pc),len(bootstrap_fits)))
    bootstrap_preds[0,:] = vehicle_stock_0
    for i in range(1,len(gdp_pc)):
        for j,bf in enumerate(bootstrap_fits):
            bootstrap_preds[i,j] = max(0,vehicle_ownership_backward(*bf[0:7],bf[beta_0_idx+betas_order[country_code]],pop_dens[i],
                                                    urban_pop_perc[i],urban_pop_dens[i],
                                                    gdp_pc[i]<=gdp_pc[i-1],gdp_pc[i]>gdp_pc[i-1],gdp_pc[i],
                                                    bootstrap_preds[i-1,j])+draw_residual_from_percentiles(residuals))
            
    return bootstrap_preds
        


