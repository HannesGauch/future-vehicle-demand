import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

MainPath = os.path.join(dir_path, 'odym', 'modules')

import future_vehicle_demand_model.stock_model.odym.modules.dynamic_stock_model as dsm 
import numpy as np
import matplotlib.pyplot as plt
import openpyxl


def vehicle_stock_model(years,population,vehicle_ownership,car_lifetime,bev_share):

    TotalVehicleStock = [a*b/1000 for (a,b) in zip(population,vehicle_ownership)] # unit: 1

    GlobalVehicles_DSM = dsm.DynamicStockModel(t = years, s = TotalVehicleStock, 
                                        lt = {'Type': 'Normal', 'Mean': np.array(car_lifetime), 
                                                'StdDev': 0.3*np.array(car_lifetime) })
    #CheckStr = GlobalVehicles_DSM.dimension_check()

    S_C, O_C, I = GlobalVehicles_DSM.compute_stock_driven_model()
    # S_C: Stock by cohort
    # O_C: Outflow by cohort
    # I: inflow (new registration) of cars

    O   = GlobalVehicles_DSM.compute_outflow_total() # Total outflow
    DS  = GlobalVehicles_DSM.compute_stock_change()  # Stock change
    Bal = GlobalVehicles_DSM.check_stock_balance()   # Vehicle balance

    #print(np.abs(Bal).sum()) # show sum absolute of all mass balance mismatches.

    #make individual models for BEVs and ICEs 
    GlobalICEs_DSM = dsm.DynamicStockModel(t = years, i = I*(1-np.array(bev_share)), 
                                        lt = {'Type': 'Normal', 'Mean': np.array(car_lifetime), 
                                                'StdDev': 0.3*np.array(car_lifetime) })
    GlobalICEs_DSM.compute_s_c_inflow_driven()
    GlobalICEs_DSM.compute_o_c_from_s_c()
    GlobalICEs_DSM.compute_stock_total()
    GlobalICEs_DSM.compute_stock_change()

    GlobalBEVs_DSM = dsm.DynamicStockModel(t = years, i = I*(np.array(bev_share)), 
                                        lt = {'Type': 'Normal', 'Mean': np.array(car_lifetime), 
                                                'StdDev': 0.3*np.array(car_lifetime) })
    GlobalBEVs_DSM.compute_s_c_inflow_driven()
    GlobalBEVs_DSM.compute_o_c_from_s_c()
    GlobalBEVs_DSM.compute_stock_total()
    GlobalBEVs_DSM.compute_stock_change()

    results = {"years":years,"ICE":{"stock":GlobalICEs_DSM.s,"inflow":GlobalICEs_DSM.i,"outflow":GlobalICEs_DSM.o},
                            "BEV":{"stock":GlobalBEVs_DSM.s,"inflow":GlobalBEVs_DSM.i,"outflow":GlobalBEVs_DSM.o},
                            "total":{"stock":GlobalVehicles_DSM.s,"inflow":GlobalVehicles_DSM.i,"outflow":GlobalVehicles_DSM.o}}
    return results

