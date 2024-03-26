# -*- coding: utf-8 -*-
"""
Based on https://github.com/IndEcol/ODYM 

Specialised for stock-driven model with typesplit 

"""

import numpy as np
import scipy.stats

def __version__():
    """Return a brief version string and statement for this class."""
    return str('1.0'), str('Class DynamicStockModel, dsm. Version 1.0. Last change: July 25th, 2019. Check https://github.com/IndEcol/ODYM for latest version.')


class DynamicStockModel(object):

    """ Class containing a dynamic stock model

    Attributes
    ----------
    t : Series of years or other time intervals
    i : Discrete time series of inflow to stock

    o : Discrete time series of outflow from stock
    o_c :Discrete time series of outflow from stock, by cohort

    s_c : dynamic stock model (stock broken down by year and age- cohort)
    s : Discrete time series for stock, total


    name : string, optional
        Name of the dynamic stock model, default is 'DSM'
    """

    """
    Basic initialisation and dimension check methods
    """

    def __init__(self, t=None, name='DSM'):
        """ Init function. Assign the input data to the instance of the object."""
        self.t = t  # optional

        self.i = None  # optional
        self.i_g = None 


        self.s = None  # optional
        self.s_c = None  # optional
        self.s_cg = None 


        self.o = None  
        self.o_c = None 
        self.o_cg = None


        self.name = name  # optional

    """ Part 1: Checks and balances: """

    def compute_stock_change(self):
        """ Determine stock change from time series for stock. Formula: stock_change(t) = stock(t) - stock(t-1)."""
        if self.s is not None:
            stock_change = np.zeros(len(self.s))
            stock_change[0] = self.s[0]
            stock_change[1::] = np.diff(self.s)
            return stock_change
        else:
            return None

    def check_stock_balance(self):
        """ Check wether inflow, outflow, and stock are balanced. If possible, the method returns the vector 'Balance', where Balance = inflow - outflow - stock_change"""
        try:
            Balance = self.i - self.o - self.compute_stock_change()
            return Balance
        except:
            # Could not determine balance. At least one of the variables is not defined.
            return None

    def compute_stock_total(self):
        """Determine total stock as row sum of cohort-specific stock."""
        if self.s is not None:
            return self.s
        else:
            try:
                self.s = self.s_c.sum(axis=1)
                return self.s
            except:
                return None # No stock by cohorts exists, and total stock cannot be computed

    def compute_outflow_total(self):
        """Determine total outflow as row sum of cohort-specific outflow."""
        if self.o is not None:
            # Total outflow is already defined. Doing nothing.
            return self.o
        else:
            try:
                self.o = self.o_c.sum(axis=1)
                return self.o
            except:
                return None # No outflow by cohorts exists, and total outflow cannot be computed
            
    def compute_outflow_mb(self):
        """Compute outflow from process via mass balance. 
           Needed in cases where lifetime is zero."""
        try:
            self.o = self.i - self.compute_stock_change()
            return self.o
        except:
            return None # Variables to compute outflow were not present
 


    def compute_evolution_initialstock(self,InitialStock,SwitchTime):
        """ Assume InitialStock is a vector that contains the age structure of the stock at time t0, 
        and it covers as many historic cohorts as there are elements in it.
        This method then computes the future stock and outflow from the year SwitchTime onwards.
        Only future years, i.e., years after SwitchTime, are computed.
        NOTE: This method ignores and deletes previously calculated s_c and o_c.
        The InitialStock is a vector of the age-cohort composition of the stock at SwitchTime, with length SwitchTime"""
        if self.lt is not None:
            self.s_c = np.zeros((len(self.t), len(self.t)))
            self.o_c = np.zeros((len(self.t), len(self.t)))
            self.compute_sf()
            # Extract and renormalize array describing fate of initialstock:
            Shares_Left = self.sf[SwitchTime,0:SwitchTime].copy()
            self.s_c[SwitchTime,0:SwitchTime] = InitialStock # Add initial stock to s_c
            self.s_c[SwitchTime::,0:SwitchTime] = np.tile(InitialStock.transpose(),(len(self.t)-SwitchTime,1)) * self.sf[SwitchTime::,0:SwitchTime] / np.tile(Shares_Left,(len(self.t)-SwitchTime,1))
        return self.s_c
    
    

    """
    Stock driven model
    Given: total stock, lifetime dist.
    Default order of methods:
    1) determine inflow, outflow by cohort, and stock by cohort
    2) determine total outflow
    3) determine stock change
    4) check mass balance.
    """

        
  
    def compute_stock_driven_model_initialstock_typesplit(self,FutureStock,InitialStock,SFArrayCombined,TypeSplit):
        """ 
        With given total future stock and lifetime distribution, the method builds the stock by cohort and the inflow.
        The age structure of the initial stock is given for each technology, and a type split of total inflow into different technology types is given as well.
        
        SPECIFICATION: Stocks are always measured AT THE END of the discrete time interval.
        
        Indices:
          t: time: Entire time frame: from earliest age-cohort to latest model year.
          c: age-cohort: same as time.
          T: Switch time: DEFINED as first year where historic stock is NOT present, = last year where historic stock is present +1.
             Switchtime is calculated internally, by subtracting the length of the historic stock from the total model length.
          g: product type
        
        Data:
          FutureStock[t],           total future stock at end of each year, starting at T
          InitialStock[c,g],        0...T-1;0...T-1, stock at the end of T-1, by age-cohort c, ranging from 0...T-1, and product type g
                                    c-dimension has full length, all future years must be 0.
          SFArrayCombined[t,c,g],   Survival function of age-cohort c at end of year t for product type g
                                    this array spans both historic and future age-cohorts
          Typesplit[t,g],           splits total inflow into product types for future years 
            
        The extra parameter InitialStock is a vector that contains the age structure of the stock at time t0, and it covers as many historic cohorts as there are elements in it.
        In the year SwitchTime the model switches from the historic stock to the stock-driven approach.
        Only future years, i.e., years after SwitchTime, are computed and returned.
        The InitialStock is a vector of the age-cohort composition of the stock at SwitchTime, with length SwitchTime.
        The parameter TypeSplit splits the total inflow into Ng types. """
        
                
        SwitchTime = SFArrayCombined.shape[0] - FutureStock.shape[0]
        Ntt        = SFArrayCombined.shape[0] # Total no of years
        Nt0        = FutureStock.shape[0]     # No of future years
        Ng         = SFArrayCombined.shape[2] # No of product groups

        if self.t:
            assert len(self.t)==Ntt, "SFArray not of same length as time vector" 
        
        s_cg = np.zeros((Nt0,Ntt,Ng)) # stock for future years, all age-cohorts and product
        o_cg = np.zeros((Nt0,Ntt,Ng)) # outflow by future years, all cohorts and products
        i_g  = np.zeros((Ntt,Ng))     # inflow by product

        # Construct historic inflows
        for c in range(0,SwitchTime): # for all historic age-cohorts til SwitchTime - 1:
            for g in range(0,Ng):
                if SFArrayCombined[SwitchTime-1,c,g] != 0:
                    i_g[c,g] = InitialStock[c,g] / SFArrayCombined[SwitchTime-1,c,g]
                    
                    # if InitialStock is 0, historic inflow also remains 0, 
                    # as it has no impact on future anymore.
                    
                    # If survival function is 0 but initial stock is not, the data are inconsisent and need to be revised.
                    # For example, a safety-relevant device with 5 years fixed lifetime but a 10 year old device is present.
                    # Such items will be ignored and break the mass balance.
    
        # year-by-year computation, starting from SwitchTime
        for t in range(SwitchTime, Ntt):  # for all years t, starting at SwitchTime
            # 1) Compute stock at the end of the year:
            s_cg[t - SwitchTime,:,:] = np.einsum('cg,cg->cg',i_g,SFArrayCombined[t,:,:])
            # 2) Compute outflow during year t from previous age-cohorts:
            if t == SwitchTime:
                o_cg[t -SwitchTime,:,:] = InitialStock - s_cg[t -SwitchTime,:,:]
            else:
                o_cg[t -SwitchTime,:,:] = s_cg[t -SwitchTime -1,:,:] - s_cg[t -SwitchTime,:,:] # outflow table is filled row-wise, for each year t.
            # 3) Determine total inflow from mass balance:
            i0 = FutureStock[t -SwitchTime] - s_cg[t - SwitchTime,:,:].sum()
            # 4) Add new inflow to stock and determine future decay of new age-cohort
            i_g[t,:] = TypeSplit[t -SwitchTime,:] * i0
            for g in range(0,Ng): # Correct for share of inflow leaving during first year.
                if SFArrayCombined[t,t,g] != 0: # Else, inflow leaves within the same year and stock modelling is useless
                    i_g[t,g] = i_g[t,g] / SFArrayCombined[t,t,g] # allow for outflow during first year by rescaling with 1/SF[t,t,g]
                s_cg[t -SwitchTime,t,g]  = i_g[t,g] * SFArrayCombined[t,t,g]
                o_cg[t -SwitchTime,t,g]  = i_g[t,g] * (1 - SFArrayCombined[t,t,g])
            
        # Add total values of parameter to enable mass balance check:
        self.s_c = s_cg.sum(axis =2)
        self.o_c = o_cg.sum(axis =2)
        self.i   =  i_g[SwitchTime::,:].sum(axis =1)
        
        return s_cg, o_cg, i_g

    
        
    def compute_stock_driven_model_initialstock_typesplit_negativeinflowcorrect(self,SwitchTime,InitialStock,SFArrayCombined,TypeSplit,NegativeInflowCorrect = False):
        """ 
        With given total future stock and lifetime distribution, the method builds the stock by cohort and the inflow.
        The age structure of the initial stock is given for each technology, and a type split of total inflow into different technology types is given as well.
        For the option "NegativeInflowCorrect", see the explanations for the method compute_stock_driven_model(self, NegativeInflowCorrect = True).
        NegativeInflowCorrect only affects the future stock time series and works exactly as for the stock-driven model without initial stock.
        
        SPECIFICATION: Stocks are always measured AT THE END of the discrete time interval.
        
        Indices:
          t: time: Entire time frame: from earliest age-cohort to latest model year.
          c: age-cohort: same as time.
          T: Switch time: DEFINED as first year where historic stock is NOT present, = last year where historic stock is present +1.
             Switchtime must be given as argument. Example: if the first three age-cohorts are historic, SwitchTime is 3, which indicates the 4th year.
             That also means that the first 3 time-entries for the stock and typesplit arrays must be 0.
          g: product type
        
        Data:
          s[t],                     total future stock time series, at end of each year, starting at T, trailing 0s for historic years.
                                    ! is not handed over with the function call but earlier, when defining the dsm.
          InitialStock[c,g],        0...T-1;0...T-1, stock at the end of T-1, by age-cohort c, ranging from 0...T-1, and product type g
                                    c-dimension has full length, all future years must be 0.
          SFArrayCombined[t,c,g],   Survival function of age-cohort c at end of year t for product type g
                                    this array spans both historic and future age-cohorts
          Typesplit[t,g],           splits total inflow into product types for future years 
          NegativeInflowCorrect     BOOL, retains items in stock if their leaving would lead to negative inflows. 
            
        The extra parameter InitialStock is a vector that contains the age structure of the stock at time t0, and it covers as many historic cohorts as there are elements in it.
        In the year SwitchTime the model switches from the historic stock to the stock-driven approach.
        Only future years, i.e., years after SwitchTime, are computed and returned.
        The InitialStock is a vector of the age-cohort composition of the stock at SwitchTime, with length SwitchTime.
        The parameter TypeSplit splits the total inflow into Ng types. """
        

                
        Ntt        = SFArrayCombined.shape[0] # Total no of years
        Ng         = SFArrayCombined.shape[2] # No of product groups
        
        s_cg = np.zeros((Ntt,Ntt,Ng)) # stock for future years, all age-cohorts and products
        o_cg = np.zeros((Ntt,Ntt,Ng)) # outflow by future years, all cohorts and products
        i_g  = np.zeros((Ntt,Ng))     # inflow for all years by product
        NIC_Flags = np.zeros((Ntt,1)) # inflow flog for future years, will be set to calculated negative inflow value if negative inflow occurs and is corrected for.
        
        self.s_c = np.zeros((len(self.t), len(self.t)))
        self.o_c = np.zeros((len(self.t), len(self.t)))
        self.i   = np.zeros(len(self.t))
        
        # construct the sdf of a product of cohort tc leaving the stock in year t
        self.compute_sf() # Computes sf if not present already.
        # Construct historic inflows
        for c in range(0,SwitchTime): # for all historic age-cohorts til SwitchTime - 1:
            for g in range(0,Ng):
                if SFArrayCombined[SwitchTime-1,c,g] != 0:
                    i_g[c,g] = InitialStock[c,g] / SFArrayCombined[SwitchTime-1,c,g]
                    
                    # if InitialStock is 0, historic inflow also remains 0, 
                    # as it has no impact on future anymore.
                    
                    # If survival function is 0 but initial stock is not, the data are inconsisent and need to be revised.
                    # For example, a safety-relevant device with 5 years fixed lifetime but a 10 year old device is present.
                    # Such items will be ignored and break the mass balance.
                    
        # Compute stocks from historic inflows
        s_cg[:,0:SwitchTime,:] = np.einsum('tcg,cg->tcg',SFArrayCombined[:,0:SwitchTime,:],i_g[0:SwitchTime,:])
        # calculate historic outflows
        for m in range(0,SwitchTime):
            o_cg[m,m,:]      = i_g[m,:] * (1 - SFArrayCombined[m,m,:])
            o_cg[m+1::,m,:]  = s_cg[m:-1,m,:] - s_cg[m+1::,m,:]
        # add historic age-cohorts to total stock:
        self.s[0:SwitchTime] = np.einsum('tcg->t',s_cg[0:SwitchTime,:,:])
        
        # for future: year-by-year computation, starting from SwitchTime
        if NegativeInflowCorrect is False:
            for m in range(SwitchTime, len(self.t)):  # for all years m, starting at SwitchTime
                # 1) Determine inflow from mass balance:
                i0_test = self.s[m] - s_cg[m,:,:].sum()
                if i0_test < 0:
                    NIC_Flags[m] = i0_test
                for g in range(0,Ng):
                    if SFArrayCombined[m,m,g] != 0: # Else, inflow is 0.
                        i_g[m,g] = TypeSplit[m,g] * i0_test / SFArrayCombined[m,m,g] # allow for outflow during first year by rescaling with 1/sf[m,m]
                        # NOTE: The stock-driven method may lead to negative inflows, if the stock development is in contradiction with the lifetime model.
                        # In such situations the lifetime assumption must be changed, either by directly using different lifetime values or by adjusting the outlfows, 
                        # cf. the option NegativeInflowCorrect in the method compute_stock_driven_model.
                        # 2) Add new inflow to stock and determine future decay of new age-cohort
                    s_cg[m::,m,g]   = i_g[m,g] * SFArrayCombined[m::,m,g]
                    o_cg[m,m,g]     = i_g[m,g] * (1 - SFArrayCombined[m,m,g])
                    o_cg[m+1::,m,g] = s_cg[m:-1,m,g] - s_cg[m+1::,m,g]
                    
        if NegativeInflowCorrect is True:
            for m in range(SwitchTime, len(self.t)):  # for all years m, starting at SwitchTime
                # 1) Determine inflow from mass balance:
                i0_test = self.s[m] - s_cg[m,:,:].sum()
                if i0_test < 0:
                    NIC_Flags[m] = i0_test                        
                    Delta = -1 * i0_test # Delta > 0!
                    i_g[m,:] = 0 # Set inflow to 0 and distribute mass balance gap onto remaining cohorts:
                    if s_cg[m,:,:].sum() != 0:
                        Delta_percent = Delta / s_cg[m,:,:].sum() 
                        # Distribute gap equally across all cohorts (each cohort is adjusted by the same %, based on surplus with regards to the prescribed stock)
                        # Delta_percent is a % value <= 100%
                    else:
                        Delta_percent = 0 # stock in this year is already zero, method does not work in this case.
                    # correct for outflow and stock in current and future years
                    # adjust the entire stock AFTER year m as well, stock is lowered in year m, so future cohort survival also needs to decrease.
                    o_cg[m, :,:]    = o_cg[m, :,:]    + (s_cg[m, :,:] * Delta_percent).copy()  # increase outflow according to the lost fraction of the stock, based on Delta_c
                    s_cg[m::,0:m,:] = s_cg[m::,0:m,:] * (1-Delta_percent.copy())               # shrink future description of stock from previous age-cohorts by factor Delta_percent in current AND future years.
                    o_cg[m+1::,:,:] = s_cg[m:-1,:,:] - s_cg[m+1::,:,:]                         # recalculate future outflows
                
                else:       
                    for g in range(0,Ng):
                        if SFArrayCombined[m,m,g] != 0: # Else, inflow is 0.
                            i_g[m,g] = TypeSplit[m,g] * i0_test / SFArrayCombined[m,m,g] # allow for outflow during first year by rescaling with 1/sf[m,m]
                            # NOTE: The stock-driven method may lead to negative inflows, if the stock development is in contradiction with the lifetime model.
                            # In such situations the lifetime assumption must be changed, either by directly using different lifetime values or by adjusting the outlfows, 
                            # cf. the option NegativeInflowCorrect in the method compute_stock_driven_model.
                            # 2) Add new inflow to stock and determine future decay of new age-cohort
                        s_cg[m::,m,g]   = i_g[m,g] * SFArrayCombined[m::,m,g]
                        o_cg[m,m,g]     = i_g[m,g] * (1 - SFArrayCombined[m,m,g])
                        o_cg[m+1::,m,g] = s_cg[m:-1,m,g] - s_cg[m+1::,m,g]    
                        
        # Add total values of parameter to enable mass balance check:
        self.s_c = s_cg.sum(axis =2)
        self.o_c = o_cg.sum(axis =2)
        self.i   = i_g.sum(axis =1)
        
        return s_cg, o_cg, i_g, NIC_Flags

      
        

#
#
# The end.
#

