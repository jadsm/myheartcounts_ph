# Utils (tools) to compute statistical metrics on time series

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from itertools import combinations
import statsmodels.api as sm

def stat_feature_extraction(df,colname=0): 
    FEATURES = ['N','MIN','MAX','MEAN','RMS','VAR','STD','POWER','PEAK','P2P','CREST FACTOR','SKEW','KURTOSIS']
    
    N =[];Min=[];Max=[];Mean=[];Rms=[];Var=[];Std=[];Power=[];Peak=[];Skew=[];Kurtosis=[];P2p=[];CrestFactor=[]
    FormFactor=[]; PulseIndicator=[]
    
    X = df[colname].values
    
    ## TIME DOMAIN ##
    N.append(X.shape[0])
    Min.append(np.min(X))
    Max.append(np.max(X))
    Mean.append(np.mean(X))
    Rms.append(np.sqrt(np.mean(X**2)))
    Var.append(np.var(X))
    Std.append(np.std(X))
    Power.append(np.mean(X**2))
    Peak.append(np.max(np.abs(X)))
    P2p.append(np.ptp(X))
    CrestFactor.append(np.max(np.abs(X))/np.sqrt(np.mean(X**2)))
    Skew.append(stats.skew(X))
    Kurtosis.append(stats.kurtosis(X))
    FormFactor.append(np.sqrt(np.mean(X**2))/np.mean(X))
    PulseIndicator.append(np.max(np.abs(X))/np.mean(X))
    
    # Create dataframe from features
    df_features = pd.DataFrame(index = [FEATURES], 
                               data = [N,Min,Max,Mean,Rms,Var,Std,Power,Peak,P2p,CrestFactor,Skew,Kurtosis],
                              columns=[colname]).reset_index().rename(columns={"level_0":"features"})

    return df_features

def fft_feature_extraction(df,colname=0): 
    FEATURES = ['MAX_f','SUM_f','MEAN_f','VAR_f','PEAK_f','SKEW_f','KURTOSIS_f']
    

    Max_f=[];Sum_f=[];Mean_f=[];Var_f=[];Peak_f=[];Skew_f=[];Kurtosis_f=[]
    
    X = df[colname].values
  
    ## FREQ DOMAIN ##
    ft = np.fft.fft(X)
    S = np.abs(ft**2)/len(df)
    Max_f.append(np.max(S))
    Sum_f.append(np.sum(S))
    Mean_f.append(np.mean(S))
    Var_f.append(np.var(S))
    
    Peak_f.append(np.max(np.abs(S)))
    Skew_f.append(stats.skew(X))
    Kurtosis_f.append(stats.kurtosis(X))
    
    # Create dataframe from features
    df_features = pd.DataFrame(index = [FEATURES], 
                               data = [Max_f,Sum_f,Mean_f,Var_f,Peak_f,Skew_f,Kurtosis_f],
                              columns=[colname]).reset_index().rename(columns={"level_0":"features"})

    return df_features



def ARIMA_feature_extraction(i,v,order = (2,1,0),plot_flag=False,verbose=True):
    # fit model
    model = ARIMA(v["value"], order=order)
    model_fit = model.fit()
    # line plot of residuals
    residuals = pd.DataFrame(model_fit.resid)
    if plot_flag:
        residuals.plot()
        plt.show()
        # density plot of residuals
        residuals.plot(kind='kde')
        plt.show()
    # summary stats of residuals
    # summary of fit model
    if verbose:
        print(i,model_fit.summary())
    
        print(i,residuals.describe())
    
    return model_fit.params.reset_index().rename(columns={"index":"features",0:"value"})


# cross correlation between each combination of variables
def CC_feature_extraction(df_):
    combo = list(combinations(df_["variable"].unique(), 2))
    lags = 3
    OUT = []
    for i,v in df_.groupby(["patient"]):
        for c in combo:
            cc = sm.tsa.stattools.ccf(df_.query(f'variable == "{c[0]}"')["value"],
                                df_.query(f'variable == "{c[1]}"')["value"], adjusted=False)
            aux3 = pd.DataFrame([[cc[ii],"cc_lag"+str(ii)] for ii in range(lags)],columns=["value","features"])
            aux3['variable'] = "_".join(c)
            aux3["patient"] = i[0]
            OUT.append(aux3)
    OUT = pd.concat(OUT,axis=0)
    return OUT
