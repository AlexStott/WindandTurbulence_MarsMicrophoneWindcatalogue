#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train GP for wind speed prediction from Mars Microphone data
For "Wind and turbulence observations with the Mars microphone on Perseverence" Stott et al. (2023) JGR: Planets

@author: a.stott
"""

import pandas as pd  
import numpy as np
import scipy as sp
import seaborn as sns
import os
# import pandas_profiling
#import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib  import cm
from matplotlib.dates import date2num
import re
from pandas.plotting import scatter_matrix

import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.units as munits
import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import MinMaxScaler

import GPy
from sklearn.model_selection import train_test_split
from scipy import signal


import glob

print("libraries imported")



#%% Read in data
NormSet = pd.read_csv("GP_DataSet.csv")

#%% Normalise dataset
minscaler = NormSet.min()
maxscaler = NormSet.max()

print(minscaler)
print(maxscaler)

NormSet = (NormSet-minscaler)/(maxscaler-minscaler)
NormSet = NormSet*2 - 1#%%

#%% Separate data into test and training datasets 

Data = NormSet.to_numpy()


y = Data[:,0].reshape(-1,1)
X = Data[:,1:]

IndNan = ~np.isnan(X)
IndNan = IndNan.all(axis=1)
X= X[IndNan,:]
y=y[IndNan]


ArrayOfVals = np.arange(int(len(y)*0.9))/int(len(y)*0.9)

ArrayOfVals = ArrayOfVals*2 - 1

SampleIndex = np.zeros(len(ArrayOfVals),dtype=int)


y_cycle = y
X_cycle = X

y_train = np.zeros(len(ArrayOfVals))
X_train = np.zeros((len(ArrayOfVals),2))

cnt = 0
for grid in ArrayOfVals:
    AbsDiff = np.absolute(y_cycle-grid)
    Ind = int(AbsDiff.argmin())
    SampleIndex[cnt] = int(Ind)
    y_train[cnt] = y_cycle[Ind]
    X_train[cnt,:] = X_cycle[Ind,:]
    y_cycle = np.delete(y_cycle, Ind, axis=0)
    X_cycle = np.delete(X_cycle, Ind,axis=0)    
    cnt=cnt+1


y_train = y_train.reshape(-1,1)
y_test = y_cycle
X_test = X_cycle


#%% Create GP and train

Q = X.shape[1]

ker = GPy.kern.RBF(Q, ARD=True)

m = GPy.models.GPRegression(X_train,y_train,ker)#KerRMS+KerTemp)#KerDens*KerRMS+KerTemp)

m.optimize(messages=1)
print(m)
print(m.kern.lengthscale)

#%% Load full data table for all files
 
CompleteTable = pd.read_csv("AllRecordingDataTable.csv")

CompleteDuplicate = CompleteTable.copy()


#%%  Cycle through every file and calculate prediction


SaveFiles = 1 #Set to 1 to save wind speeds as csv

SecondTable = CompleteDuplicate.copy()


SecondTable["Gust ML"] = -np.ones(len(SecondTable))
SecondTable["Mic wind STD"] = -np.ones(len(SecondTable))

SecondTable["Mic wind"] = -np.ones((len(SecondTable)))
SecondTable["Mic wind"] = SecondTable["Mic wind"].astype('object')

SecondTable["Wind vec"] = -np.ones((len(SecondTable)))
SecondTable["Wind vec"] = SecondTable["Wind vec"].astype('object')

SecondTable["Mic wind UB"] = -np.ones((len(SecondTable)))
SecondTable["Mic wind UB"] = SecondTable["Mic wind UB"].astype('object')

SecondTable["Mic wind LB"] = -np.ones(len(SecondTable))
SecondTable["Mic wind LB"] = SecondTable["Mic wind LB"].astype('object')




filelist = glob.glob("./CSVdata/*1_OVLP_990.csv")



FullGust = np.zeros(len(filelist))
count=0



for f in filelist:

    Soltouse = pd.read_csv(f)
    SeriesToUse = Soltouse[["WSpeed","RMS20_60","Temp1","Temp2","Temp3","Pressure"]]#,"IncidentWdirection","Pressure","Temp1","Temp2","Temp3","Temp4","Temp5"]]

    FileID = f[10:-19]
    
    CompTableVal = CompleteTable[CompleteTable["Filename"].str.contains(FileID)]

    Temp1 = (SeriesToUse["Temp1"]).to_numpy()
    Temp2 = (SeriesToUse["Temp2"]).to_numpy()
    Temp3 = (SeriesToUse["Temp3"]).to_numpy()


    MinTemp1 = np.min([Temp1,Temp2,Temp3],axis=0)


    SeriesToUse["Ta"] = MinTemp1

    SeriesToUse = SeriesToUse[["WSpeed","RMS20_60","Ta"]]#


    SeriesToUse.columns =  ["MeanCleanWS","RMS_20_60","Ta"]
    NormSeries = (SeriesToUse-minscaler)/(maxscaler-minscaler)
    NormSeries = NormSeries*2 - 1

    DataSeries = NormSeries.to_numpy()

    y_hat = DataSeries[:,0].reshape(-1,1)
    X_hat = DataSeries[:,1:]

    IndNan = ~np.isnan(X_hat)
    IndNan = IndNan.all(axis=1)
    X_hat = X_hat[IndNan,:]
    y_hat=y_hat[IndNan]

    mu_hat, var_hat = m.predict(X_hat)
    
    minscalerwind = minscaler.to_numpy()[0]
    maxscalerwind = maxscaler.to_numpy()[0]
    mu_hatnorm = (mu_hat+1)/2
    mu_hatnorm = (mu_hatnorm*(maxscalerwind-minscalerwind)) + minscalerwind
        
    y_hatnorm = (y_hat+1)/2
    y_hatnorm = (y_hatnorm*(maxscalerwind-minscalerwind)) + minscalerwind

    print('gustiness = '+str(np.std(mu_hatnorm)/np.mean(mu_hatnorm)))

    
    if len(mu_hat) > 500:
    
        timelim = 167


        SecondTable.loc[SecondTable["Filename"].str.contains(FileID),"Gustiness"] = np.std(mu_hatnorm)/np.mean(mu_hatnorm)

        


         
        TimeVec = Soltouse["Time"].to_numpy()[IndNan]
        
        #plt.rcParams['figure.dpi'] = 300
        fig = plt.figure(constrained_layout=True, figsize=(12,9))

        axd = fig.subplot_mosaic(
            """
            A
            B
            C
            """
            )
        axd["A"].plot(TimeVec,y_hatnorm,label='wind speed sensor')
        axd["A"].plot(TimeVec,mu_hatnorm,label='prediction',marker='.',ms=4,color='k',alpha=0.6)
        uplim = mu_hat+1.96*np.sqrt(var_hat)
        lowlim = mu_hat-1.96*np.sqrt(var_hat)
    
        uplim = (uplim+1)/2
        uplim = (uplim*(maxscalerwind-minscalerwind)) + minscalerwind
        

        lowlim = (lowlim+1)/2
        lowlim = (lowlim*(maxscalerwind-minscalerwind)) + minscalerwind
        
        ynan = np.isnan(y_hatnorm)
        mnan = np.isnan(mu_hatnorm)
        Ind = ~ynan & ~mnan
        
            
        ab = SecondTable["Filename"].str.contains(FileID)
        ab = np.where(ab)[0]
        SecondTable.iat[ab[0],SecondTable.columns.get_loc('Mic wind')] = mu_hatnorm
        SecondTable.iat[ab[0],SecondTable.columns.get_loc('Wind vec')] = y_hatnorm
        SecondTable.iat[ab[0],SecondTable.columns.get_loc('Mic wind UB')] = uplim
        SecondTable.iat[ab[0],SecondTable.columns.get_loc('Mic wind LB')] = lowlim


        if SaveFiles == 1:        
            WindPredFiles = {'Wind Speed MEDA':y_hatnorm.reshape(-1,),
                             'RMS_20_60':X_hat[:,0].reshape(-1,),
                             'Ta': MinTemp1.reshape(-1,),
                             'Wind Speed Mic': mu_hatnorm.reshape(-1,),
                             'Wind speed upperbound': uplim.reshape(-1,),
                             'Wind speed lowerbound': lowlim.reshape(-1,)} 

            WindPredFiles_df = pd.DataFrame(WindPredFiles)
            NewFileName = f[:-4]+'_pred.csv'
            WindPredFiles_df.to_csv(NewFileName, index=False)
       
        
       
        axd["A"].fill_between(TimeVec,lowlim.reshape(-1,),uplim.reshape(-1,),alpha=0.3,label='confidence interval')
        axd["A"].legend(loc = 'upper right', fontsize = 18,ncol=3)
        axd["A"].set_ylabel("Wind Speed (m/s)", size=20)
        axd["A"].set_xlabel("Time s", size=20)
        axd["A"].set_xlim((1,timelim))
        axd["A"].set_ylim((0,16))
        axd["A"].tick_params(axis='both', labelsize=18)
        axd["A"].grid()
        axd["A"].set_title('Sol '+str(CompTableVal["Sol"].to_numpy()[0])+', LTST '+str(CompTableVal["LTST"].to_numpy()[0])+', Gustiness = '+str(np.std(mu_hatnorm)/np.mean(mu_hatnorm))[0:4] ,size=20)

        
        axd["B"].plot(TimeVec,X_hat[:,0],marker='.',color='k',ms=4)
        axd["B"].set_xlim((1,timelim))
        axd["B"].set_ylim((-1.1,1.1))
        axd["B"].set_ylabel("Mic RMS (Norm.)", size=20)
        axd["B"].set_xlabel("Time s", size=20)
        axd["B"].tick_params(axis='both', labelsize=18)
        axd["B"].grid()
    
        
        axd["C"].plot(TimeVec,MinTemp1,marker='_',color='k',ms=4, label='Ta')

        axd["C"].plot(TimeVec,Soltouse[["Temp1"]].to_numpy()[IndNan] ,label='ATS 1' )
        axd["C"].plot(TimeVec,Soltouse[["Temp2"]].to_numpy()[IndNan] ,label='ATS 2' )
        axd["C"].plot(TimeVec,Soltouse[["Temp3"]].to_numpy()[IndNan] ,label='ATS 3' )
        
        
        axd["C"].set_xlim((1,timelim))
        axd["C"].set_ylabel("Temp (K)", size=20)
        axd["C"].set_xlabel("Time s", size=20)
        axd["C"].legend(loc = 'upper right',fontsize = 18,ncol=4)
        axd["C"].tick_params(axis='both', labelsize=18)
        axd["C"].grid()
        
        