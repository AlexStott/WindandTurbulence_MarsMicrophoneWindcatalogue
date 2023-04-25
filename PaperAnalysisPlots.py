#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Make analysis plots for "Wind and turbulence observations with the Mars microphone on Perseverence" Stott et al. (2023) JGR: Planets
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

#%% Read in data table
DataSet = pd.read_csv("AllRecordingDataTable.csv")
DDcat = pd.read_csv("DDcat.csv")


#%% RMS v wind speed fig 


Mic_RMS2060 = (DataSet["RMS_20_60"]).to_numpy()
Mic_RMS1001000 = (DataSet["RMS_100_1000"]).to_numpy()



WindSpeed = (DataSet["MeanCleanWS"]).to_numpy()


DummyWind = np.array([0.01,0.1,1,10,30])


LsVect = DataSet["Ls_deg_"].to_numpy()*10

sortedyInd = np.argsort(LsVect,axis=0)

Ls_min = np.min(LsVect)
Ls_max = np.max(LsVect)

Col = np.arange(np.round(Ls_max-Ls_min)+1)

x = np.round(LsVect)-np.round(Ls_min)
x = x.astype(int)


LTSTVect = (DataSet["LTST"]).to_numpy()

LTSTIntvect = np.zeros(LTSTVect.shape[0])

iter = 0
for time in LTSTVect:
    hh, mm, ss = time.split(':')
    LTSTIntvect[iter] = (int(hh) + int(mm)/60)
    iter += 1

LTSTIntvect = np.round(LTSTIntvect*100)
LTSTIntvect = LTSTIntvect.astype(int)

Colltst = np.arange(2400)


pref = 20e-6
SPL_2060 = 20*np.log10(Mic_RMS2060/(pref))

SPL_1001000 = 20*np.log10(Mic_RMS1001000/(pref))

plt.rcParams['figure.dpi'] = 300

fig = plt.figure(constrained_layout=True, figsize=(8,4))
axd = fig.subplot_mosaic(
   """
   AB
   """
   )

axd["A"].scatter(WindSpeed,SPL_2060,s=25,c=Colltst[LTSTIntvect], marker = 'o', cmap = cm.twilight_shifted,alpha=0.8 )

axd["A"].plot(DummyWind,20*np.log10((0.000005*(DummyWind)**4)/pref),c='green',alpha=0.4,linewidth=5,label='x^4')


axd["A"].set_xscale('log')
axd["A"].set_yscale('linear')
axd["A"].set_ylabel('SPL (dB)',size=16)
axd["A"].set_title('20-60 Hz',size=16)

axd["A"].set_xlabel('Wind speed (m/s)',size=16)
axd["A"].set_xlim((0.1,30))
axd["A"].set_ylim((10,80))
axd["A"].legend(fontsize=14)
axd["A"].grid()
axd["A"].tick_params(axis='both', labelsize=14)


axd["B"].scatter(WindSpeed,SPL_1001000,s=25,c=Colltst[LTSTIntvect], marker = 'o', cmap = cm.twilight_shifted,alpha=0.8 )


axd["B"].set_xscale('log')
axd["B"].set_yscale('linear')
axd["B"].set_ylabel('SPL (dB)',size=16)
axd["B"].set_xlabel('Wind speed (m/s)',size=16)

axd["B"].set_xlim((0.1,30))
axd["B"].set_ylim((10,80))
axd["B"].grid()
axd["B"].tick_params(axis='both', labelsize=14)
axd["B"].set_title('100-1000 Hz',size=16)


#%% Wind v std*mean and direction


Mic_RMS2060 = (DataSet["RMS_20_60"]).to_numpy()
Mic_RMS1001000 = (DataSet["RMS_100_1000"]).to_numpy()

WindSpeed = (DataSet["MeanCleanWS"]).to_numpy()
WindStd = (DataSet["SD_CleanWS"]).to_numpy()
WindDir = (DataSet["Incident_WD"]).to_numpy()


LsVect = DataSet["Ls_deg_"].to_numpy()*10

sortedyInd = np.argsort(LsVect,axis=0)

Ls_min = np.min(LsVect)
Ls_max = np.max(LsVect)

Col = np.arange(np.round(Ls_max-Ls_min)+1)

x = np.round(LsVect)-np.round(Ls_min)
x = x.astype(int)



LTSTVect = (DataSet["LTST"]).to_numpy()
LTSTIntvect = np.zeros(LTSTVect.shape[0])

iter = 0
for time in LTSTVect:
    hh, mm, ss = time.split(':')
    LTSTIntvect[iter] = (int(hh) + int(mm)/60)
    iter += 1

LTSTIntvect = np.round(LTSTIntvect*100)
LTSTIntvect = LTSTIntvect.astype(int)

Colltst = np.arange(2400)



pref = 20e-6
SPL_2060 = 20*np.log10(Mic_RMS2060/(pref))



Mic_RMS2060_norm = Mic_RMS2060/(0.000005*(WindSpeed)**4)


EnoughSpeed = np.where(WindSpeed>=3)



fig = plt.figure(constrained_layout=True, figsize=(8,4))
axd = fig.subplot_mosaic(
   """
   BA
   """
   )


axd["A"].plot(WindDir[EnoughSpeed],Mic_RMS2060_norm[EnoughSpeed],'.',ms=12,markeredgewidth=0,alpha=0.5,c='black')


axd["A"].set_ylabel('Mic normalised (Ratio)',size=16)
axd["A"].set_xlabel('Wind Dir (deg.)',size=16)
axd["A"].set_xlim((-180,180))
axd["A"].grid()
axd["A"].tick_params(axis='both', labelsize=14)


axd["B"].scatter((WindSpeed*WindStd)**0.5,SPL_2060,s=25,c=Colltst[LTSTIntvect], marker = 'o', cmap = cm.twilight_shifted,alpha=0.8 )

axd["B"].plot(DummyWind,20*np.log10((0.0001*(DummyWind)**4)/pref),c='green',alpha=0.4,linewidth=5,label='x^4')


axd["B"].set_xscale('log')
axd["B"].set_yscale('linear')
axd["B"].set_ylabel('SPL (dB)',size=16)
axd["B"].set_xlabel('Wind (speed*std)^0.5 (m/s)',size=16)
axd["B"].set_xlim((0.1,20))
#axd["A"].set_ylim((0.00012,0.3))
axd["B"].set_ylim((10,80))
axd["B"].legend(fontsize=14)
axd["B"].grid()
axd["B"].tick_params(axis='both', labelsize=14)

#%% temp and pressure


SubSet = DataSet[DataSet["Comment"] == "Passive Atmospheric Recording"]


SubSet = SubSet[SubSet["MeanPressure"] > 1]


Mic_RMS2060 = (SubSet["RMS_20_60"]).to_numpy()


pref = 20e-6
SPL_2060 = 20*np.log10(Mic_RMS2060/(pref))


Temp1 = (SubSet["MeanTemp1"]).to_numpy()
Temp2 = (SubSet["MeanTemp2"]).to_numpy()
Temp3 = (SubSet["MeanTemp3"]).to_numpy()

PressSTD = np.log10((SubSet["SD_Pressure"]).to_numpy())



MinTemp = np.min([Temp1,Temp2,Temp3],axis=0)

Temp_Ground = (SubSet["GroundTemperature"]).to_numpy()

ToPlot = {'SPL (dB)':SPL_2060,
          'Ta (K)':MinTemp,
          'Tg (K)':Temp_Ground,
          'Tg-Ta (K)':Temp_Ground-MinTemp,
          'Press. STD log(Pa)':PressSTD}

PlotDF = pd.DataFrame(ToPlot)
plt.rcParams['figure.dpi'] = 300


fig = plt.figure(constrained_layout=True, figsize=(12,4))
axd = fig.subplot_mosaic(
   """
   CAB
   """
   )

lm=sns.kdeplot(ax=axd["A"], data=PlotDF, x="Ta (K)", y="SPL (dB)", alpha=0.7, fill=True,cmap="Greys",levels=8)#, kde=True)#,rug=True)


axd["A"].set_xlim((180,265))
axd["A"].set_ylim((10,80))
axd["A"].grid()
axd["A"].tick_params(axis='both', labelsize=14)
plt.rc('axes', labelsize=16)  

lm=sns.kdeplot(ax=axd["B"], data=PlotDF, x="Tg (K)", y="SPL (dB)", alpha=0.7, fill=True,cmap="Greys",levels=8)#, kde=True)#,rug=True)


axd["B"].set_xlim((180,300))
axd["B"].set_ylim((10,80))
axd["B"].grid()
axd["B"].tick_params(axis='both', labelsize=14)
plt.rc('axes', labelsize=16)  



lm=sns.kdeplot(ax=axd["C"], data=PlotDF, x="Tg-Ta (K)", y="SPL (dB)", alpha=0.7, fill=True,cmap="Greys",levels=8)#, kde=True)#,rug=True)


axd["C"].set_xlim((-20,50))
axd["C"].set_ylim((10,80))
axd["C"].grid()
axd["C"].tick_params(axis='both', labelsize=14)
plt.rc('axes', labelsize=16)  

fig = plt.figure(constrained_layout=True, figsize=(4,4))
axd = fig.subplot_mosaic(
   """
   D
   """
   )

lm=sns.kdeplot(ax=axd["D"], data=PlotDF, x="Press. STD log(Pa)", y="SPL (dB)", alpha=0.7,  fill=True,cmap="Greys",levels=10)#, kde=True)#,rug=True)



axd["D"].set_xlim((-1.8,-0.5))
axd["D"].set_ylim((10,80))
axd["D"].grid()
axd["D"].tick_params(axis='both', labelsize=14)
plt.rc('axes', labelsize=16)  

#%% gustiness correlation plots


SubSet = DataSet[DataSet["Comment"] == "Passive Atmospheric Recording"]

SubSet = SubSet[SubSet["MeanPressure"] > 1]


Mic_RMS2060 = (SubSet["RMS_20_60"]).to_numpy()

Gust = SubSet["Gustiness"].to_numpy()


WindSpeed = (SubSet["MeanCleanWS"]).to_numpy()


Temp1 = (SubSet["MeanTemp1"]).to_numpy()
Temp2 = (SubSet["MeanTemp2"]).to_numpy()
Temp3 = (SubSet["MeanTemp3"]).to_numpy()

LTSTVect = (SubSet["LTST"]).to_numpy()

LTSTIntvect = np.zeros(LTSTVect.shape[0])

iter = 0
for time in LTSTVect:
    hh, mm, ss = time.split(':')
    LTSTIntvect[iter] = (int(hh) + int(mm)/60)
    iter += 1

LTSTIntvect = np.round(LTSTIntvect*100)
LTSTIntvect = LTSTIntvect.astype(int)

Colltst = np.arange(2400)



MinTemp = np.min([Temp1,Temp2,Temp3],axis=0)

Temp_Ground = (SubSet["GroundTemperature"]).to_numpy()


TurbHeatflux = (SubSet["TurbulentHeatFlux"]).to_numpy()


Downwellingflux = (SubSet["Downwelling Flux"]).to_numpy()

LsVect = SubSet["Ls_deg_"].to_numpy()*10

sortedyInd = np.argsort(LsVect,axis=0)


Ls_min = np.min(LsVect)
Ls_max = np.max(LsVect)

Col = np.arange(np.round(Ls_max-Ls_min)+1)

x = np.round(LsVect)-np.round(Ls_min)
x = x.astype(int)



SolVect = SubSet["Sol"].to_numpy()




plt.rcParams['figure.dpi'] = 300

fig = plt.figure(constrained_layout=True, figsize=(12,4))
axd = fig.subplot_mosaic(
   """
   BAC
   """
   )


axd["A"].scatter(MinTemp,Gust,s=40,c=Colltst[LTSTIntvect], marker = 'o', cmap = cm.twilight_shifted,alpha=0.8 )



axd["A"].set_yscale('linear')
axd["A"].set_ylabel('Gustiness',size=16)
axd["A"].set_xlabel('Ta (K)',size=16)
axd["A"].set_ylim((-0.02,0.4))

axd["A"].set_xlim((180,265))

axd["A"].legend(fontsize=14)

axd["A"].grid()
axd["A"].tick_params(axis='both', labelsize=14)


axd["B"].scatter(Temp_Ground - MinTemp,Gust,s=40,c=Colltst[LTSTIntvect], marker = 'o', cmap = cm.twilight_shifted,alpha=0.8 )



axd["B"].set_yscale('linear')
axd["B"].set_ylabel('Gustiness',size=16)
axd["B"].set_xlabel('Tg - Ta (K)',size=16)
axd["B"].set_ylim((-0.02,0.4))
axd["B"].set_xlim((-20,50))
axd["B"].legend(fontsize=14)

axd["B"].grid()
axd["B"].tick_params(axis='both', labelsize=14)


axd["C"].scatter(Temp_Ground,Gust,s=40,c=Colltst[LTSTIntvect], marker = 'o', cmap = cm.twilight_shifted,alpha=0.8 )


axd["C"].set_ylabel('Gustiness',size=16)
axd["C"].set_xlabel('Tg (K)',size=16)

axd["C"].legend(fontsize=14)
axd["C"].set_ylim((-0.02,0.4))
axd["C"].set_xlim((180,300))



axd["C"].grid()
axd["C"].tick_params(axis='both', labelsize=14)






fig = plt.figure(constrained_layout=True, figsize=(12,4))
axd = fig.subplot_mosaic(
   """
   BCA
   """
   )

axd["A"].scatter(WindSpeed,Gust,s=40,c=Colltst[LTSTIntvect], marker = 'o', cmap = cm.twilight_shifted,alpha=0.8 )


axd["A"].set_yscale('linear')
axd["A"].set_ylabel('Gustiness',size=16)
axd["A"].set_xlabel('Wind speed (m/s)',size=16)
axd["A"].set_ylim((-0.02,0.4))
#axd["A"].set_ylim((0.00012,0.3))
axd["A"].legend(fontsize=14)

axd["A"].grid()
axd["A"].tick_params(axis='both', labelsize=14)

axd["B"].scatter(TurbHeatflux,Gust,s=45,c=Colltst[LTSTIntvect], marker = 'o', cmap = cm.twilight_shifted,alpha=0.9 )


axd["B"].set_ylabel('Gustiness',size=16)
axd["B"].set_xlabel('Turbulent Heat Flux (W/m^2)',size=16)
#axd["D"].set_xlim((-20,10))
#axd["D"].set_ylim((0.00012,0.3))
axd["B"].legend(fontsize=14)
axd["B"].set_ylim((-0.02,0.4))

axd["B"].grid()
axd["B"].tick_params(axis='both', labelsize=14)#%%

axd["C"].scatter(Downwellingflux,Gust,s=45,c=Colltst[LTSTIntvect], marker = 'o', cmap = cm.twilight_shifted,alpha=0.9 )


axd["C"].set_ylabel('Gustiness',size=16)
axd["C"].set_xlabel('Downwelling IR Flux (W/m^2)',size=16)
#axd["D"].set_xlim((-20,10))
#axd["D"].set_ylim((0.00012,0.3))
axd["C"].legend(fontsize=14)
axd["C"].set_ylim((-0.02,0.4))

axd["C"].grid()
axd["C"].tick_params(axis='both', labelsize=14)#%%


#%% Dust devil and gustiness plots

    

timeofdevil = (DDcat["LT(hr)"]).to_numpy()/4
sizeofdrop = (DDcat["Amp(Pa)"]).to_numpy()



fonts = 16

DDcat["LTST_round"] = np.floor(timeofdevil)

DDcat["lnPa"] =  np.log(sizeofdrop)


DDcatUse =  DDcat#
DDcat_daytime =  DDcatUse[DDcatUse["LTST_round"].isin([0,1,2,3,4,5,6,7,8,9,10,11])]

fig = plt.figure(constrained_layout=True, figsize=(9,4))
axd = fig.subplot_mosaic(
    """
    A
    """
    )
axd["A"] = sns.violinplot(x="LTST_round",y="lnPa",data=DDcat_daytime,scale="width", inner="stick",bw=0.5)


axd["A"].set_xlabel('LTST hour',size=fonts)
axd["A"].set_ylabel('ln(Pa.)',size=fonts)
axd["A"].set_xticklabels(['0-4','4-8','8-12','12-16','16-20','20-24'])

axd["A"].tick_params(axis='both', labelsize=fonts)


fonts = 20



SubSet = DataSet[DataSet["Comment"] == "Passive Atmospheric Recording"]

SubSet = SubSet[SubSet["MeanPressure"] > 1]
SubSet = SubSet[SubSet["Gustiness"] > 0]


LsVect = SubSet["Ls_deg_"].to_numpy()*10


LTSTVect = (SubSet["LTST"]).to_numpy()
LTSTIntvect = np.zeros(LTSTVect.shape[0])

iter = 0
for time in LTSTVect:
    hh, mm, ss = time.split(':')
    LTSTIntvect[iter] = (int(hh) + int(mm)/60)
    iter += 1

LTSTIntvect = np.floor(LTSTIntvect/4)
LTSTIntvect = LTSTIntvect.astype(int)


LTSTVect_round = np.floor(LTSTIntvect)

SubSet["LTST_round"] = LTSTVect_round

fig = plt.figure(constrained_layout=True, figsize=(9,4))
axd = fig.subplot_mosaic(
    """
    A
    """
    )
axd["A"] = sns.violinplot(x="LTST_round",y="Gustiness",data=SubSet,scale="width", inner="stick", bw=0.5)

axd["A"].set_xlabel('LTST hour',size=fonts)
axd["A"].set_xticklabels(['0-4','4-8','8-12','12-16','16-20','20-24'])
axd["A"].set_ylabel('Gustiness',size=fonts)
axd["A"].tick_params(axis='both', labelsize=fonts)

#%% Pressure drop histrogram and gustiness plot
 
timeofdevil = (DDcat["LT(hr)"]).to_numpy()
sizeofdrop = (DDcat["Amp(Pa)"]).to_numpy()



fonts = 20

DDcat["LTST_round"] = np.floor(timeofdevil)


SubSet = DataSet[DataSet["Comment"] == "Passive Atmospheric Recording"]
SubSet = SubSet[SubSet["MeanPressure"] > 1]
SubSet = SubSet[SubSet["Gustiness"] > 0]



Mic_RMS2060 = (SubSet["RMS_20_60"]).to_numpy()
Gust = SubSet["Gustiness"].to_numpy()

LsVect = SubSet["Ls_deg_"].to_numpy()*10
LTSTVect = (SubSet["LTST"]).to_numpy()
LTSTIntvect = np.zeros(LTSTVect.shape[0])

iter = 0
for time in LTSTVect:
    hh, mm, ss = time.split(':')
    LTSTIntvect[iter] = (int(hh) + int(mm)/60)
    iter += 1

LTSTIntvect_plot = LTSTIntvect
LTSTIntvect = np.round(LTSTIntvect*100)
LTSTIntvect = LTSTIntvect.astype(int)


sortedyInd = np.argsort(LTSTVect,axis=0)


Ls_min = np.min(LsVect)
Ls_max = np.max(LsVect)

Col = np.arange(np.round(Ls_max)-np.round(Ls_min)+1)

x = np.round(LsVect)-np.round(Ls_min)
x = x.astype(int)

Col = Col/10

fig = plt.figure(constrained_layout=True, figsize=(9,4))
axd = fig.subplot_mosaic(
    """
    A
    """
    )
l1 = axd["A"].hist(timeofdevil, bins=np.arange(24))

Gust_ins = (Gust>0)

ax2 = axd["A"].twinx()
l2 = ax2.scatter(LTSTIntvect_plot[Gust_ins],Gust[Gust_ins],s=40,c='k', marker = 'o',alpha=0.6)



axd["A"].set_ylabel('Number of vortices',size=18)
ax2.set_ylabel('Gustiness',size=18)
#ax2.set_ylim((0,0.4))

axd["A"].set_xlabel('LTST',size=18)
axd["A"].set_xlim((0,24))



axd["A"].grid()
axd["A"].tick_params(axis='both', labelsize=18)
ax2.tick_params(axis='both', labelsize=18)



#%% Gustiness over sols plot
SubSet = DataSet[DataSet["Comment"] == "Passive Atmospheric Recording"]

SubSet = SubSet[SubSet["MeanPressure"] > 1]
SubSet = SubSet[SubSet["Gustiness"] > 0]


Mic_RMS2060 = (SubSet["RMS_20_60"]).to_numpy()
Gust = SubSet["Gustiness"].to_numpy()


LTSTVect = (SubSet["LTST"]).to_numpy()
LTSTIntvect = np.zeros(LTSTVect.shape[0])

iter = 0
for time in LTSTVect:
    hh, mm, ss = time.split(':')
    LTSTIntvect[iter] = (int(hh) + int(mm)/60)
    iter += 1

LTSTIntvect = np.round(LTSTIntvect*100)
LTSTIntvect = LTSTIntvect.astype(int)

Colltst = np.arange(2400)




SolVect = SubSet["Sol"].to_numpy()

LTSTVect = pd.to_datetime(SubSet["LTST"]).to_numpy()#*10
LsVect = SubSet["Ls_deg_"].to_numpy()

area = np.log10(Mic_RMS2060)
maxval = np.max(area)
minvam = np.min(area)
area = (area-minvam)/(maxval-minvam)
area = area*200 + 10

maxvalGust = np.max(Gust)
minvamGust = np.min(Gust)
Gust = (Gust-minvamGust)/(maxvalGust-minvamGust)
Gust = Gust*10000 
Ls_min = np.min(Gust)
Ls_max = np.max(Gust)

Col = np.arange(np.round(Ls_max-Ls_min)+1)

x = np.round(Gust)-np.round(Ls_min)
x = x.astype(int)

Col = Col/10000
Col = (Col)*(maxvalGust-minvamGust) + minvamGust


fig = plt.figure(constrained_layout=True, figsize=(15,5))
axd = fig.subplot_mosaic(
   """
   A
   """
   )



im = axd["A"].scatter(LsVect, LTSTVect, s=area, c=Col[x], cmap = cm.winter, alpha=0.6)
hh_mm = mdates.DateFormatter('%H:%M')
axd["A"].yaxis.set_major_formatter(hh_mm)
axd["A"].set_ylabel('LTST',size=22)
axd["A"].set_xlabel('Ls deg',size=22)
axd["A"].tick_params(axis='both', labelsize=22)

axd["A"].grid()

axins = inset_axes(axd["A"],
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=axd["A"].transAxes,
                   borderpad=0,
                   )
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('Gustiness', size = 22)
cbar.ax.tick_params(labelsize=18) 
