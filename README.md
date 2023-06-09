# Repository for code and data in "Wind and turbulence observations with the Mars microphone on Perseverance"

This repository contains the code and data for the results in "Wind and turbulence observations with the Mars microphone on Perseverance" Stott et al. (2023), JGR: Planets. The aim of this work is to produce Mars wind speed estimates based on the microphone data recorded by the NASA Perseverance rover. The included data consists of the original microphone and MEDA meteorological data along with the wind speed predictions from the microphone generated by the methods. Code is also included to reproduce the analysis in the paper.

Contents:
  - "GP_DataSet.csv" - The microphone RMS, MEDA Air temperature and wind speed data used to train the Gaussian Process regression model used to predict wind speeds.
  - "AllRecordingDataTable.csv" - A table of data from each microphone recording along with coincident meteorological data and derived gustiness values of each microphone based wind speed.
  - "DDcat.csv" - catalogue of pressure drop size and local time.
  - "CSVdata/" - folder containing pre-processed RMS, air temperature and wind data for each microphone recording used to predict the wind speed
  - "WindSpeedcsv/" - folder containing wind speed estimates based on the microphone and air temperature data produced in this work
  - "GPtrainingandwindspeedprediction.py" - code to train the GP regression model and predict wind speeds for each microphone recording
  - "PaperAnalysisPlots.py" - code to produce all analysis plots generated in this paper
  - "MicTimeSeries.mat" - Matlab data file of a single microphone recording
  - "PlotMicSpectrogram.m" - Matlab code used to plot the microphone recording spectrogram
  - "turbo.mat" - colourmap for Matlab

The pressure drop catalogue is originally available in Hueso et al. (2022) https://doi.org/10.5281/zenodo.7315863

The turbulent heatflux values are originally available in Martinez et al. (2022) https://repository.hou.usra.edu/handle/20.500.11753/1839

The downwelling flux values are originally available in Smith et al. (2022) http://doi.org/10.17632/48phhtkcj8.1

The original data available on the NASA PDS at https://doi.org/10.17189/1522646 for the microphone data and https://doi.org/10.17189/1522849 for the MEDA meteorological data.
