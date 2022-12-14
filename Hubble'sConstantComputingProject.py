# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:12:10 2019

@author: Alexander Norrbrink, an159, Group L2, Imperial College London
"""

#%%
#Importing packages
import scipy as sp
import matplotlib.pyplot as plt
import glob as glob
#%%
#Variables
lambdae=656.28 #nm (units cancel out in equation)
c=2.9979*10**8 #m/s

#%%
#Defining Functions
def InstrumentResponse(filename):
    with open(filename, "r") as file: 
        Header=file.readline()        
        Headersplit=Header.split(",")
        header=Headersplit[3].split(":")
        Response=header[1].strip()
    if Response=="Good":
        response=True
    else:
        response=False
    return response

def ObservationNumber(filename):
    with open(filename, "r") as file: 
        Header=file.readline()
        Headersplit=Header.split(",")
        header=Headersplit[2].split(":")
        Number=header[1].strip()
    return Number

def RedshiftVelocity(lambda0):
    v0=c*((((lambda0/lambdae)**2)-1)/(((lambda0/lambdae)**2)+1))
    return v0

def fitfunction(Wavelength, amplitude, mean, stddev, m, b):
    gaussian=(amplitude * sp.exp(-((Wavelength - mean) / 4 / stddev)**2))
    line=m*Wavelength+b
    return gaussian+line

def gradient(y2,y1,x2,x1):
    m=(y2-y1)/(x2-x1)
    return m
#%%
#attaining data with good response    
AllData=glob.glob("Halpha_Spectral_data/*.csv")
GoodFiles=[]

for i in range(0,30):
    if InstrumentResponse(AllData[i])==True:
        GoodFiles.append(AllData[i])
#%%
#Attaining Distances Data
Observation, Distance=sp.loadtxt("Data/Distance_Mpc.csv", skiprows=1, delimiter=",", unpack=True)
ObservationNumb=[]
for i in range(0,30):
    ObservationNumb.append(int(Observation[i]))
#%%    
#Reading Files and recording data from them     
RedshiftVelocities=[]
Distances=[]
errorRedshift=[] #percentage error is so small compared to the line fit so it is not nessecary to propagate errors
for i in range(0, 24):
    Wavelength, Intensity=sp.loadtxt(GoodFiles[i], skiprows=2, delimiter=',', unpack=True)
    GradientGuess, YinterceptGuess=sp.polyfit(Wavelength, Intensity, 1) #line of best fit for function to be used in function
    LineGuess=GradientGuess*Wavelength+YinterceptGuess 
    GaussianGuessArray=Intensity-(GradientGuess*Wavelength+YinterceptGuess) #subtracting intensity values by line of best fit to get Gaussian Equation
    maxGau=max(GaussianGuessArray) #finding peak of gaussian(amplitude)
    MaxInti=sp.argmax(GaussianGuessArray) #index of peak to find guess of wavelength
    initial_guess=[maxGau, Wavelength[MaxInti], sp.std(GaussianGuessArray, ddof=1)/2, GradientGuess, YinterceptGuess]
    Fit, fit_cov=sp.optimize.curve_fit(fitfunction, Wavelength, Intensity, initial_guess)
    data_fit=fitfunction(Wavelength, Fit[0], Fit[1], Fit[2], Fit[3], Fit[4]) 
    RedshiftVelocities.append(RedshiftVelocity(Fit[1])*10**-3) #calculating RedshiftVelocity (in km/s) from peak wavelength
    Obsnumb=ObservationNumber(GoodFiles[i])
    for j in range(0, 30):
        if int(Obsnumb)==ObservationNumb[j]:
            Distances.append(Distance[j])
    errorRedshift.append(2**3/2*sp.sqrt(fit_cov[1,1]))         
#%%
#Plotting graph of Redshift Velocity as a function of distnace 
plt.plot(Distances, RedshiftVelocities, ".", color="black")
plt.title("Redshift of Hydrogen Emmision spectrum over Distance", fontname="Times New Roman") 
plt.ylabel("Redshift (km/s)", fontname="Times New Roman") 
plt.xlabel("Distance (Mpc)", fontname="Times New Roman")
plt.errorbar(Distances, RedshiftVelocities, yerr=errorRedshift, fmt=".", mew=3, ms=4,)
plt.grid() 
#plt.xticks(sp.arange(0, , ))
params = {
            'axes.labelsize': 18,
            'font.size': 18,
            'legend.fontsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.figsize': [8.8, 8.8/1.618]
            } 
plt.rcParams.update(params)
#%%    
#Linear fit of Hubble's constant
x=sp.arange(0, 600, 0.5)
RedshiftvsDistance, covMatrixHubble=sp.polyfit(Distances, RedshiftVelocities, 1, cov=True)
Line=sp.poly1d(RedshiftvsDistance)
HubblesConstant, shift=sp.polyfit(Distances, RedshiftVelocities, 1)
plt.plot(Distances, RedshiftVelocities, ".", mew=3, ms=4, color="red")
plt.plot(x, x*HubblesConstant+shift, label="Hubble's Constant= 71 \u00B1 5 km/s", color="black")
plt.title("Redshift of Hydrogen Emmision spectrum over Distance", fontname="Times New Roman") 
plt.ylabel("Redshift (km/s)", fontname="Times New Roman") 
plt.xlabel("Distance (Mpc)", fontname="Times New Roman")
plt.grid() 
plt.xlim(0, 600)
plt.ylim(0, 5*10**4)
plt.legend(loc="upper left")
plt.show()
print("Hubble's Constant is", HubblesConstant, u"\u00B1", sp.sqrt(covMatrixHubble[0,0]) , "km/s or 71", u"\u00B1", "5 km/s")   #only from covariance of this matrix as the percentage error of redshift was miniscule 