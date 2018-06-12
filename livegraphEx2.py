# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:00:31 2017
for sampling rate --> task.timing.cfg_samp_clk_timing(10000, u'',10280,10178,20) 
@author: minsooyeo119112
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

import time
import os
from peakdetect import peakdetect 
import threading
from scipy.signal import butter, lfilter, sosfilt
import pickle
import scipy as sc
import nidaqmx as ni
from biosppy.signals import ecg

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def calAlphaBP(data, Fs):
    return ( np.sum(np.abs(np.fft.fft(butter_bandpass_filter(data,8,13,Fs) )) ) * 2) / len(data)

def calFatigureFactor(data, Fs):
    A = ( np.sum(np.abs(np.fft.fft(butter_bandpass_filter(data,8,13,Fs) )) ) * 2) / len(data)
    B = ( np.sum(np.abs(np.fft.fft(butter_bandpass_filter(data,13,35,Fs) )) ) * 2) / len(data)
    T = ( np.sum(np.abs(np.fft.fft(butter_bandpass_filter(data,4,8,Fs) )) ) * 2) / len(data)
    return (T + A) / (B)

class ProcessingTotal:
    def __init__(self, EEG, ECG, Fs, modelName):
        self.EEG = EEG
        self.ECG = ECG
        
        EEGOb = ProcessingEEG(EEG, Fs, modelName)
        PPGOb = ProcessingPPG(ECG, Fs, modelName)

        with open('./realtileModel/' + modelName, 'rb') as f:    
            self.RFClassifier = pickle.load(f)
            
        self.Feature = np.r_[EEGOb.EEGFeatureEXPre(), PPGOb.TimeDomainFeatures, PPGOb.FreDomainFeatures]

    def Classification(self):
        return self.RFClassifier.predict(self.Feature.reshape([1, len(self.Feature)]))

class ProcessingEEG:
    def __init__(self, data, Fs, modelName):
        self.data = data
        self.Fs = Fs
        self.Feature = self.EEGFeatureEXPre()
        with open('./realtileModel/' + modelName, 'rb') as f:    
            self.RFClassifier = pickle.load(f)
        
    def EEGFeatureEXPre(self):
        AlphaSig = butter_bandpass_filter(self.data,8,13,self.Fs)
        BetaSig = butter_bandpass_filter(self.data,13,20,self.Fs)
        ThetaSig = butter_bandpass_filter(self.data,4,8,self.Fs)
        
        Beta = np.sum(np.abs(   (np.fft.fft(BetaSig) / len(BetaSig))*2  )) 
        Alpha = np.sum(np.abs(   (np.fft.fft(AlphaSig) / len(BetaSig))*2  ))
        Theta = np.sum(np.abs(   (np.fft.fft(ThetaSig) / len(BetaSig))*2  ))
        
        EEGFeature1 = (Theta + Alpha) / (Beta)
        EEGFeature2 = (Alpha) / (Beta)
        EEGFeature3 = (Theta + Alpha) / (Alpha + Beta)
        EEGFeature4 = (Theta) / (Beta)     
       
        return np.array([EEGFeature1, EEGFeature2, EEGFeature3, EEGFeature4])   
            
    def Classification(self):
        return self.RFClassifier.predict(self.Feature.reshape([1, len(self.Feature)]))

class ProcessingPPG:
    def __init__(self, data, Fs, modelName):
        self.data = data
        self.currentcwd = os.getcwd()
        self.Fs = Fs
        self.TimeDomainFeaturesName = ['meanNN','SDNN','RMSSD','SDSD','SDANN',
                                       'SDNNi','NN10','NN20','NN30','NN40',
                                       'NN50','meanHR','sdHR',
                                       'pNN10','pNN20','pNN30','pNN40','pNN50']
        
        self.FreDomainFeaturesName = ['TF','VLF','LF','HF','LFn',
                                       'HFn','LFHF']
        
        with open('./realtileModel/' + modelName, 'rb') as f:    
            self.RFClassifier = pickle.load(f)
                                       

        self.peakMax, self.filteredData = self.RRIntervalDetection()
        self.RRInterval = np.diff(self.peakMax[:,0] ) / self.Fs

        self.TimeDomainFeatures = self.CalTimedomainFeatures()
        self.FreDomainFeatures = self.CalFredomainFeatures()
         
    def RpeakArrange(self, Rpeaks):
        RpeakInfo = []
        DataPoint = len(Rpeaks)
        
        for i in range(DataPoint):
            RpeakInfo.append([Rpeaks[i], self.data[Rpeaks[i]]])
        
        return RpeakInfo
    
    def RRIntervalDetection(self):
        filteredData = butter_bandpass_filter(self.data,0.14,0.4,self.Fs)
#        out = ecg.ecg(signal=self.data, sampling_rate = self.Fs, show=False)
#        peaksMax = self.RpeakArrange(out['rpeaks'])
#        peaksMax = np.array(peaksMax)
        peaksMax = peakdetect(self.data, lookahead=300)[0]
        peaksMax = np.stack(peaksMax)
        np.save(self.currentcwd + '/RealTimePPGDumpfile/PeakInfo', peaksMax)
        return peaksMax, filteredData
        
    def CalTimedomainFeatures(self):
        TimeDomainFeaturesNumber = len(self.TimeDomainFeaturesName)
        PPGTimeDomainFeatures = np.zeros([TimeDomainFeaturesNumber])
        DiffRRInterval = np.diff(self.RRInterval)
        HR = 60 / (self.RRInterval)
        
        PPGTimeDomainFeatures[0] = np.mean(self.RRInterval)
        PPGTimeDomainFeatures[1] = np.std(self.RRInterval)
        PPGTimeDomainFeatures[2] = np.linalg.norm(DiffRRInterval) / np.sqrt(len(DiffRRInterval))
        PPGTimeDomainFeatures[3] = np.std(DiffRRInterval)
        PPGTimeDomainFeatures[4] = 1 # SDANN
        PPGTimeDomainFeatures[5] = 1 # SDNNi
        
        # NN(x)_count --> The number of absolute differences in successive NN values > x (ms)
        if( (np.mean(self.RRInterval) / self.Fs) > 0.1 ):
            PPGTimeDomainFeatures[6] = len(np.where(np.abs(DiffRRInterval) > 10)[0] )
            PPGTimeDomainFeatures[7] = len(np.where(np.abs(DiffRRInterval) > 20)[0] )
            PPGTimeDomainFeatures[8] = len(np.where(np.abs(DiffRRInterval) > 30)[0] )
            PPGTimeDomainFeatures[9] = len(np.where(np.abs(DiffRRInterval) > 40)[0] )
            PPGTimeDomainFeatures[10] = len(np.where(np.abs(DiffRRInterval) > 50)[0] )            
        else:
            PPGTimeDomainFeatures[6] = len(np.where(np.abs(DiffRRInterval) > 0.01)[0] )
            PPGTimeDomainFeatures[7] = len(np.where(np.abs(DiffRRInterval) > 0.02)[0] )
            PPGTimeDomainFeatures[8] = len(np.where(np.abs(DiffRRInterval) > 0.03)[0] )
            PPGTimeDomainFeatures[9] = len(np.where(np.abs(DiffRRInterval) > 0.04)[0] )
            PPGTimeDomainFeatures[10] = len(np.where(np.abs(DiffRRInterval) > 0.05)[0] )            

        PPGTimeDomainFeatures[11] = np.mean(HR)
        PPGTimeDomainFeatures[12] = np.std(HR)
        
        PPGTimeDomainFeatures[13] =( PPGTimeDomainFeatures[6] / len(DiffRRInterval) ) * 100
        PPGTimeDomainFeatures[14] =( PPGTimeDomainFeatures[7] / len(DiffRRInterval) ) * 100
        PPGTimeDomainFeatures[15] =( PPGTimeDomainFeatures[8] / len(DiffRRInterval) ) * 100
        PPGTimeDomainFeatures[16] =( PPGTimeDomainFeatures[9] / len(DiffRRInterval) ) * 100
        PPGTimeDomainFeatures[17] =( PPGTimeDomainFeatures[10] / len(DiffRRInterval) ) * 100
        return PPGTimeDomainFeatures
        
    def CalFredomainFeatures(self):
        freDomainFeaturesNumber = len(self.FreDomainFeaturesName)
        PPGFreDomainFeatures = np.zeros([freDomainFeaturesNumber])
        PPGFreDomainFeatures[0] = ( np.sum(np.abs(np.fft.fft(butter_bandpass_filter(self.RRInterval,0.14,0.4,self.Fs) )) ) * 2) / len(self.data)
        PPGFreDomainFeatures[1] = ( np.sum(np.abs(np.fft.fft(butter_bandpass_filter(self.RRInterval,0.001,0.04,self.Fs) )) ) * 2) / len(self.data)
        PPGFreDomainFeatures[2] = ( np.sum(np.abs(np.fft.fft(butter_bandpass_filter(self.RRInterval,0.04,0.15,self.Fs) )) ) * 2) / len(self.data)
        PPGFreDomainFeatures[3] = ( np.sum(np.abs(np.fft.fft(butter_bandpass_filter(self.RRInterval,0.15,0.4,self.Fs) )) ) * 2 ) / len(self.data)
        PPGFreDomainFeatures[4] = ( (PPGFreDomainFeatures[2]) / (PPGFreDomainFeatures[2] + PPGFreDomainFeatures[3]) ) * 100
        PPGFreDomainFeatures[5] = ( (PPGFreDomainFeatures[3]) / (PPGFreDomainFeatures[2] + PPGFreDomainFeatures[3]) ) * 100
        PPGFreDomainFeatures[6] = ( (PPGFreDomainFeatures[2]) / (PPGFreDomainFeatures[3]) ) * 100

        return PPGFreDomainFeatures
        
    def Classification(self):
        Feature = np.r_[self.TimeDomainFeatures, self.FreDomainFeatures]
        #Feature = np.delete(Feature,[4,5],0)
        return self.RFClassifier.predict(Feature.reshape([1, len(Feature)]))
    
    def VisuallizationPeakInfo(self):
        plt.figure()
        plt.plot(self.data,c = 'b')
        plt.scatter(self.peakMax[:,0], self.data[self.peakMax[:,0].astype('int')], c = 'r')


  
if __name__ == '__main__':
    print('This code is used for utills')


