# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:33:18 2018

@author: minsooyeo119112
"""

import nidaqmx as ni
from matplotlib import pylab as plt
from livegraphEx2 import ProcessingPPG, ProcessingEEG, calAlphaBP, butter_bandpass, ProcessingTotal
import numpy as np
import time
import matplotlib.animation as animation
import pickle
import threading
from collections import deque
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from player import QApplication, VideoWindow
import sys

class PlotAnal():
    def __init__(self, Fs, EstiInt, app, player):
        self.player = player
        self.app = app
        self.player.resize(640, 480)
        self.player.show()
            
        self.Fs = Fs
        self.EstiInt = EstiInt
        
        self.currentTime = 0
        self.taskai0 = ni.Task()
        self.taskai0.ai_channels.add_ai_voltage_chan('Dev2/ai0:3',terminal_config=ni.constants.TerminalConfiguration.RSE, min_val=-10, max_val=10)           
        self.taskai0.timing.cfg_samp_clk_timing(rate=Fs)
        
        self.TotalData = []
        self.TotalDataNumpyArray = np.zeros([1000,4])
        self.TotalData2 = np.random.rand(4,100)
        
        self.RecordTime = []
        self.AlphaBandPower = []
        self.Estimations = []
        self.TotalRecordTime = 0
        self.tmpData = 0
        
        self.l1, self.l2, self.l3, self.l4, self.fig, self.TimeText = self.MakeFigure()
        self.count = 0

        self.axStart = plt.axes([0.7, 0, 0.1, 0.065])
        self.axStop = plt.axes([0.81, 0, 0.1, 0.065])
        self.axTextBox = plt.axes([0.1, 0, 0.2, 0.065])
        self.axTime = [0.35,0.025]
        self.CurrentRecordTime = 0
        self.bpStart, self.bpStop, self.TextBox = self.MakeButton()
        self.line_ani = 0
        
        self.RecordData = 0
        self.ProcessingTime = EstiInt
              
    def MakeFigure(self):
        fig1 = plt.figure()
        sub1 = fig1.add_subplot(4,1,1)
        sub1.grid()
        l, = plt.plot([],[], 'r-')
        sub1.add_line(l)
        sub1.set_xlim(0, int(self.Fs * 10))
        sub1.set_ylim(-10, 10)
        sub1.set_xlabel('x')
        sub1.set_title('EEG')
        
        sub2 = fig1.add_subplot(4,1,2)
        sub2.grid()
        l2, = plt.plot([],[], 'r-') 
        sub2.add_line(l2)
        sub2.set_xlim(0, int(self.Fs * 10))
        sub2.set_ylim(-10, 10)
        sub2.set_xlabel('x')
        sub2.set_title('PPG')
    
        sub3 = fig1.add_subplot(4,1,3)
        sub3.grid()
        l3, = plt.plot([],[], 'r-') 
        sub3.add_line(l3)
        sub3.set_xlim(0, int(self.Fs * 10))
        sub3.set_ylim(-10, 10)
        sub3.set_xlabel('x')
        sub3.set_title('PD Sensor')
    
        sub4 = fig1.add_subplot(4,1,4)
        sub4.grid()
        l4, = plt.plot([],[], 'r-')   
        sub4.add_line(l4)
        sub4.set_xlim(0, int(self.Fs * 10))
        sub4.set_ylim(-10, 10)
        sub4.set_title('Switch')
        
        axtext2 = plt.axes([0.25, 0, 0.15, 0.095])            
        axtext2.axis("off")          
        time1 = axtext2.text(0.5,0.5, 'Record Time --> ', ha="left", va="top")     

        axtext = fig1.add_axes([0.25, 0, 0.50, 0.095])            
        axtext.axis("off")          
        time = axtext.text(0.5,0.5, str(0), ha="left", va="top")  
        
        fig1.show()            
        return l, l2, l3, l4, fig1, time
    
    def SetTimeText(self, Time):
        self.fig.text(self.axTime[0],self.axTime[1],'Record Time --> ' + str(Time))            
        return 1
    
    def MakeButton(self):
        bpStart = Button(self.axStart, 'Start')   
        bpStop = Button(self.axStop, 'Exit & Save')           
        textBox = TextBox(self.axTextBox, 'File Name', initial="")

        
        bpStart.on_clicked(self.Start)
        bpStop.on_clicked(self.Stop)

        return bpStart, bpStop, textBox
    
    def Start(self,event):     
        
        self.line_ani = self.RunAnimation(10)
        return 0
    
    def Stop(self,event):
        self.fig = plt.close()
        FileName = self.TextBox.text
        TotalData = np.vstack(self.TotalData)
        SaveData = {'Data' : TotalData, 'Fs' : self.Fs, "TimeInfo": self.RecordTime, "AlphaBand" : self.AlphaBandPower, "Estimation" : self.Estimations,"TotalTime" : self.TotalRecordTime}
        
        with open('./Data/' + FileName + '.pkl', 'wb') as f:
            pickle.dump(SaveData, f)            
        
        with open('./Data/' + FileName + '.pkl', 'rb') as f:
            self.RecordData = pickle.load(f)
            
        sys.exit(self.app.exec_()) 
        return 0
    
    def RunAnimation(self, intervels):
        if(self.player.IsOpen == 0):
            print('Load Video First ! !')
            raise ValueError
        else:
            self.player.mediaPlayer.play()
            self.currentTime = time.time()
            line_ani = animation.FuncAnimation(self.fig, self.update_lineThread,
                                              interval=intervels,blit = True)             
        return line_ani
    
    def Graph(self, data, Time):            
        self.TotalData.append(np.array(data).T)
        Totaldata = np.vstack(self.TotalData)
        length = Totaldata[-(self.Fs * 10 +1):-1,3].shape[0]

        self.l1.set_data(range(length), Totaldata[-(self.Fs * 10 +1):-1,0])
        self.l2.set_data(range(length), Totaldata[-(self.Fs * 10 +1):-1,1])
        self.l3.set_data(range(length), Totaldata[-(self.Fs * 10 +1):-1,2])
        self.l4.set_data(range(length), Totaldata[-(self.Fs * 10 +1):-1,3])
        
        self.TimeText.set_text( str(Time) )
        
        
    def ProcessingParts(self):
        Totaldata = np.vstack(self.TotalData)
        EEGOB = ProcessingEEG(Totaldata[-(self.EstiInt * self.Fs + 1):-1,0],self.Fs, 'EEGModel.pkl')
        PPGOB = ProcessingPPG(Totaldata[-(self.EstiInt * self.Fs + 1):-1,1],self.Fs, 'PPGModel.pkl') 
        TotalOB = ProcessingTotal(Totaldata[-(self.EstiInt * self.Fs + 1):-1,0], Totaldata[-(self.EstiInt * self.Fs + 1):-1,1], self.Fs, 'TotalModel.pkl')
        
#        with open('./TestOb', 'wb') as f:
#            pickle.dump(TotalOB, f)            
#        
        
        Estimation = EEGOB.Classification()
        #Estimation = PPGOB.Classification()
        #Estimation = TotalOB.Classification()
        
        
        
        
        if(Estimation):
            analPlot.player.mediaPlayer.pause()
        else:
            analPlot.player.mediaPlayer.play()
        
        Alpha = calAlphaBP(Totaldata[-(self.EstiInt * self.Fs + 1):-1,1],self.Fs)
        RecordTime = time.time() - self.currentTime
        
        self.RecordTime.append(RecordTime)
        self.AlphaBandPower.append(Alpha)
        self.Estimations.append(Estimation)
        
        print('state --> ' + str(Estimation))            
        print('Alpha Band Power --> ' + str(Alpha))
        print('Record Time --> ' + str(RecordTime))
        
    def ReadData(self): 
        data = self.taskai0.read(number_of_samples_per_channel=int(self.Fs / 10))
        self.TotalData.append(np.array(data).T)
        self.TotalRecordTime = round(time.time() - self.currentTime,3)
        Result = self.DataStack()
        return Result
  
    def GraphThread(self, Time):            
        
        length = self.TotalDataNumpyArray[-(self.Fs * 10 +1):-1,3].shape[0]

        self.l1.set_data(range(length), self.TotalDataNumpyArray[-(self.Fs * 10 +1):-1,0])
        self.l2.set_data(range(length), self.TotalDataNumpyArray[-(self.Fs * 10 +1):-1,1])
        self.l3.set_data(range(length), self.TotalDataNumpyArray[-(self.Fs * 10 +1):-1,2])
        self.l4.set_data(range(length), self.TotalDataNumpyArray[-(self.Fs * 10 +1):-1,3])
        
        self.TimeText.set_text( str(Time) )     
        
    def DataStack(self):
        self.TotalDataNumpyArray = np.vstack(self.TotalData[-101:-1])
        return 1
        
    def update_lineThread(self, num):
        tData = threading.Thread(target=self.ReadData)
        tGraph = threading.Thread(target=self.GraphThread, args = ([self.TotalRecordTime]))
        tDataProcessing = threading.Thread(target=self.ProcessingParts)
        
        tData.start()
        tGraph.start()
        
        if(self.TimeCal2() >= self.ProcessingTime):
            
            tDataProcessing.start()
            self.ProcessingTime += self.EstiInt
            
        return self.l1, self.l2, self.l3, self.l4, self.TimeText         
    
    def update_line5(self, num):

        data = self.taskai0.read(number_of_samples_per_channel=int(self.Fs / 10))
        self.TotalRecordTime = round(time.time() - self.currentTime,3)
        tGraph = threading.Thread(target=self.Graph, args = ([data, self.TotalRecordTime]))
        tPro = threading.Thread(target=self.ProcessingParts)
        
        tGraph.start()
        self.count += 1
        
        if(self.count == int(10 * self.EstiInt)):
            tPro.start()
            self.count = 0 
                   
        return self.l1, self.l2, self.l3, self.l4, self.TimeText,  
      
    def update_line6(self, num):

        Data = self.taskai0.read(number_of_samples_per_channel=int(self.Fs / 10))
        
        tTime = threading.Thread(target=self.TimeCal)  
        tGraph = threading.Thread(target=self.Graph, args = ([Data, self.TotalRecordTime]))
        tPro = threading.Thread(target=self.ProcessingParts)  
        
        tTime.start()
        tGraph.start()
        self.count += 1
        
        if(self.count == int(10 * self.EstiInt)):
            tPro.start()
            self.count = 0 
                   
        return self.l1, self.l2, self.l3, self.l4, self.TimeText,
    
    def TimeCal(self):
        self.TotalRecordTime = round(time.time() - self.currentTime,3)
        return 0

    def TimeCal2(self):
        TotalRecordTime = round(time.time() - self.currentTime,3)
        return TotalRecordTime

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    analPlot = PlotAnal(1000, 10, app, player)
    sys.exit(app.exec_())     
