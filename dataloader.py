import numpy as np
from neuralnet import DynamicNNstage1, Dynamicdataset
import torch
import torch.nn as nn
import torch.nn.functional as F

#reads the files

def filereader(plate, type, Angle = None, Frequency = None):
    import pandas as pd
    import os
    read = []

    headerrow = ['Time [s]','Pot [V]','Pot [degree]','AF [V]','AF_f [V]','AR [V]','AR_f [V]','SC [mV]','SF [mV]','SR [mv]','Bending [N-mm]','Servo','Trigger [V]']
    #for static cases
    if(type == 'Static' or type == 'static'): 
        type = 'Static'
        platenameformat = plate.replace(" ","_")
        dirpath = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','DATA',f'Plate {plate}',f'{type}'))
        allfiles = [f for f in os.listdir(dirpath)]
        for f in allfiles:
            tempf = os.path.join(dirpath,f)
            data = pd.read_csv(tempf,encoding = 'cp1252')
            data = data.dropna(axis = 1)
            data = data.set_axis(headerrow,axis = 1,copy=True)
            read.append([data,f])

    #for dynamic cases
    elif(type == 'Dynamic' or type == 'dynamic'): 
        type = 'Dynamic'
        dirpath = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','DATA',f'Plate {plate}',f'{type}',f'A{Angle}',f'F{Frequency}'))
        allfiles = [f for f in os.listdir(dirpath)]
        for f in allfiles:
            tempf = os.path.join(dirpath,f)
            data = pd.read_csv(tempf,encoding = 'cp1252')
            data = data.dropna(axis = 1)
            data =data.set_axis(headerrow,axis = 1,copy=True)
            read.append([data,f])

    
    out = np.array(read, dtype=object)        
    return out

#split the files into each response (only dynamic has the responces)
def Separateruns(data,hz = 0.5):
    """
        Split the files into each response (only dynamic has the responces)
    """
    runs = []
    multirun = data[0]
    Timedata = multirun[['Time [s]','Trigger [V]']]

    #vartemp = np.where(Timedata['Trigger [V]'] > 3)[0]


    triggerpoints = multirun['Time [s]'].iloc[np.where(Timedata['Trigger [V]'] > 3)[0]]
    startpoints = []
    for i in triggerpoints:
        
        smallestingroup = True
        for a in triggerpoints:
            if a<i and i<a+1:
                smallestingroup = False
        
        if smallestingroup:

            startpoints.append(np.where(Timedata['Time [s]'] == i)[0][0])
            

    hzcorrection = 1/hz /2
    for i in startpoints:
        
        run = multirun[:].iloc[np.where((Timedata['Time [s]'] < (4+hzcorrection+Timedata['Time [s]'].iloc[i])) & (Timedata['Time [s]'] >= Timedata['Time [s]'].iloc[i]))[0]]
        run['Time [s]'] = run['Time [s]'] - run['Time [s]'].values[:1]
        runs.append(run) 
        
        #runs = np.append(runs,multirun['Time [s]'].iloc[np.where((Timedata['Time [s]'] < (4+Timedata['Time [s]'].iloc[i])) & (Timedata['Time [s]'] >= Timedata['Time [s]'].iloc[i]))[0]])

    
    #print(vartemp)
    #print(Timedata)
    return runs

def Train_NN(model,data,epoch,lr):

    loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
    print(data[:])


    return model


class data:
    """
        Data class for use when processing, holds methods that automate the processing
    """
    #Initialise plate
    def __init__(self,Plate):
        self.Plate = Plate
        self.static = filereader(Plate,'static')
     
    #returns the array with the all the data from 1 dynamic case and loads it into object 
    def Get_Dynamic(self,angle,frequency):
        """
        Gets all files in a dynamic case, loads them into the objects and returns them
        """
          
        A = str(angle)
        F = str(frequency).replace(".","")

        Dynamiccase = filereader(self.Plate,'Dynamic',A,F)
        self.dynamicloaded = Dynamiccase
        return Dynamiccase

    #gets the indexes of the free and locked cases
    def Split_Static(self):
        """
        Splits the static cases for the plates between locked and free and loads them into the class object

        """
        
        listfreeindex = np.empty((0,1), dtype=int)
        listlockedindex = np.empty((0,1), dtype=int)
        #get the stringnames with either locked or free in it
        for datafile in self.static[:,1]:     
            idfile = np.where(self.static[:,1] == datafile)[0][0]
            name = datafile
            name = name.split('_')
        
            for part in name:
                if(part == "Free"):
                    listfreeindex = np.append(listfreeindex ,idfile)
                if(part == "Locked"):
                    listlockedindex = np.append(listlockedindex,idfile)

        self.static_free_index = listfreeindex
        self.static_locked_index = listlockedindex

    #returns the array with only data from the free hinges
    def Get_Static_Free(self):
        """
        returns the data for the static free case
        """
        self.Split_Static()
        splitdata = self.static[self.static_free_index]
        return splitdata

    #returns the array with only data from the locked hinges
    def Get_Static_Locked(self):
        """
        returns the data for the static locked case
        """
        self.Split_Static()
        splitdata = self.static[self.static_locked_index]
        return splitdata

    #
    def Split_Dynamic_Loaded(self,fileselect,hz):
        """
        returns and loads the data for the dynamic case with a given file ID
        """
        splitdata = Separateruns(self.dynamicloaded[fileselect,:],hz)
        self.dynamicsplit = splitdata
        self.dynamichz = hz
        return splitdata

    def Split_Dynamic_Unloaded(self,fileselect,data,hz):
        """
        returns the data for the dynamic case with a given file ID

        Dynamic_Loaded is prefered if data is proccessed afterwards
        """
        return Separateruns(data[fileselect,:],hz)

    def Train_Dynamic_models_2D_Loaded(self,Xname,Yname,epoch = 500,lr = 0.01):
        """
        returns ai models per splitdata and loads them

        """
        models = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i in self.dynamicsplit:
            
            targets = torch.tensor(i[Yname].values)
            inputdata = torch.tensor(i[Xname].values)
            dataset = torch.data_utils.TensorDataset(inputdata, targets)  #Dynamicdataset(inputdata,targets)
            trained = Train_NN( DynamicNNstage1(len(Xname),len(Yname)), dataset , epoch, lr)
            models.append(trained)

        self.Dynamicmodelstrained = models
        return models

    

    
        

    

      

    


