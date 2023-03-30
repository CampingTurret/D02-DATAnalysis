
import numpy as np
from neuralnet import DynamicNNstage1
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm,trange
from GapDetect import GapDetect
from ExtremeValues import EVDetect
import os
import pandas as pd
#reads the files

def filereader(plate, ftype, Angle = None, Frequency = None):
   
    read = []

    headerrow = ['Time [s]','Pot [V]','Pot [degree]','AF [V]','AF_f [V]','AR [V]','AR_f [V]','SC [mV]','SF [mV]','SR [mv]','Bending [N-mm]','Servo','Trigger [V]']
    #for static cases
    if(ftype == 'Static' or ftype == 'static'): 
        ftype = 'Static'
        platenameformat = plate.replace(" ","_")
        dirpath = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','DATA',f'Plate {plate}',f'{ftype}'))
        allfiles = [f for f in os.listdir(dirpath)]
        for f in allfiles:
            tempf = os.path.join(dirpath,f)
            data = pd.read_csv(tempf,encoding = 'cp1252')
            data = data.dropna(axis = 1)
            data = data.set_axis(headerrow,axis = 1,copy=True)
            read.append([data,f])

    #for dynamic cases
    elif(ftype == 'Dynamic' or ftype == 'dynamic'): 
        ftype = 'Dynamic'
        dirpath = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','DATA',f'Plate {plate}',f'{ftype}',f'A{Angle}',f'F{Frequency}'))
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
        
        run = multirun[:].iloc[np.where((Timedata['Time [s]'] < (3+hzcorrection+Timedata['Time [s]'].iloc[i])) & (Timedata['Time [s]'] >= Timedata['Time [s]'].iloc[i]))[0]]
        run['Time [s]'] = run['Time [s]'] - run['Time [s]'].values[:1]
        runs.append(run) 
        
        #runs = np.append(runs,multirun['Time [s]'].iloc[np.where((Timedata['Time [s]'] < (4+Timedata['Time [s]'].iloc[i])) & (Timedata['Time [s]'] >= Timedata['Time [s]'].iloc[i]))[0]])

    
    #print(vartemp)
    #print(Timedata)
    return runs

def Train_NN(model,data,epoch,lr):
    
    test, val = torch.utils.data.random_split(data, [0.8,0.2],torch.Generator(device='cpu'))
    loader_train = torch.utils.data.DataLoader(test, batch_size=256, shuffle=True)
    loader_val = torch.utils.data.DataLoader(val, batch_size=256, shuffle=True)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    prevlost = 2*10**4
    startpatience = 4
    patience = startpatience

    for i in trange(epoch,desc = "epoch",leave=False):
        model.train(True)
        for data in iter(loader_train):
            inputs, wanted = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, wanted)
            loss.backward()
            optimizer.step()


        
        if i > epoch/2:
            model.eval()
            with torch.no_grad():
                for data in iter(loader_val):
                    inputs, wanted = data
                    outputs = model(inputs)
                    lossval = torch.sum(loss_fn(outputs, wanted))
                lossvalnew = lossval.item()
            if abs(lossvalnew < 20):
                if abs(lossval.item()) >= abs(prevlost + 0.001):
                    if patience < 0:
                        return model
                    else: 
                        patience = patience - 1
                else:
                    patience = startpatience
                    prevlost = min(lossval.item(),prevlost)
    return model


class data:
    """
        Data class for use when processing, holds methods that automate the processing
    """
    #Initialise plate
    def __init__(self,Plate,AOA,hz,device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.Plate = Plate
        self.dynamichz  = hz
        if self.dynamichz == 'Flap': self.dynamichz =1.5
        if self.dynamichz == 'Bend': self.dynamichz =3
        self.dynamichz = float(self.dynamichz)
        self.dynamicAOA = AOA
        self.device = device
        self.static = filereader(Plate,'static')

     
    #returns the array with the all the data from 1 dynamic case and loads it into object 
    def Get_Dynamic(self):
        """
        Gets all files in a dynamic case, loads them into the objects and returns them
        """
          
        A = str(self.dynamicAOA)
        F = str(self.dynamichz).replace(".","")
        if self.dynamichz == 3: F = 'Bend'
        if self.dynamichz == 1.5: F = 'Flap'

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
    def Split_Dynamic_Loaded(self,fileselect):
        """
        returns and loads the data for the dynamic case with a given file ID
        """
        splitdata = Separateruns(self.dynamicloaded[fileselect,:],self.dynamichz)
        self.dynamicsplit = splitdata
        return splitdata

    def Split_Dynamic_Unloaded(self,fileselect,data):
        """
        returns the data for the dynamic case with a given file ID

        Dynamic_Loaded is prefered if data is proccessed afterwards
        """
        return Separateruns(data[fileselect,:])

    def Remove_Gaps_Dynamic(self):

        splitdata = GapDetect(self.dynamicsplit)

        self.dynamicsplit = splitdata
        return splitdata

    def Remove_Outliers_Dynamic(self):

        models = self.Dynamicmodelsstrained
        splitdata = self.dynamicsplit

        #Maindata = self.dynamicsplit[1]
        Maindata = EVDetect(models,splitdata)

        self.Dynamicfullload = Maindata
        return Maindata

    def Train_Dynamic_models_2D_Loaded(self,Xname,Yname,epoch = 500,lr = 0.01):
        """
        returns ai models per splitdata and loads them

        """
        models = []
        device = self.device
        for i in trange(len(self.dynamicsplit),desc= 'Training 1st step'):
            maindataset = self.dynamicsplit[i]
            targets = torch.tensor(maindataset[Yname].values, dtype= torch.float64).to(device)
            inputdata = torch.tensor(maindataset[Xname].values,dtype= torch.float64).to(device)
            dataset = torch.utils.data.TensorDataset(inputdata, targets)  
            trained = Train_NN( DynamicNNstage1(len(Xname),len(Yname)).to(device), dataset , epoch, lr)
            #passing model
            models.append(trained)
        self.Dynamicmodelsstrained = models
        return models

    def Train_Dynamic_Model_Main_2D_Loaded(self,Xname,Yname,epoch = 2000,lr = 0.01):
        device = self.device
        for i in trange(1,desc = 'Training Main Model'):
            maindataset = self.Dynamicfullload
            targets = torch.tensor(maindataset[Yname].values, dtype= torch.float64).to(device)
            inputdata = torch.tensor(maindataset[Xname].values,dtype= torch.float64).to(device)
            dataset = torch.utils.data.TensorDataset(inputdata, targets)  
            trained = Train_NN( DynamicNNstage1(len(Xname),len(Yname)).to(device), dataset , epoch, lr)
            trained.eval()


        self.Dynamicmainmodeltrained = trained
        return trained

    def Save_Model(self,model,filename :str):
        plate = self.Plate
        ftype = 'Dynamic'

        A = str(self.dynamicAOA)
        F = str(self.dynamichz).replace(".","")
        if F == '3': F = 'Bend'
        if F == '15': F = 'Flap'
        name = ''
        f = filename.split('_')
        for i in f:
            if i == 'Free': name = f'Free.help'
            if i == 'Locked': name = f'Locked.help'
            if i == 'Pre': name = f'Pre.help'
            if i == 'Rel0': name = f'Rel0.help'
            if i == 'Rel50': name = f'Rel50.help'
            if i == 'Rel100': name = f'Rel100.help'
        Path = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','MODELS',f'Plate {plate}',f'{ftype}',f'A{A}',f'F{F}',name))
        os.makedirs(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','MODELS',f'Plate {plate}',f'{ftype}',f'A{A}',f'F{F}')), exist_ok = True)
        print(Path)
        torch.save(model,Path)
        return

    def Load_Model(self, model_type: str):
        valid_types = ['Free', 'Locked', 'Pre', 'Rel0', 'Rel50', 'Rel100']
        if model_type not in valid_types:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_types}")
        plate = self.Plate
        ftype = 'Dynamic'
        A = str(self.dynamicAOA)
        print(self.dynamichz)
        F = str(self.dynamichz).replace(".0","").replace(".", "")
        name = f'{model_type}.help'
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'MODELS', f'Plate {plate}', f'{ftype}', f'A{A}', f'F{F}', name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model = torch.load(path)
        return model
    

    def Plot_Regression_intermodels_2D_Loaded(self,Xname,Yname):
        """

        Plots the results from the models made during the intrim regression

        """
        models = self.Dynamicmodelsstrained
        device = self.device
        if Yname == "Bending [N-mm]":
            q = 1
        if Yname == "Pot [degree]":
            q = 0


        fig, axs = plt.subplots(len(self.dynamicsplit),sharex=True)
        for i in range(models):
            maindataset = self.dynamicsplit[i]
            x =  np.linspace(0, 3 +  0.5/self.dynamichz , 10000)
            x = torch.from_numpy(x)
            x = x.view(-1, 1).to(device)
            x = x.to(device)
            y = self.Dynamicmodelsstrained[i](x).to(device)
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()[:,q]
            axs[i].plot(x,y,label = 'model')
            axs[i].plot(maindataset[Xname].values,maindataset[Yname].values[:,1])
        plt.xlabel(Xname)
        plt.ylabel(Yname)
        plt.show()
        return

    def Plot_Model_2D_Loaded(self,Xname,Yname,label = 'model'):
        """

        Plots the results from the main model

        """
        
        device = self.device
        model = self.Dynamicmainmodeltrained.to(device)
        if Yname == "Bending [N-mm]":
            q = 1
        if Yname == "Pot [degree]":
            q = 0
        

        hz = float(self.dynamichz)
        x =  np.linspace(0, 3 +  0.5/hz , 10000)
        x = torch.from_numpy(x)
        x = x.view(-1, 1).to(device)
        x = x.to(device)
        y = model(x).to(device)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()[:,q]
        plt.plot(x,y,label = label)
        return

    def Plot_Raw_2D_Loaded(self,Xname:str,Yname:str):
        """

        Plots the raw data, needs a dynamic case to be split

        """
       
        for i in range(len(self.dynamicsplit)):
            maindataset = self.dynamicsplit[i]
            plt.plot(maindataset[Xname].values,maindataset[Yname].values[:])
        plt.xlabel(Xname)
        plt.ylabel(Yname)
        return
    def run_analysis_2D_Quick(self, mode:str = 'quick', types:list = [],Xname:str = 'Time [s]',Yname:str = 'Bending [N-mm]',show:bool = True):
        """

        Runs the 2D proccessing procedure from the models.
        This function can be configured to limited models or to do a quick version of run_analysis_2D()

        2 modes: 
        'quick' will plot all cases.
        'limit' requires types to be an list of the wanted cases selected from: ['Free', 'Locked', 'Pre', 'Rel0', 'Rel50', 'Rel100']

        show:
        True will run plt.show() and set he labels of the axis
        """
        valid_modes = ['quick','limit']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        valid_types = ['Free', 'Locked', 'Pre', 'Rel0', 'Rel50', 'Rel100']
        if mode == 'quick':
            cycle_types = valid_types
        if mode == 'limit':
            for t in types:
                if t not in valid_types:
                    raise ValueError(f"Invalid model type: {types}. Must be one of {valid_types}")
            cycle_types = types
        for case in tqdm(cycle_types,desc='Types'):
            self.Dynamicmainmodeltrained = self.Load_Model(case)
            self.Plot_Model_2D_Loaded(Xname,Yname,case)

        if(show):
            plt.xlabel(Xname)
            plt.ylabel(Yname)
            plt.legend()
            plt.show()
        
        return

    def run_Train_2D(self):
        """
        Trains all models associated with the data class.

        """
        self.Get_Dynamic()
        for i in trange(len(self.Get_Dynamic()),desc='Files completed'):
            fileselect = i
            self.Split_Dynamic_Loaded(fileselect)
            self.Remove_Gaps_Dynamic()
            self.Train_Dynamic_models_2D_Loaded(["Time [s]"],["Pot [degree]","Bending [N-mm]"],1000,0.01)
            self.Remove_Outliers_Dynamic()
            self.Train_Dynamic_Model_Main_2D_Loaded(["Time [s]"],["Pot [degree]","Bending [N-mm]"],5000,0.001)
            self.Save_Model(self.Dynamicmainmodeltrained,self.dynamicloaded[i,1])
        return
    def run_analysis_2D(self):
        """

        Runs the 2D proccessing procedure from scratch.
        it is fully automated

        This function will only plot for this class.

        """
        self.run_Train_2D()
        self.run_analysis_2D_Quick()
        return
            

    
        

    

      

    


