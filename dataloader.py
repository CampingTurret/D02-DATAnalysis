

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
            data.set_axis(headerrow,axis = 1,copy=False)
            read.append(data)

    #for dynamic cases
    elif(type == 'Dynamic' or type == 'dynamic'): 
        type = 'Dynamic'
        dirpath = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','DATA',f'Plate {plate}',f'{type}',f'A{Angle}',f'F{Frequency}'))
        allfiles = [f for f in os.listdir(dirpath)]
        for f in allfiles:
            tempf = os.path.join(dirpath,f)
            data = pd.read_csv(tempf,encoding = 'cp1252')
            data = data.dropna(axis = 1)
            data.set_axis(headerrow,axis = 1,copy=False)
            read.append(data)
    return read


class data:

    def __init__(self,Plate):
        self.Plate = Plate
        self.static = filereader(Plate,'static')
     
    def Get_Dynamic(self,angle,frequency):
          
        A = str(angle)
        F = str(frequency).replace(".","")

        Dynamiccase = filereader(self.Plate,'Dynamic',A,F)
        return Dynamiccase

    def Split_static(self):
        


        self.static_free = 0
        self.static_locked = 0

    
        
Pa = data('A')
print(Pa.static)
#print(Pa.Get_Dynamic(0,0.5))
    


