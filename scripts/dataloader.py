
import numpy as np

def filereader(plate, type, file):
    import pandas as pd
    import os

    if(type == 'Static' or type == 'static'): 
        type = 'Static'
        filepath = f"../DATA/Plate {plate}/{type}/{filename}.txt"

    elif(type == 'Dynamic' or type == 'dynamic'): 
        type = 'Dynamic'
        filepath = f"../DATA/Plate {plate}/{type}/{Angle}/{Frequency}/{filename}.txt"


    
    read = pd.read_csv(filepath)

    return read


class data:

    def __init__(self,Plate):
        self.Plate = Plate
        
        
    


