
import numpy as np
from pathlib import Path

def filereader(plate, type):
    import pandas as pd
    import os
    from os.path import isfile, join
    read = []


    #for static cases
    if(type == 'Static' or type == 'static'): 
        type = 'Static'
        platenameformat = plate.replace(" ","_")
        dirpath = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','DATA',f'Plate {plate}',f'{type}'))
        allfiles = [f for f in os.listdir(dirpath)]
        print(allfiles)
        for f in allfiles:
            tempf = os.path.join(dirpath,f)
            read.append(pd.read_csv(tempf,encoding = 'cp1252'))

    #for dynamic cases
    elif(type == 'Dynamic' or type == 'dynamic'): 
        type = 'Dynamic'
        filepath = f"../DATA/Plate {plate}/{type}/{Angle}/{Frequency}/{filename}.txt"

    print(read)
    
    

    return read


class data:

    def __init__(self,Plate):
        self.Plate = Plate
        self.static = filereader(Plate,'static')
        self.dynamic = filereader(Plate,'Dynamic')
        
Pa = data('A')
print(Pa.static)
    


