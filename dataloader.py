
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
        dirpath = os.path.join('DATA','Files',f'Plate {plate}',f'{type}')
        print(dirpath)
        dirpath = dirpath.replace(r"\\",r'\a'.replace('a',''))
        print(dirpath)
        input()
        allfiles = [f for f in os.listdir(dirpath)]
        print(allfiles)
        input()
        for f in allfiles:
            read.append(pd.read_csv(f))

    #for dynamic cases
    elif(type == 'Dynamic' or type == 'dynamic'): 
        type = 'Dynamic'
        filepath = f"../DATA/Plate {plate}/{type}/{Angle}/{Frequency}/{filename}.txt"

    print(read)
    
    

    return read


class data:

    def __init__(self,Plate):
        self.Plate = Plate
        
filereader('A','static')
    


