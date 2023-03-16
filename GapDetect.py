from msilib.schema import Condition
from tqdm.auto import tqdm,trange
import numpy as np
import pandas as pd

def GapDetect(DirtyData:list):
    """
    Removes Gaps in data due to errors in data aquisition
    """
    #DirtyData is a list of a list 
    CleanData = []
    for i in trange(len(DirtyData)):
        print(DirtyData)
        condition = False
        if condition:
            CleanData = CleanData.append(DirtyData[i])
            


    print(CleanData)
    return CleanData
