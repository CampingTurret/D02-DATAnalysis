from msilib.schema import Condition
from tqdm.auto import tqdm,trange
import numpy as np
import pandas as pd



def GapDetect(DirtyData: list):
    """
    Removes Gaps in data due to errors in data acquisition
    """
    indinces = []
    CleanData = []
    for i in range(len(DirtyData)):
        indinces.append(len(DirtyData[i]))
    avglength = np.mean(indinces)
    
    for i in range(len(DirtyData)):
        if len(DirtyData[i]) < (avglength + 200) and len(DirtyData[i]) > (avglength - 200):
            CleanData.append(DirtyData[i])
    print(CleanData)
    return CleanData

