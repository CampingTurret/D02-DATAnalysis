
from neuralnet import DynamicNNstage1
import torch
from tqdm.auto import tqdm,trange
import numpy as np
import pandas as pd

def EVDetect(Models, Data):
    """
    Removes extreme values from the dataset

    """
    indexlist = []
    for i in trange(len(Models)):

        extremecondition = True
        if extremecondition:
            indexlist.append(i)

    GoodData = Data[indexlist]
    return GoodData
