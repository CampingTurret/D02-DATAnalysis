
from neuralnet import DynamicNNstage1
import torch
from tqdm.auto import tqdm,trange
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk

def EVDetect(Models, Data):
    """
    Removes extreme values from the dataset

    """
    testlen = 10000
    margin = 0.2
    x =  np.linspace(0, 3 , testlen)
    arry = np.empty((testlen,len(Models)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(x)
    x = x.view(-1, 1).to(device)
    x = x.to(device)
    for i in trange(len(Models),desc='Evaluating'):
        m = Models[i].to(device)
        y = m(x).to(device)
        y = y.detach().cpu().numpy()[:,1]
        arry[:,i] = y

    arry = arry/np.linalg.norm(arry)
    x = x.detach().cpu().numpy()
    z = np.sum(arry,1)
    z = z/len(Models)
    q = np.empty_like(arry)
    for i in trange(len(Models),desc='Compairing'):
        for j in range(testlen):
            if (arry[j,i] < z[j]*(1-margin)) or (z[j]*(1+margin) <  arry[j,i]):
                q[j,i] = 1
            else:
                q[j,i] = 0
    C = np.sum(q,0)
    GoodFrames = []

    for i in range(len(Models)):
        if C[i] < 1000:
            GoodFrames.append(Data[i])

    GoodData = pd.concat(GoodFrames)
        
    return GoodData
