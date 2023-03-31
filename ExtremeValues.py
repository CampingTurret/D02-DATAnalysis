
#import torch
from tqdm.auto import trange
import numpy as np
import pandas as pd
import scipy.signal as sc

def EVDetect(Data):
    """
    Removes extreme values from the dataset

    """
    GoodFrames = []
    indexlist = []
    indexlistlen = []
    b = 50
    a = 0

    for i in trange(len(Data),desc='Evaluating'):
        ylist = Data[i]['Bending [N-mm]']
        indexes = sc.find_peaks_cwt(ylist,20)
        indexlist.append(indexes)
        indexlistlen.append(len(indexes))
        
    
    amountofpeaks = min(indexlistlen)
    remove = np.zeros((len(Data),amountofpeaks))

    for q in range(amountofpeaks):
        y = []
        for i in range(len(Data)):
            y.append(Data[i]['Bending [N-mm]'].iloc[indexlist[i][q]])
        ymedian = np.median(y)
        deviation = a*ymedian + b

        for i in range(len(Data)):
             if deviation  < abs(ymedian - Data[i]['Bending [N-mm]'].iloc[indexlist[i][q]]):
                 remove[i,q] = 1

    removemodels = np.sum(remove,1)
    for i in range(len(Data)):
        if removemodels[i] <1:
            GoodFrames.append(Data[i])


        
    




    #testlen = 10000
    #margin = 0.2
    #x =  np.linspace(0, 3 , testlen)
    #arry = np.empty((testlen,len(Models)))
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #x = torch.from_numpy(x)
    #x = x.view(-1, 1).to(device)
    #x = x.to(device)
    #for i in trange(len(Models),desc='Evaluating'):
    #    m = Models[i].to(device)
    #    y = m(x).to(device)
    #    y = y.detach().cpu().numpy()[:,1]
    #    arry[:,i] = y

    #arry = arry/np.linalg.norm(arry)
    #x = x.detach().cpu().numpy()
    #z = np.sum(arry,1)
    #z = z/len(Models)
    #q = np.empty_like(arry)
    #for i in trange(len(Models),desc='Compairing'):
    #    for j in range(testlen):
    #        if (arry[j,i] < z[j]*(1-margin)) or (z[j]*(1+margin) <  arry[j,i]):
    #            q[j,i] = 1
    #        else:
    #            q[j,i] = 0
    #C = np.sum(q,0)
    #GoodFrames = []

    #for i in range(len(Models)):
    #    if C[i] < 1000:
    #        GoodFrames.append(Data[i])

    GoodData = pd.concat(GoodFrames)
        
    return GoodData
