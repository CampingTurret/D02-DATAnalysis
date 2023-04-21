import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DynamicNNstage1(nn.Module):

    def __init__(self,xlen,ylen,nlen =3):
        super().__init__()
        

        layers = []
        layers.append(nn.Linear(xlen, 100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100, 200))
        layers.append(nn.ReLU())
        for i in range(nlen):
            layers.append(nn.Linear(200, 200))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(200, 100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100, ylen))
        self.linear_relu_stack = nn.Sequential(*layers)
        self.linear_relu_stack.to(torch.float64)
   
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class DynamicNNstage2(nn.Module):

    def __init__(self,xlen,ylen,nlen =5):
        super().__init__()
        

        layers = []
        layers.append(nn.Linear(xlen, 100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100, 100))
        layers.append(nn.ReLU())
        for i in range(nlen):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(100, 100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100, ylen))
        self.linear_relu_stack = nn.Sequential(*layers)
        self.linear_relu_stack.to(torch.float64)
   
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class Dynamicdataset(Dataset):
    """
    Depreciated
    """

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        inp = self.inputs.iloc[idx]
        out = self.targets.iloc[idx]
        return inp, out

