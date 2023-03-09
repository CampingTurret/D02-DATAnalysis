import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class DynamicNNstage1(nn.Module):

    def __init__(self,xlen,ylen):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(xlen, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, ylen),
        )
   
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class Dynamicdataset(Dataset):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        inp = self.inputs.iloc[idx]
        out = self.targets.iloc[idx]
        return inp, out

