import torch
import torch.nn as nn
import math

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc2 = nn.Linear(2, 2)
        nn.init.constant_(self.fc2.weight[0], 0.5)  # set the slope of the first output
        nn.init.constant_(self.fc2.weight[1], 0.3)  # set the slope of the second output
        nn.init.constant_(self.fc2.bias, 1.0)       # set the intercept
        self.amplitude_factor = 1
        self.phase = 0
        
    def forward(self, x):
        time = x[:, :, 0]
        freq = x[:, :, 1]
        amplitude = freq * self.amplitude_factor
        cosine_wave = amplitude.unsqueeze(-1) * torch.cos(2 * math.pi * freq.unsqueeze(-1) * time.unsqueeze(1) + self.phase)
        zeros = torch.zeros_like(cosine_wave)
        out = torch.stack((zeros, cosine_wave), dim=-1)
        return out

# create an instance of the model
model = TestModel()

# save the model to a file
torch.save(model, 'MODELS3D/test_model.pt')