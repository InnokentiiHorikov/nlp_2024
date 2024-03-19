import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model =  nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels = 32,
                               kernel_size=3, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            

            nn.Conv2d(in_channels=32, out_channels = 64,
                               kernel_size=3, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
            
            nn.Flatten(),
            nn.Linear(64*10*10, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
            )

        
        
    def forward(self, x):
        x = self.model(x)
        return x
