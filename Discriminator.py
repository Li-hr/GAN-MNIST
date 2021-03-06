import torch.nn as nn
import numpy as np
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        n_features = 784
        n_out = 1

        self.hidden0=nn.Sequential(
            nn.Linear(n_features,1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )

        self.out=nn.Sequential(
            torch.nn.Linear(256,n_out),
            nn.Sigmoid()

        )

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x





