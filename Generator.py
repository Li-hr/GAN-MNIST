import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        n_features = 100
        n_out = 784

        self.hidden0=nn.Sequential(
            nn.Linear(n_features,256),
            nn.LeakyReLU(0.2),
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256,512),
            nn.LeakyReLU(0.2),
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
        )

        self.out = nn.Sequential(
            nn.Linear(1024,n_out),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

