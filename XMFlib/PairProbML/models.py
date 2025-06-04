import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_inputs=2, num_outputs=3): # default parameters
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 72),
            nn.ReLU(),
            nn.Linear(72, 36),
            nn.ReLU(),
            nn.Linear(36, 8),
            nn.ReLU(),
            nn.Linear(8, num_outputs)
        )

    def forward(self, x):
        return self.net(x)