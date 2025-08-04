import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_layers, num_inputs=2, num_outputs=1):
        super().__init__()
        layers = []
        for h in hidden_layers:
            layers.append(nn.Linear(num_inputs, h))
            layers.append(nn.ReLU())
            num_inputs = h
        layers.append(nn.Linear(num_inputs, num_outputs))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class MLP_2nn(nn.Module):
    def __init__(self, num_inputs=3, num_outputs=3): # default parameters
        super(MLP_2nn, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 72),
            nn.ReLU(),
            nn.Linear(72, 36),
            nn.ReLU(),
            nn.Linear(36, 8),
            nn.ReLU(),
            nn.Linear(8, num_outputs),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Example usage
    model = MLP(num_inputs=2, num_outputs=3)
    print(model)