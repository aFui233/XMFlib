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

if __name__ == "__main__":
    # Example usage
    model = MLP(num_inputs=2, num_outputs=3)
    print(model)