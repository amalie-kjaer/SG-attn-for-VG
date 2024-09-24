import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        with open('./data/prepositions.txt', 'r') as f:
            self.prepositions = f.readlines()
        self.prepositions = [word.strip() for line in self.prepositions for word in line.split(',')]
        print(self.prepositions)
        self.in_dim = 7 # TODO: set input dimension
        self.out_dim = len(self.prepositions)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        with open('./data/prepositions.txt', 'r') as f:
            self.prepositions = f.readlines()
        self.prepositions = [word.strip() for line in self.prepositions for word in line.split(',')]
        
        self.in_dim = 7  # TODO: set input dimension
        self.out_dim = len(self.prepositions)
        
        # Model layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.in_dim, 512)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, self.out_dim)

    def forward(self, x):
        x = self.flatten(x)
        
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        logits = self.fc3(x)
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = MLP().to(device)
    print(model)