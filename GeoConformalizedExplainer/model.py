import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += residual
        out = self.activation(out)
        return out


class MultipleTargetRegression(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=3):
        super(MultipleTargetRegression, self).__init__()
        self.input_layer = nn.Linear(input_dim, 256)
        self.output_layer = nn.Linear(256, output_dim)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_blocks)])

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.residual_blocks(x)
        return self.output_layer(x)


class MultipleTargetRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_dataloader(X, y, batch_size, num_workers=4):
    dataloader = DataLoader(MultipleTargetRegressionDataset(X, y), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return dataloader
