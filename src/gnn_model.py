import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class TrafficAnomalyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TrafficAnomalyGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = Linear(out_channels, 1)  # Output anomaly score

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.fc(x)
        return x


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, torch.zeros(out.size()))  # Assume zero is non-anomalous
    loss.backward()
    optimizer.step()
    return loss
