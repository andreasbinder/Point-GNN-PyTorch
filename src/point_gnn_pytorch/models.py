import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from layers import PointGNNConv
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MLP


class PointGNN(torch.nn.Module):
    def __init__(self, 
                num_node_features=3, 
                num_classes=40, 
                MLP_h: list = [3, 64, 3],
                MLP_f: list = [6, 64, 3],
                MLP_g: list = [3, 64, 3]
        ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_classes = num_classes

        self.conv1 = PointGNNConv(state_channels=num_node_features, MLP_h=MLP(MLP_h), MLP_f=MLP(MLP_f), MLP_g=MLP(MLP_g), aggr="max")
        self.conv2 = PointGNNConv(state_channels=num_node_features, MLP_h=MLP(MLP_h), MLP_f=MLP(MLP_f), MLP_g=MLP(MLP_g), aggr="max")
        self.conv3 = PointGNNConv(state_channels=num_node_features, MLP_h=MLP(MLP_h), MLP_f=MLP(MLP_f), MLP_g=MLP(MLP_g), aggr="max")

        self.linear = nn.Linear(num_node_features, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        pos = x

        x = self.conv1(x, pos, edge_index)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.conv2(x, pos, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, pos, edge_index)

        x = global_mean_pool(x, data.batch)

        x = self.linear(x)

        return F.log_softmax(x, dim=1)