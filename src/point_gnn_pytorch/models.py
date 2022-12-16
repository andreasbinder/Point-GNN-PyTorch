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

        lin_hidden_space = [300, 300]

        mlp_h_space = [300, 3]
        mlp_f_space = [303, 300]
        mlp_g_space = [300, 300]

        embedd_space = [3, 64, 128, 300]
        self.project = MLP(embedd_space)

        self.conv1 = PointGNNConv(mlp_h=MLP(mlp_h_space), mlp_f=MLP(mlp_f_space), mlp_g=MLP(mlp_g_space), aggr="max")
        self.conv2 = PointGNNConv(mlp_h=MLP(mlp_h_space), mlp_f=MLP(mlp_f_space), mlp_g=MLP(mlp_g_space), aggr="max")
        self.conv3 = PointGNNConv(mlp_h=MLP(mlp_h_space), mlp_f=MLP(mlp_f_space), mlp_g=MLP(mlp_g_space), aggr="max")

        lin_hidden_space = [300, 300]
        self.lin1 = MLP(lin_hidden_space)
        self.lin2 = MLP(lin_hidden_space)
        self.lin3 = MLP(lin_hidden_space)

        self.decision = nn.Linear(300, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        pos = x

        x = self.project(x)

        x = self.conv1(x, pos, edge_index)
        x = F.relu(x)
        x = self.lin1(x)

        x = F.dropout(x, training=self.training)

        x = self.conv2(x, pos, edge_index)
        x = F.relu(x)
        x = self.lin2(x)

        x = F.dropout(x, training=self.training)

        x = self.conv3(x, pos, edge_index)
        x = F.relu(x)
        x = self.lin3(x)


        x = global_mean_pool(x, data.batch)

        x = self.decision(x)

        return F.log_softmax(x, dim=1)


# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from layers import PointGNNConv
# import torch.nn as nn
# from torch_geometric.nn import global_mean_pool
# from torch_geometric.nn import MLP


# class PointGNN(torch.nn.Module):
#     def __init__(self, 
#                 num_node_features=3, 
#                 num_classes=40, 
#                 MLP_h: list = [3, 64, 3],
#                 MLP_f: list = [6, 64, 3],
#                 MLP_g: list = [3, 64, 3]
#         ):
#         super().__init__()

#         self.num_node_features = num_node_features
#         self.num_classes = num_classes

#         self.conv1 = PointGNNConv(state_channels=num_node_features, MLP_h=MLP(MLP_h), MLP_f=MLP(MLP_f), MLP_g=MLP(MLP_g), aggr="max")
#         self.conv2 = PointGNNConv(state_channels=num_node_features, MLP_h=MLP(MLP_h), MLP_f=MLP(MLP_f), MLP_g=MLP(MLP_g), aggr="max")
#         self.conv3 = PointGNNConv(state_channels=num_node_features, MLP_h=MLP(MLP_h), MLP_f=MLP(MLP_f), MLP_g=MLP(MLP_g), aggr="max")

#         self.linear = nn.Linear(num_node_features, num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         pos = x

#         x = self.conv1(x, pos, edge_index)
#         x = F.relu(x)

#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, pos, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv3(x, pos, edge_index)

#         x = global_mean_pool(x, data.batch)

#         x = self.linear(x)

#         return F.log_softmax(x, dim=1)