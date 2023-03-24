import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from layers import PointGNNConv
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MLP


from collections import OrderedDict
from itertools import chain

class PointGNNConv_Edgeweight(PointGNNConv):
    
    def __init__(self, mlp_h: torch.nn.Module,
            mlp_f: torch.nn.Module,
            mlp_g: torch.nn.Module,
            **kwargs,
        ):
        super().__init__(mlp_h,
            mlp_f,
            mlp_g,
            **kwargs,
        )

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, edge_weight) -> Tensor:
        """"""
        # propagate_type: (x: Tensor, pos: Tensor)
        out = self.propagate(edge_index, x=x, pos=pos, edge_weight=edge_weight, size=None)
        out = self.mlp_g(out)
        return x + out

    def message(self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor,
                x_j: Tensor, edge_weight) -> Tensor:
        delta = self.mlp_h(x_i)
        e = torch.cat([pos_j - pos_i + delta, x_j], dim=-1)
        return self.mlp_f(e) if edge_weight is None else edge_weight.view(-1, 1) * self.mlp_f(e)
        



class ConvBlock(torch.nn.Module):
    def __init__(self, 
                mlp_h_space, 
                mlp_f_space, 
                mlp_g_space, 
                aggr, 
                activation) -> None:
        super().__init__()
        
        self.conv = PointGNNConv_Edgeweight(
                        mlp_h=MLP(mlp_h_space), 
                        mlp_f=MLP(mlp_f_space), 
                        mlp_g=MLP(mlp_g_space), 
                        aggr=aggr
                    )

        projected_feature_channels = mlp_h_space[0]
        self.bn = nn.BatchNorm1d(projected_feature_channels)

        self.activation = activation
                    

    def forward(self, arguments):
        x, pos, edge_index, edge_weight = arguments

        x = self.conv(x, pos, edge_index, edge_weight)

        x = self.bn(x)

        x = self.activation(x)

        return x, pos, edge_index, edge_weight




class PointGNN_Normalization(torch.nn.Module):
    def __init__(self, architecture_args = {
        'model_type': 'PointGNN_Normalization',
        'num_node_features': 3,
        'num_classes': 40,
        'projected_feature_channels': 128,
        'embedd_space': [3, 32, 64, 128],
        'n_layers': 2
    }):
        super().__init__()

        pos_channels = architecture_args['num_node_features']
        num_classes = architecture_args['num_classes']
        projected_feature_channels = architecture_args['projected_feature_channels']
        embedd_space = architecture_args['embedd_space']
        n_layers = architecture_args['n_layers']

        # TODO static, not relevant to be in config file        
        activation = nn.LeakyReLU()
                

        self.pos_channels = pos_channels
        self.num_classes = num_classes

        # lin_hidden_space = [300, 300]

        mlp_h_space = [projected_feature_channels, pos_channels]
        mlp_f_space = [projected_feature_channels + pos_channels, projected_feature_channels]
        mlp_g_space = [projected_feature_channels, projected_feature_channels]

        # embedd_space = [3, 64, 128, 300]
        # embedd_space = [pos_channels, projected_feature_channels]
        self.project = MLP(embedd_space, act='LeakyReLU')

        self.convolutions = nn.Sequential(OrderedDict(
            list(chain(*[
                [(f'conv_block_{idx}', 
                    ConvBlock(
                        mlp_h_space, 
                        mlp_f_space, 
                        mlp_g_space, 
                        aggr="max",
                        activation=activation)
                    )
                ]
                for idx
                in range(n_layers)
            ])) 
        ))
        

        lin_hidden_space = [300, 300]
        # self.lin1 = MLP(lin_hidden_space)
        # self.lin2 = MLP(lin_hidden_space)
        # self.lin3 = MLP(lin_hidden_space)

        self.decision = nn.Linear(projected_feature_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        pos = x

        # 1. do projection into high-dimensional space
        x = self.project(x)
        x = F.leaky_relu(x)
        
        # 2. apply various PointGNN convolutions 
        x, pos, edge_index, edge_weight = self.convolutions((x, pos, edge_index, edge_weight))    

        # 3.  
        x = global_mean_pool(x, data.batch)

        # 4. finding decisions 
        x = self.decision(x)

        return x 