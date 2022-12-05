from typing import List, Optional, Tuple, Union

import torch

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import MLP


class PointGNNConv(MessagePassing):
   
    def __init__(
        self,
        state_channels: Union[int, Tuple[int, int]],
        MLP_h: MLP = MLP([3, 64, 3]),
        MLP_f: MLP = MLP([6, 64, 3]),
        MLP_g: MLP = MLP([3, 64, 3]),
        aggr: Optional[Union[str, List[str], Aggregation]] = "max",
    ):
        self.state_channels = state_channels

        super().__init__(aggr)

        # calculate off-set delta x
        self.lin_h  = MLP_h 
        
        # calculate edge update
        self.lin_f = MLP_f 
        
        # calculate final state update
        self.lin_g = MLP_g 

    def forward(self, x, pos, edge_index):
        out = self.propagate(edge_index, x=x, pos=pos)
        
        out = self.lin_g(out)
        return x + out

    def message(self, pos_j, pos_i, x_i, x_j):
        delta = self.lin_h(x_i)

        e = torch.cat([pos_j - pos_i + delta, x_j], dim=-1)

        e = self.lin_f(e)
        return e