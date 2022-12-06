from typing import List, Optional, Tuple, Union

import torch

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import MLP


class PointGNNConv(MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_ paper, where the graph is
    statically constructed using radius-based cutoff.
    The relative position is used in the message passing step to introduce global translation invariance.
    To also counter shifts in the local neighborhood of the center vertex, the authors propose an alignment offset.

    Args:
        state_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        MLP_h (MLP): Calculate alignment off-set  :math:`\Delta` x
        MLP_f (MLP): Calculate edge update using relative coordinates, alignment off-set 
            and neighbor feature.
        MLP_g (MLP): Transforms aggregated messages.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          node coordinates :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,   
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        state_channels: int,
        MLP_h: MLP,
        MLP_f: MLP,
        MLP_g: MLP,
        aggr: Optional[Union[str, List[str], Aggregation]] = "max",
        **kwargs
    ):
        self.state_channels = state_channels

        super().__init__(aggr, **kwargs)

        self.lin_h  = MLP_h 
        self.lin_f = MLP_f 
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