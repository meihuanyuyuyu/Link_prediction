from turtle import forward
from torch_geometric.nn import GCN,GraphSAGE,LightGCN
import torch.nn as nn

class Graph2GCN(nn.Module):
    def __init__(self,in_c,out_c,num_layers) -> None:
        super().__init__()
        self.conv = GCN(in_c,256,num_layers=num_layers,out_channels=out_c)
    
    def forward(self,x,edge_index):
        return self.conv(x,edge_index)
        

class SAGE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net=GraphSAGE(768,256,10,512)
    
    def forward(self,x,edge_index):
        return self.net(x,edge_index)
                