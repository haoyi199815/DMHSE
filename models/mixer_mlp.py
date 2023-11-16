import torch.nn as nn
from functools import partial

class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, norm_layer=nn.BatchNorm1d,act_layer=nn.Sigmoid):
        super(MLP,self).__init__()
        out_features = out_features
        hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MixerBlock(nn.Module):

    def __init__(self, dim, mlp_layer=MLP, 
                 norm_layer=nn.BatchNorm1d, act_layer=nn.Sigmoid):
        super(MixerBlock,self).__init__()
        
        self.norm1 = norm_layer(dim)
        self.mlp_dim = mlp_layer(dim, 4*dim, dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)
        self.mlp_dim2 = mlp_layer(dim, 4*dim, dim, act_layer=act_layer)
        

    def forward(self, x):
        x = self.mlp_dim(self.norm1(x)) + x
        x = self.mlp_dim2(self.norm2(x)) + x
        return x


        