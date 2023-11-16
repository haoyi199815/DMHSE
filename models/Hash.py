import torch
import torch.nn as nn
import math

# Base model for Hash Coding
class HashNet(nn.Module):

    def __init__(self, feature_dim, binary_dim):
    
        super(HashNet, self).__init__()
        self.feature_dim = feature_dim
        self.binary_dim = binary_dim

        self.fc1 = nn.Linear(self.feature_dim, self.feature_dim)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(self.feature_dim, self.feature_dim)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(self.feature_dim, self.binary_dim)
        self.dropout = nn.Dropout(0.5)
        self.hash_layer = nn.Sequential(self.fc1, self.activation1, self.dropout, self.fc2, self.activation2, self.fc3)

        self.weights = self.init_weights()
        self.gamma = 100.0
        # self.gamma = 10.0
        self.power = 0.5
        self.init_scale = 1.0
        self.last_layer = nn.Tanh()
        # self.act = nn.LeakyReLU(0.2)
        self.act = nn.LeakyReLU(0.2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):#用来判断m是否是Linear线性层
                nn.init.xavier_normal_(m.weight)#初始化
                nn.init.constant_(m.bias, 0.0)#初始化

    def forward(self, H, ep):
        t_h = self.hash_layer(H)
        self.scale = self.init_scale * (math.pow((1.+self.gamma*ep), self.power))
        t = self.last_layer(self.scale * t_h)
        b = torch.sign(t)
        return t, b

