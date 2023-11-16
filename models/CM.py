import torch.nn as nn
from .mixer_mlp import MixerBlock

#  Base model for Complete Multi-modal learning
class CMNet(nn.Module):
    
    def __init__(self, in_dim):

        super(CMNet, self).__init__()
        self.in_dim = in_dim
        self.fc_scene = MixerBlock(self.in_dim)
        self.fc_track = MixerBlock(self.in_dim)
        self.fc_audio = MixerBlock(self.in_dim)
        #权重
        self.weights = self.init_weights()

    #初始化参数
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): #用来判断m是否是Linear线性层
                nn.init.xavier_normal_(m.weight) #初始化
                nn.init.constant_(m.bias, 0.0) #初始化

    def forward(self, H):
        """
        :param H: #视频特征
        :return: x_view: #各模态的视频特征
        """
        # h = self.down_dim(H)
        x_scene = self.fc_scene(H)
        x_track = self.fc_track(H)
        x_audio = self.fc_audio(H)
        
        return x_scene,x_track,x_audio