import torch
import torch.nn as nn
from .mixer_mlp import MixerBlock
import clip
import numpy as np
from torch.nn import Parameter

# Freezed TextEncoder From CLIP for Prompt Learning
class TextEncoder(nn.Module):
    
    def __init__(self, opt, in_dim):

        super(TextEncoder, self).__init__()
        self.opt = opt
        self.in_dim = in_dim
        self.down_dim = nn.Linear(self.in_dim, 512)
        self.text_coder = MixerBlock(512)
        self.logit_scale = Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.CLIP_loss = nn.CrossEntropyLoss()
        #权重
        self.weights = self.init_weights()

    #初始化参数
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): #用来判断m是否是Linear线性层
                nn.init.xavier_normal_(m.weight) #初始化
                nn.init.constant_(m.bias, 0.0) #初始化

    def forward(self, h):
        
        video_features = self.down_dim(h)
        
        predict_text = self.text_coder(video_features)

        # normalized features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        # text_features = text_features.float() / (text_features.norm(dim=1, keepdim=True)).float()
        text_features = predict_text / predict_text.norm(dim=1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_video.t()
        labels = torch.from_numpy(np.arange(logits_per_video.shape[0])).cuda(self.opt.device)
        # labels = torch.eye(logits_per_video.shape[0]).cuda(self.opt.device)
        loss_h = self.CLIP_loss(logits_per_video,labels)
        loss_l = self.CLIP_loss(logits_per_text,labels)
        clip_loss = (loss_h + loss_l)/2
        
        return predict_text, clip_loss