from cgitb import text
from torch.utils.data import Dataset
import numpy as np
from torch.autograd import Variable
import torch

class MyDataset(Dataset):
    def __init__(self,opt,flag = 'train'):
        self.opt = opt
        assert flag in ['train','valid','test']
        with open(self.opt.datasets_path+ flag +'.npy','rb') as f :
            scene = np.load(f)
            track = np.load(f)
            audio = np.load(f)
            label = np.load(f)
        self.x = list(zip(scene,track,audio,label))
        
    def __getitem__(self, index):
        assert index < len(self.x)
        return self.x[index]
    
    def __len__(self):
        return len(self.x)

class UpdateDataset(Dataset):
    def __init__(self,h,label):
        self.x = list(zip(h,label))
        
    def __getitem__(self, index):
        assert index < len(self.x)
        return self.x[index]
    
    def __len__(self):
        return len(self.x)

class PromptDataset(Dataset):
    def __init__(self,opt,flag = 'train'):
        self.opt = opt
        with open(self.opt.datasets_path+ flag +'.npy','rb') as f :
            scene = np.load(f)
            track = np.load(f)
            audio = np.load(f)
            label = np.load(f)
        with open(self.opt.prompt_text_path + 'prompt_text.npy','rb') as f :
            text = np.load(f)
        self.x = list(zip(scene,track,audio,text,label))
        
    def __getitem__(self, index):
        assert index < len(self.x)
        return self.x[index]
    
    def __len__(self):
        return len(self.x)
    
class HashDataset(Dataset):
    def __init__(self,opt,flag = 'train'):
        self.opt = opt
        assert flag in ['train','valid','test']
        with open(self.opt.fusion_h_path + flag + '.npy','rb') as f :
            h = np.load(f)
        with open(self.opt.datasets_path+ flag +'.npy','rb') as f :
            scene = np.load(f)
            track = np.load(f)
            audio = np.load(f)
            label = np.load(f)
        self.x = list(zip(h,label))
        
    def __getitem__(self, index):
        assert index < len(self.x)
        return self.x[index]
    
    def __len__(self):
        return len(self.x)
