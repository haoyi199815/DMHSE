# random generation
import torch
import numpy as np
from easydict import EasyDict
import yaml #yaml
import clip
from torch.utils.data import DataLoader
from datasets import *
from utils import Label2Text
from tqdm import *
import logging

def get_prompt_text(opt):
    get_text_time = 0
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True) #the times
    torch.cuda.set_device(opt.device)
    start.record()
    print(" * * * Get TextPrompt Features * * * ")
    model,_ = clip.load("ViT-B/32",device=opt.device)
    train_dataset = MyDataset(opt,'train')
    train_DataLoader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)
    is_start = True
    batchs = enumerate(tqdm(train_DataLoader, position=0, desc="batch", leave=False, colour='Green', ncols=80))
    for idx, (scene,track,audio,label) in batchs:
        prompt_text = Label2Text(label)
        text_token = clip.tokenize(prompt_text).to(opt.device)
        with torch.no_grad():
            text_features = model.encode_text(text_token)
        if is_start:
            all_features = text_features
            is_start = False
        else:
            all_features = torch.cat((all_features, text_features), 0)
        
    with open(opt.prompt_text_path + 'prompt.npy', 'wb') as f:
        np.save(f,all_features.cpu().detach().numpy())
    
    prompt_dataset = PromptDataset(opt,'train')
    
    end.record()
    torch.cuda.synchronize()
    get_text_time = start.elapsed_time(end)
    logging.info('Get_Text_Time: {Get_Text_Time:.4f}\t'.format(Get_Text_Time = get_text_time))
    
    return prompt_dataset