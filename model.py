import random
import torch
from models import CM, Hash, TextEncoder
from torch.optim import SGD,Adam
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *
import shutil
from collections import OrderedDict
import numpy as np
import logging
import math
from tqdm import *
from torch.utils.data import DataLoader
from datasets import *


#固定随机种子
random.seed(613)
np.random.seed(613)
torch.manual_seed(613)
torch.cuda.manual_seed(613)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

class DMHSE(object):
    def __init__(self,opt):
        self.opt = opt
        self.CM = CM.CMNet(self.opt.feature_dim)
        self.params_cpm = list(self.CM.parameters())
        self.cpm_optimizer = SGD(self.params_cpm, lr= self.opt.fusion_lr,
                                   momentum=opt.optimizer.momentum,
                                   weight_decay=opt.optimizer.weight_decay,
                                   nesterov=opt.optimizer.nesterov
                                   )
        self.TextEncoder = TextEncoder.TextEncoder(opt,opt.feature_dim)
        self.params_text = list(self.TextEncoder.parameters())
        self.optimizer_textcoder = Adam(self.params_text,lr=self.opt.text_lr,betas=(0.9,0.999))
        self.Hash = Hash.HashNet(opt.feature_dim, opt.binary_dim)
        self.params_hashlearning = list(self.Hash.parameters())
        self.optimizer_hashlearning = Adam(self.params_hashlearning,lr=self.opt.hash_lr,betas=(0.9,0.999))

        if torch.cuda.is_available():
            self.CM.cuda(opt.device)
            self.TextEncoder.cuda(opt.device)
            self.Hash.cuda(opt.device)

    def train_start(self):
        self.CM.train()
        self.TextEncoder.train()
        self.Hash.train()
    def eval_start(self):
        self.CM.eval()
        self.TextEncoder.eval()
        self.Hash.eval()

    def train(self, opt, prompt_dataset, train_dataset, true_hash, valid_dataset, model, writer):
        train_time = 0
        best_map = 0
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True) #the times
        print(" * * * Start Train * * * ")
        targets = torch.load(true_hash) #转为cuda，tensor
        global random_center
        random_center = torch.randint_like(targets[0], 2)
        torch.cuda.set_device(opt.device)
        start.record()
        print(" * * * Start Fusion Train * * * ")
        self.H_train,_,_ = self.H_init()
        fusion_dataloader = DataLoader(prompt_dataset, batch_size=opt.batch_size, shuffle=False)
        # FoldCoder
        for fusion_epoch in trange(opt.fusion_epochs, position=0, desc="epoch", leave=False, colour='yellow', ncols=80):
            
            model.train_start()
            
            loss_recover_epoch = [] #特征融合阶段总损失
            loss_clip_epoch = []
            loss_semantic_epoch = []
            
            fusion_batchs = enumerate(tqdm(fusion_dataloader, position=1, desc="batch", leave=False, colour='red', ncols=80))
            for idx,(scene, track, audio, text, label) in fusion_batchs:
                scene = torch.Tensor(scene).cuda(self.opt.device)
                track = torch.Tensor(track).cuda(self.opt.device)
                audio = torch.Tensor(audio).cuda(self.opt.device)
                text = torch.Tensor(text).cuda(self.opt.device)
                start_idx,end_idx = idx * opt.batch_size, (idx+1) * opt.batch_size
                h = self.H_train[start_idx: end_idx, ...]
                h.requires_grad = True
                view_data = dict()
                view_data[0], view_data[1], view_data[2] = scene, track, audio
                self.train_h_optimizer = SGD([h], lr= self.opt.fusion_lr, 
                                   momentum=opt.optimizer.momentum,
                                   weight_decay=opt.optimizer.weight_decay,
                                   nesterov=opt.optimizer.nesterov
                                   )

                for i in range(self.opt.cm_loop):
                    loss_recover = self.loss_criterion_reconsruction(h, view_data)
                    self.cpm_optimizer.zero_grad()
                    loss_recover.backward(retain_graph=True)
                    self.cpm_optimizer.step()
                    
                    text_pred,loss_clip = self.TextEncoder(h)
                    loss_semantic = torch.norm((text_pred-text), 2.0)
                    loss_text = self.opt.param_clip * loss_clip + self.opt.param_semantic * loss_semantic
                    self.optimizer_textcoder.zero_grad()
                    loss_text.backward(retain_graph=True)
                    self.optimizer_textcoder.step()

                    loss_recover = self.loss_criterion_reconsruction(h, view_data)
                    _,loss_clip = self.TextEncoder(h)
                    loss_update = loss_recover + self.opt.param_gen * loss_clip
                    self.train_h_optimizer.zero_grad()
                    loss_update.backward(retain_graph=True)
                    self.train_h_optimizer.step()
                loss_recover = self.loss_criterion_reconsruction(h, view_data)
                text_pred,loss_clip = self.TextEncoder(h)
                loss_semantic = torch.norm((text_pred-text), 2.0)
                loss_recover_epoch.append(loss_recover.item())
                loss_clip_epoch.append(loss_clip.item())
                loss_semantic_epoch.append(loss_semantic.item())
                
            loss_recover_avg = np.average(loss_recover_epoch)
            loss_clip_avg = np.average(loss_clip_epoch)
            loss_semantic_avg = np.average(loss_semantic_epoch)
            # Log Info
            logging.info(
                'Fusion_Train: Epoch: [{0}]\t'
                'Loss_recover_avg: {loss_recover_avg:.4f}\t'
                'Loss_clip_avg: {loss_clip_avg:.4f}\t'
                'Loss_semantic_avg: {loss_semantic_avg:.4f}\t'
                'lr: {lr:.4f}\n'.format(
                fusion_epoch, loss_recover_avg=loss_recover_avg, loss_clip_avg=loss_clip_avg, loss_semantic_avg=loss_semantic_avg, lr=self.train_h_optimizer.state_dict()['param_groups'][0]['lr']))
            writer.add_scalar('Fusion_Loss_recover', loss_recover_avg, fusion_epoch)
            writer.add_scalar('Fusion_Loss_clip', loss_clip_avg, fusion_epoch)
            writer.add_scalar('Fusion_Loss_semantic', loss_semantic_avg, fusion_epoch)
        
        # Generate Fusion_H
        print(" * * * Generate Fusion Feature * * * ")
        database_dataset = self.gen_fusion_h(model, 'train', train_dataset)
        print(" * * * Database Feature Generated * * * ")
        query_dataset = self.gen_fusion_h(model, 'valid', valid_dataset)
        print(" * * * Query Feature Generated * * * ")
        query_dataloader = DataLoader(query_dataset, batch_size=opt.batch_size, shuffle=True)
        
        print(" * * * Start Hash * * * ")
        # HashCoder
        for hash_epoch in trange(opt.hash_epochs, position=0, desc="epoch", leave=False, colour='yellow', ncols=80):
            model.train_start()
            
            hash_dataloader_1 = DataLoader(database_dataset, batch_size=opt.batch_size, shuffle=True)
            hash_dataloader_2 = DataLoader(database_dataset, batch_size=opt.batch_size, shuffle=True)
            
            if hash_epoch % 5 == 0 and hash_epoch != 0:  # 每迭代5次，更新一次学习率        
                for params in self.optimizer_hashlearning.param_groups:             
                    # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9            
                    params['lr'] *= 0.9
            
            loss_Q_epoch = [] #约束损失
            loss_central_epoch = [] #哈希中心损失
            loss_similar_epoch = [] #视频相似性损失
            loss_train_epoch = [] #训练阶段总损失
            
            hash_batchs = enumerate(tqdm(hash_dataloader_1, position=1, desc="batch", leave=False, colour='red', ncols=80))
            for idx,(h, label) in hash_batchs:
                if idx == 0:
                    new_h = h
                    new_label = label
                else:
                    new_h = torch.cat((new_h,h), 0)
                    new_label = torch.cat((new_label,label), 0)
                h_1 = torch.Tensor(h).cuda(self.opt.device)
                label_1 = torch.Tensor(label).cuda(self.opt.device)           
                h_2,label_2 = next(iter(hash_dataloader_2))
                h_2 = torch.Tensor(h_2).cuda(self.opt.device)
                label_2 = torch.Tensor(label_2).cuda(self.opt.device)
                h = torch.cat((h_1,h_2), 0)         
                label = torch.cat((label_1,label_2), 0)
                center = self.Hash_center_multilables(label, targets)
                center = Variable(center).cuda(self.opt.device)
                
                for i in range(self.opt.hash_loop):
                    t,_ = self.Hash(h, hash_epoch)
                    loss_train,loss_central,loss_similar,loss_Q = self.loss_criterion_train(label, t, center)
                    self.optimizer_hashlearning.zero_grad()
                    loss_train.backward(retain_graph=True)
                    self.optimizer_hashlearning.step()
                       
                loss_central_epoch.append(loss_central.item())
                loss_similar_epoch.append(loss_similar.item())
                loss_Q_epoch.append(loss_Q.item())
                loss_train_epoch.append(loss_train.item())
                
            loss_central_avg = np.average(loss_central_epoch)
            loss_similar_avg = np.average(loss_similar_epoch)
            loss_Q_avg = np.average(loss_Q_epoch)
            loss_train_avg = np.average(loss_train_epoch)
            
            with torch.no_grad():
                base_B, base_label = self.predict_hash_code(model,hash_dataloader_1,hash_epoch)
                query_B, query_label = self.predict_hash_code(model,query_dataloader,hash_epoch)
            mAP_K,_,_ = mean_average_precision(base_B, query_B, base_label, query_label, self.opt.K)
            
            logging.info(
                'Hash_Train: Epoch: [{0}]\t'
                'Loss_central_avg: {loss_central_avg:.4f}\t'
                'Loss_similar_avg: {loss_similar_avg:.4f} \t'
                'Loss_Q_avg: {loss_Q_avg:.4f} \t'
                'Loss_train_avg: {loss_train_avg:.4f} \t'
                'mAP_K: {mAP_K:.4f}\t'
                'lr: {lr:.4f}\n'.format(
                hash_epoch, loss_central_avg=loss_central_avg, loss_similar_avg=loss_similar_avg, loss_Q_avg=loss_Q_avg,
                loss_train_avg=loss_train_avg ,mAP_K=mAP_K, lr=self.optimizer_hashlearning.state_dict()['param_groups'][0]['lr']))
            writer.add_scalar('Hash_Train_Loss_central', loss_central_avg, hash_epoch)
            writer.add_scalar('Hash_Train_Loss_similar', loss_similar_avg, hash_epoch)
            writer.add_scalar('Hash_Train_Loss_Q', loss_Q_avg, hash_epoch)
            writer.add_scalar('Hash_Train_Loss_train', loss_train_avg, hash_epoch)
            writer.add_scalar('mAP_K', mAP_K, hash_epoch)
            
            is_best = mAP_K > best_map
            best_map = max(mAP_K,best_map)
            if is_best:
                self.best_epoch = hash_epoch
                torch.save(
                    {'model':self.state_dict(),
                     'best_epoch':self.best_epoch},
                     opt.ckpt_path + '/' + 'best_hash.pth')
                end.record()
            
            database_dataset = UpdateDataset(new_h,new_label)
            pass
        
        logging.info('Best_map: {best_map:.4f}\t'
                     'Best_epoch: {best_epoch:.4f}\t'.format(best_map=best_map,best_epoch=self.best_epoch))
        torch.cuda.synchronize()
        train_time = start.elapsed_time(end)
        logging.info('Train_time: {train_time:.4f}\t'.format(train_time = train_time))
    
    def test(self, opt, valid_dataset, test_dataset, model):
        test_time = 0
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True) #the times
        best_params = torch.load(opt.ckpt_path + '/' + 'best_hash.pth')
        self.load_state_dict(best_params['model'])
        self.best_epoch = best_params['best_epoch']
        print(" * * * Start Test * * * ")
        torch.cuda.set_device(opt.device)
        start.record()
        # 生成融合特征
        print(" * * * Generate Fusion Feature * * * ")
        database_dataset = self.gen_fusion_h(model, 'test', test_dataset)
        print(" * * * Database Feature Generated * * * ")
        query_dataset = self.gen_fusion_h(model, 'valid', valid_dataset)
        print(" * * * Query Feature Generated * * * ")
        database_dataloader = DataLoader(database_dataset, batch_size=self.opt.batch_size, shuffle=True)
        query_dataloader = DataLoader(query_dataset, batch_size=opt.batch_size, shuffle=True)
        # 生成哈希码
        with torch.no_grad():
            base_B, base_label = self.predict_hash_code(model,database_dataloader,self.best_epoch)
            query_B, query_label = self.predict_hash_code(model,query_dataloader,self.best_epoch)
        mAP_K,_,_ = mean_average_precision(base_B, query_B, base_label, query_label, self.opt.K)
        end.record()
        torch.cuda.synchronize()
        test_time = start.elapsed_time(end)
        #显示最高mAP@K
        logging.info('Test_mAP_K: {mAP_K:.4f}\t'.format(mAP_K = mAP_K))
        logging.info('Test_time: {test_time:.4f}\t'.format(test_time = test_time))
        
    def calculate(self, h):
        h_views = dict()
        h_views[0],h_views[1],h_views[2] = self.CM(h)
        return h_views

    def loss_criterion_reconsruction(self, h, view_data):
        loss = 0
        view_pred = self.calculate(h)
        for v_num in range(self.opt.view_num):
            # loss = loss + (torch.pow((view_pred[v_num]-view_data[v_num]), 2.0)).sum()
            # loss = loss + torch.norm((view_pred[v_num]-view_data[v_num]), 2.0)
            loss = loss + torch.pow(torch.norm((view_pred[v_num]-view_data[v_num])),2.0)
        return loss

    def loss_criterion_train(self, y, t_b, hash_center):
        Q_loss = torch.mean((torch.abs(t_b)-1.0)**2)
        center_criterion = torch.nn.BCEWithLogitsLoss()
        central_loss = center_criterion(0.5 * (t_b + 1), 0.5 * (hash_center + 1))
        t_1 = t_b.narrow(0,0,int(0.5*len(t_b)))
        t_2 = t_b.narrow(0,int(0.5*len(t_b)),int(0.5*len(t_b)))
        label_1 = y.narrow(0,0,int(0.5*len(y)))
        label_2 = y.narrow(0,int(0.5*len(y)),int(0.5*len(y)))
        similar_loss = self.pairwise_loss(t_1, t_2, label_1, label_2, sigmoid_param=1)
        train_loss = self.opt.param_central * central_loss + self.opt.param_similar * similar_loss + self.opt.param_Q * Q_loss 
        return train_loss, central_loss, similar_loss, Q_loss

    def pairwise_loss(self, outputs1, outputs2, label1, label2, sigmoid_param=1):
        similarity =  Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
        dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
        exp_loss = torch.log(1+torch.exp(-torch.abs(dot_product))) - similarity * dot_product + torch.max(dot_product, Variable(torch.FloatTensor([0.]).cuda(self.opt.device)))
        loss = torch.sum(exp_loss)/exp_loss.shape[0] 
        return loss

    def Hash_center_multilables(self, labels, Hash_center): # label.shape: [batch_size, num_class], Hash_center.shape: [num_class, hash_bits]
        is_start = True
        for label in labels:
            one_labels = (label == 1).nonzero()  # find the position of 1 in label
            one_labels = one_labels.squeeze(1)
            Center_mean = torch.mean(Hash_center[one_labels], dim=0)
            Center_mean[Center_mean<0] = -1
            Center_mean[Center_mean>0] = 1
            random_center[random_center==0] = -1   # the random binary vector become {-1, 1}
            Center_mean[Center_mean == 0] = random_center[Center_mean == 0]  # shape: [hash_bit]
            Center_mean = Center_mean.view(1, -1) # shape:[1,hash_bit]

            if is_start:  # the first time
                hash_center = Center_mean
                is_start = False
            else:
                hash_center = torch.cat((hash_center, Center_mean), 0)
        return hash_center
    
    def gen_fusion_h(self, model, flag, dataset):
        self.H_train,self.H_valid,self.H_test = self.H_init()  
        model.eval_start()
        gen_DataLoader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=False)
        for gen_epoch in trange(self.opt.gen_epochs, position=0, desc="epoch", leave=False, colour='yellow', ncols=80):
            gen_batchs = enumerate(tqdm(gen_DataLoader, position=1, desc="batch", leave=False, colour='red', ncols=80))
            for idx, (scene,track,audio,label) in gen_batchs:
                scene = torch.Tensor(scene).cuda(self.opt.device)
                track = torch.Tensor(track).cuda(self.opt.device)
                audio = torch.Tensor(audio).cuda(self.opt.device)
                start_idx,end_idx = idx * self.opt.batch_size, (idx+1) * self.opt.batch_size
                if flag == 'train':
                    h = self.H_train[start_idx: end_idx, ...]
                elif flag == 'valid':
                    h = self.H_valid[start_idx: end_idx, ...]
                elif flag == 'test':
                    h = self.H_test[start_idx: end_idx, ...]
                h.requires_grad = True
                view_data = dict()
                view_data[0], view_data[1], view_data[2] = scene, track, audio

                self.h_optimizer = SGD([h], lr= self.opt.gen_lr, 
                                   momentum=self.opt.optimizer.momentum,
                                   weight_decay=self.opt.optimizer.weight_decay,
                                   nesterov=self.opt.optimizer.nesterov
                                   )
                # self.h_optimizer = Adam(params=[h], lr = self.opt.gen_lr, betas=(0.9,0.999))
                for i in range(self.opt.gen_loop):
                    loss_recover = self.loss_criterion_reconsruction(h, view_data)
                    _,loss_clip = self.TextEncoder(h)
                    loss_update = loss_recover + self.opt.param_gen * loss_clip
                    self.h_optimizer.zero_grad()
                    loss_update.backward(retain_graph=True)
                    self.h_optimizer.step()
                    pass
                pass
            pass
        fusion_h = dict()
        fusion_h['train'],fusion_h['valid'],fusion_h['test'] = self.H_train,self.H_valid,self.H_test
        with open(self.opt.fusion_h_path + flag + '.npy', 'wb') as f:
            np.save(f,fusion_h[flag].cpu().detach().numpy())
        
        hash_dataset = HashDataset(self.opt,flag)
        return hash_dataset
            
    def predict_hash_code(self, model, data_loader, epoch):
        model.eval_start()
        is_start = True
        for idx, (h,label) in enumerate(data_loader):
            h = Variable(h).cuda(self.opt.device)
            _,b = self.Hash(h,epoch)
            
            if is_start:
                all_hashcode = b
                all_label = label
                is_start = False
            else:
                all_hashcode = torch.cat((all_hashcode, b), 0)
                all_label = torch.cat((all_label, label), 0)
        
        return all_hashcode.cpu().detach().numpy(), all_label.cpu().detach().numpy()
    
    def H_init(self):
        h_train = Variable(self.xavier_init(self.opt.num_train, self.opt.feature_dim))
        h_valid = Variable(self.xavier_init(self.opt.num_valid, self.opt.feature_dim))
        h_test = Variable(self.xavier_init(self.opt.num_test, self.opt.feature_dim))
        return h_train,h_valid,h_test

    def xavier_init(self, fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        a = np.random.uniform(low,high,(fan_in,fan_out))
        a = a.astype('float32')
        a = torch.from_numpy(a).cuda(self.opt.device)
        return a
    
    def state_dict(self):
        state_dict =[self.CM.state_dict(), self.TextEncoder.state_dict(), self.Hash.state_dict()]
        return state_dict 
    
    def load_state_dict(self,state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict[0].items():
            new_state_dict[k] = v
        self.CM.load_state_dict(new_state_dict, strict=True)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict[1].items():
            new_state_dict[k] = v
        self.TextEncoder.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[2].items():
            new_state_dict[k] = v
        self.Hash.load_state_dict(new_state_dict, strict=True)
                
        
                
            
            