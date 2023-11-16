import random
import torch
import math
import numpy as np


#计算汉明距离
def calc_hammingDist(B1, B2):
  q = B2.shape[1]
  if len(B1.shape) < 2: 
      B1 = B1.unsqueeze(0)
  distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
  return distH

#计算mAP@K
def calc_map_k(qB, rB, query_L, retrieval_L, device, k=None):
  # qB: {-1,+1}^{mxq} 用于查询的哈希码
  # rB: {-1,+1}^{nxq} 用于检索的哈希码
  # query_L: {0,1}^{mxl} 用于查询的哈希码的真实标签
  # retrieval_L: {0,1}^{nxl} 用于检索的哈希码的真实标签
  num_query = query_L.shape[0] #查询数量
  map = 0
  if k is None:
      k = retrieval_L.shape[0] #检索数量
  for iter in range(num_query):
      q_L = query_L[iter]
      if len(q_L.shape) < 2:
          q_L = q_L.unsqueeze(0) #增加一个维度
      gnd = (q_L.matmul(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
      tsum = torch.sum(gnd)
      if tsum == 0:
          continue #跳出本次循环
      hamm = calc_hammingDist(qB[iter, :], rB) #第iter个查询的哈希码与所有检索的哈希码计算汉明距离
      _, ind = torch.sort(hamm) #排序好的数据和对应打索引
      ind.squeeze_() #增加一个维度
      gnd = gnd[ind] #gnd按汉明距离顺序重新排序
      total = min(k, int(tsum)) #为防止有相同标签的不足K个
      count = torch.arange(1, total + 1).type(torch.float32) #建立1到total+1的张量
      tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0 #gnd中非零的索引+1(因为从0开始)
      if tindex.is_cuda:
          count = count.cuda(device)
      map = map + torch.mean(count / tindex)
  map = map / num_query
  return map

def mean_average_precision(database_hash, test_hash, database_labels, test_labels, K = None):  # R = 1000

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)  
    ids = np.argsort(-sim, axis=0)  

    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        label = test_labels[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:K], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, K + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx

#计算相似性，构建相似性矩阵
def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = (label1.float().matmul(label2.float().transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    return Sim

#将数据分为query和retrieval
def split_data(f_scene, f_track, f_audio, features, labels, query_size, device):
    X_scene = {}
    X_track = {}
    X_audio = {}

    X_scene['query'] = f_scene[0: query_size].cuda(device)
    X_scene['retrieval'] = f_scene[query_size:].cuda(device)
    X_track['query'] = f_track[0: query_size].cuda(device)
    X_track['retrieval'] = f_track[query_size:].cuda(device)
    X_audio['query'] = f_audio[0: query_size].cuda(device)
    X_audio['retrieval'] = f_audio[query_size:].cuda(device)

    X = {}
    X['query'] = features[0: query_size].cuda(device)
    X['retrieval'] = features[query_size:].cuda(device)

    L = {}
    L['query'] = labels[0: query_size].cuda(device)
    L['retrieval'] = labels[query_size:].cuda(device)

    return X_scene, X_track, X_audio, X, L

def cal_ap(y_pred,y_true):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    ap = torch.zeros(y_pred.size(0))   ## 总样本个数
    # compute average precision for each class
    for k in range(y_pred.size(0)):
        # sort scores
        scores = y_pred[k,:].reshape([1,-1]) #reshape([1,-1])转为1行
        targets = y_true[k,:].reshape([1,-1]) #reshape([1,-1])转为1行
        # compute average precision
        # ap[k] = average_precision(scores, targets, difficult_examples)
        ap[k] = avg_p(scores, targets)
    return ap

def avg_p(output, target):

    sorted, indices = torch.sort(output, dim=1, descending=True) #按列递减排序，sorted排序后的值，indices原本位置的编号
    tp = 0
    s = 0
    for i in range(target.size(1)):
        idx = indices[0,i]
        if target[0,idx] == 1:
            tp = tp + 1
            pre = tp / (i+1)
            s = s + pre
    if tp == 0:
        AP = 0
    else:
        AP = s/tp
    return AP

# def attention(query,key,value,device_idx):
#     sqrt_d_k = torch.full((query.shape[0],key.shape[0]),1/math.sqrt(key.shape[1]),device=device_idx)
#     result =  torch.matmul(torch.matmul(torch.matmul(query,key.t()),sqrt_d_k),value)
#     return result

def attention(self, query,key,value):
    sqrt_d_k = torch.full((query.shape[0],key.shape[0]),1/math.sqrt(key.shape[1]),device = self.opt.device)
    # theta = 1.0/math.pow(key.shape[1],3)
    result = torch.matmul(torch.matmul(torch.matmul(query,key.t()),sqrt_d_k),value)#*theta
    return result

def Label2Text(label):
    dataset_class = ["dog","cat","mouse","rabbit","bird","scenery","customs","dressing","baby","man",
                     "women","dessert","seafood","streetside","drinks","hotpot","claw","handsign",
                     "streetDance","international","poleDance","ballet","squareDance","folkDance",
                     "drawing","handwriting","latte","sand","slime","origami","knitting","hair",
                     "pottery","phone","drums","guitar","piano","guzheng","violin","cello","clarinet",
                     "singing","games","entertainment","animation","word","yoga","fitness","skateboard",
                     "basketball","parkour","diving","billiards","football","badminton","tennis","brow",
                     "eyeliner","skincare","lipgloss","removal","nail","cosmetic"]
    
    # 取 label 对应词
    label_text = []
    for i in range(label.shape[0]):
        label_text.append([])
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                label_text[i].append(dataset_class[j])
    
    # label 提示词组合
    phrase_text = []
    for i in range(len(label_text)):
        if len(label_text[i]) == 1:
            phrase_text.append(label_text[i][0])
        else:
            phrase = ''
            for j in range(len(label_text[i])-1):
                if j == len(label_text[i])-2:
                    phrase = phrase + label_text[i][j] + ' and '
                else:
                    phrase = phrase + label_text[i][j] + ', '
            phrase += label_text[i][-1]
            phrase_text.append(phrase)
    
    # prompt_sentence = [f"A video of {{}}.", f"This is a video of {{}}.", f"A bright video of {{}}.", 
    #             f"A good video of {{}}.", f"A cartoon of {{}}.", f"A bad video of {{}}.", f"{{}}",
    #             f"A video of a hard to see {{}}.", f"A video of the large {{}}.", f"Video classification of {{}}."]
    
    prompt_sentence = [f"A video of {{}}."]
    
    prompt_text = []
    for i in range(len(phrase_text)):
        # random_idx = random.randrange(len(prompt_sentence))
        # random_sentence = prompt_sentence[random_idx]
        random_sentence = prompt_sentence[0]
        prompt_text.append(random_sentence.format(phrase_text[i]))
    
    return prompt_text





