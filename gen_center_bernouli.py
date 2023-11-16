# random generation
import torch
import random
import numpy as np
import csv
from easydict import EasyDict
import yaml #yaml
from scipy.special import comb, perm  #calculate combination
from itertools import combinations

def generate_targets():
    # Read Parameters
    with open(r"./args.yaml") as f:
        opt = yaml.safe_load(f)
    opt = EasyDict(opt['PARAMETER'])

    hash_targets = []
    a = []  # for sampling the 0.5*hash_bit 
    b = []  # for calculate the combinations of 63 num_class
    num_class = opt.classes
    hash_bit = opt.binary_dim

    for i in range(0,hash_bit):
        a.append(i)

    for i in range(0,num_class):
        b.append(i)
    
    for j in range(10000):
        hash_targets = torch.zeros([num_class, hash_bit])
        for i in range(num_class):
            ones = torch.ones(hash_bit)
            sa = random.sample(a, round(hash_bit/2))
            ones[sa] = -1
            hash_targets[i]=ones
        com_num = int(comb(num_class, 2))
        c = np.zeros(com_num)
        for i in range(com_num):
            i_1 = list(combinations(b, 2))[i][0]
            i_2 = list(combinations(b, 2))[i][1]
            TF = torch.sum(hash_targets[i_1]!=hash_targets[i_2])
            c[i]=TF
        print(min(c))
        print(max(c))
        print(np.mean(c))

        if min(c)>=20 and np.mean(c)>=(hash_bit/2):  # guarantee the hash center are far away from each other in Hamming space, 20 can be set as 18 for fast convergence
            print(min(c))
            print("stop! we find suitable hash centers")
            name = str(hash_bit) + '_bit' + '_' + str(num_class) + '_class.pkl'
            path = opt.hash_center_path + '/' + name
            f = open(path,"wb")
            torch.save(hash_targets, f)
            break        

if __name__ == '__main__':
    generate_targets()