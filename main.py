from easydict import EasyDict
from model import *
from utils import *
import yaml
from tensorboardX import SummaryWriter
import logging
from datasets import *
from GetText import get_prompt_text

def main():
    # 读取公共参数
    with open(r"./args.yaml") as f:
        opt = yaml.safe_load(f)
    opt = EasyDict(opt['PARAMETER'])
    writer = SummaryWriter(log_dir=opt.tensorboard, flush_secs=5)
    # logging的基本配置
    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                        filename=opt.log,
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                                       # a是追加模式，默认如果不写的话，就是追加模式
                        format='%(asctime)s - %(levelname)s: %(message)s' # 日志格式
                        )
    # DataLoader
    train_dataset = MyDataset(opt,'train')
    valid_dataset = MyDataset(opt,'valid')
    test_dataset = MyDataset(opt,'test')
    #Bernouli
    true_hash = opt.hash_center_path + str(opt.binary_dim) + '_bit' + '_' + str(opt.classes) + '_class.pkl'
    print(' * * * loading data finish * * * ')
    # 定义模型
    model = DMHSE(opt)

    if opt.train == 1:
        prompt_dataset = get_prompt_text(opt)
        model.train(opt, prompt_dataset, train_dataset, true_hash, valid_dataset, model, writer)
    if opt.test == 1:
        model.test(opt, valid_dataset, test_dataset, model)
        
if __name__ == '__main__':
    main()