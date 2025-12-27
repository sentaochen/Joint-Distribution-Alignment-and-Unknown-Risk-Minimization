import torch
import numpy as np
import random
import os
import argparse

from loader.sampler import LabelSampler
from loader.data_loader import load_data_for_OSDA
from model.model import MODEL
from utils.optimizer import get_optimizer
from utils.train_OSDA import finetune_for_OSDA, train_for_OSDA
from utils.eval import predict
from utils import globalvar as gl
# import dataloader as dir_dataloader
parser = argparse.ArgumentParser(description='OSDA Classification')
parser.add_argument('--root_dir', type=str, default='./data/OfficeHome',
                    help='root dir of the dataset')  
parser.add_argument('--dataset', type=str, default='officehome',
                    help='the name of dataset')
parser.add_argument('--source', type=str, default='Art',
                    help='source domain')
parser.add_argument('--target', type=str, default='Clipart',
                    help='target domain')
parser.add_argument('--net', type=str, default='resnet',
                    help='which network to use')
parser.add_argument('--phase', type=str, default='train',
                    choices=['pretrain', 'train'],
                    help='the phase of training model')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_mult', type=float, nargs=4, default=[0.1, 0.1, 1, 1],
                    help='lr_mult (default: [0.1, 0.1, 1, 1])')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to pretrain (default: 10)')
parser.add_argument('--steps', type=int, default=50000,
                    help='maximum number of iterations to train (default: 50000)')
parser.add_argument('--lam_step', type=int, default=20000,
                    help='factor of lamda (default: 20000)')
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status(default: 100)')
parser.add_argument('--save_interval', type=int, default=500,
                    help='how many batches to wait before saving a model(default: 500)')
parser.add_argument('--update_interval', type=int, default=1000,
                    help='how many batches to wait before updating pseudo labels(default: 1000)')
parser.add_argument('--save_check', type=bool, default=True,
                    help='save checkpoint or not(default: True)')
parser.add_argument('--patience', type=int, default=10,
                    help='early stopping to wait for improvment before terminating. (default: 12 (6000 iterations))')
parser.add_argument('--early', type=bool, default=True,
                    help='early stopping or not(default: True)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='MOMENTUM of SGD (default: 0.9)')
parser.add_argument('--decay', type=int, default=0.0005,
                    help='DECAY of SGD (default: 0.0005)') 
parser.add_argument("--batch_size", default=32, type=int, help="batch size") 
parser.add_argument("--source_classes_num", default=25, type=int, help="source_class_num") 
parser.add_argument('--seed', type=int, default=0, help='seed')

# ===========================================================================================================================================================================
args = parser.parse_args()


DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
gl._init()
gl.set_value('DEVICE', DEVICE)

args.source_classes_num = {
    "officehome":25,
    "minidomainnet":40
}[args.dataset]

bottleneck_dim = 1024

select_class_num = args.batch_size // 2 
while select_class_num > args.source_classes_num:
    select_class_num //= 2



seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

record_dir = './record_OSDA/{}/{}'.format(str.upper(args.dataset), str.upper(args.net))
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
check_path = './save_model_OSDA/{}/{}'.format(str.upper(args.dataset), str.upper(args.net))

if not os.path.exists(check_path):
    os.makedirs(check_path)

if args.phase == 'pretrain':
    record_file = os.path.join(record_dir, 'pretrain_{}_{}.txt'.format(args.net, args.source))
else:
    record_file = os.path.join(record_dir, 'OSDA_{}_{}_to_{}.txt'.format(args.net, args.source, args.target))

gl.set_value('check_path', check_path)
gl.set_value('record_file', record_file)

if __name__ == '__main__':
    dataloaders = {}
    model = MODEL(args.net, args.source_classes_num+1, bottleneck_dim, args.source_classes_num).to(DEVICE)

    if args.phase == 'pretrain':
        dataloaders['src_pretrain'],dataloaders['tar_pretrain'] = load_data_for_OSDA(
            args, args.root_dir, args.dataset, args.source, args.target, args.phase, args.batch_size, args.net)
        dataloaders['src_test'], dataloaders['tar_test'] = load_data_for_OSDA(
            args, args.root_dir, args.dataset, args.source, args.target, 'test', args.batch_size, args.net)

        print(len(dataloaders['src_pretrain'].dataset))
        print(len(dataloaders['tar_pretrain'].dataset))
        print(len(dataloaders['src_test'].dataset))
        print(len(dataloaders['tar_test'].dataset))
        optimizer = get_optimizer(model, args.lr, args.lr_mult)

        finetune_for_OSDA(args, model, optimizer, dataloaders)

    elif args.phase == 'train':
        model_path = '{}/best(PT)_{}_{}_{}.pth'.format(check_path, args.net, args.source, args.target)
        print('model_path:{}'.format(model_path))
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
        label_sampler = LabelSampler(args.update_interval, args.source_classes_num, select_class_num)
        dataloaders['src_train_l'], dataloaders['tar_train_ul'] = load_data_for_OSDA(
            args, args.root_dir, args.dataset, args.source, args.target, args.phase, args.batch_size, args.net, label_sampler)
        dataloaders['src_test'], dataloaders['tar_test'] = load_data_for_OSDA(
            args, args.root_dir, args.dataset, args.source, args.target, 'test', args.batch_size, args.net)
        pseudo_labels = predict(model, dataloaders['tar_test'])
        dataloaders['tar_train_ul'].dataset.update_pseudo_labels(pseudo_labels)

        print(len(dataloaders['src_train_l'].dataset), len(dataloaders['tar_train_ul'].dataset))
        optimizer = get_optimizer(model, args.lr, args.lr_mult, args.momentum, args.decay)
        train_for_OSDA(args, model, optimizer, dataloaders)
    