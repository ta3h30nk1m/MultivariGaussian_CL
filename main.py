import sys, os
import numpy as np
import torch
import datetime
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from dataset import *
from model import *
from utils import *
from train import *
from torchvision import datasets,transforms
import argparse

inputsize = (3, 32, 32)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--experiment', default='cifar-10', type=str, help='dataset name')
parser.add_argument('--nepochs', default=10, type=int, help='number of epochs')
parser.add_argument('--lr', default=0.02, type=int, help='learning rate')
parser.add_argument('--nfolds', default=1, type=int, help='dataset name')
parser.add_argument('--ntasks', default=5, type=int, help='number of tasks for continual learning')
parser.add_argument('--checkpoint_dir', default='', type=str, help='directory to save model checkpoint')

args = parser.parse_args()
print(args)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('='*100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)
########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')
    sys.exit()

########################################################################################################################

# Load
print('Load data...')
if args.experiment =='cifar-10':
    train = datasets.CIFAR10('Data', train=True,  download=True)
    test  = datasets.CIFAR10('Data', train=False, download=True)
    nclasses = 10
elif args.experiment =='cifar-100':
    train = datasets.CIFAR100('Data', train=True,  download=True)
    test  = datasets.CIFAR100('Data', train=False, download=True)
    nclasses=100
else:
    NameError('either cifar-10 or cifar-100')
#train = datasets.CIFAR100('Data/cifar100', train=True, download=True)
#test = datasets.CIFAR100('Data/cifar100', train=False, download=True)

final_acc = 0

# Cross_validation
for fold in range(args.nfolds):
    print('current fold:', fold+1)
    print('-'*100)
    
    data = get_split(train, test, fold, args.nfolds, args.ntasks, nclasses, args.experiment)
    #data = get_split_cifar10(train, test)
    train_loader, test_loader  = [CLDataLoader(elem, train=t) for elem, t in zip(data, [True, False])]

    print('Init model')
    net = Net(inputsize).cuda()
    appr = Appr(net, nepochs=args.nepochs, lr=args.lr)
    Buffer = []
    #Buffer = np.zeros((10, 2, 4096))
    #Means = np.zeros((10, 4096))
    #Covs = np.zeros((10, 4096, 4096))
    total_num = np.zeros(10)
    Features = np.empty((10, 5000, 4096))
    print('-'*100)

    # Loop tasks
    acc = np.zeros((len(train_loader), len(train_loader)), dtype=np.float32)
    lss = np.zeros((len(train_loader), len(train_loader)), dtype=np.float32)

    for t, tr_loader in enumerate(train_loader):
        print('*'*100)
        print('Task {:2d}'.format(t+1))
        print('*'*100)

        # Train
        Buffer, Features, total_num = appr.train(t, tr_loader, test_loader, Buffer, Features, total_num)
        #Means, Covs, Features, total_num = appr.train(t, tr_loader, test_loader, Means, Covs, Features, total_num)
        print('-'*100)
        # checkpoint save
        path = args.checkpoint_dir + '/tasknum' + str(t) + '_' + args.experiment
        torch.save({'model':appr.model.state_dict()}, path)

        # Test
        for u, te_loader in enumerate(test_loader):
            if u > t: break
            test_loss, test_acc = appr.eval(t, te_loader, False)
            print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.2f}% <<<'.format(u+1, test_loss, 100*test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss


    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end='')
        for j in range(acc.shape[1]):
            print('{:5.2f}% '.format(100*acc[i, j]),end='')
        print()
    
    final_acc += (acc[4,0]+acc[4,1]+acc[4,2]+acc[4,3]+acc[4,4])/5

print('*'*100)
print('Done!')
print('Final average accuracy:', final_acc * 100 / 6, "%")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)
