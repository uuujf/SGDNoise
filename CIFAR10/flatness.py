import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from cifar10 import CIFAR10
from vgg import vgg11
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--repeat', type=int, default=100)
parser.add_argument('--noise-std', type=float, default=1e-3)
parser.add_argument('--ckptdir', type=str, default=None)
parser.add_argument('--start', type=int, default=3)
parser.add_argument('--end', type=int, default=16)
parser.add_argument('--datadir', type=str, default='/home/wjf/datasets/CIFAR10/numpy/')
parser.add_argument('--logdir', type=str, default='flat_logs/GD')

args = parser.parse_args()
logger = LogSaver(args.logdir)
logger.save(str(args), 'args')

# data
dataset = CIFAR10(args.datadir)
logger.save(str(dataset), 'dataset')
train_list = dataset.getTrainList(5000, True)
test_list = dataset.getTestList(5000, True)

# model
model = vgg11().cuda()
logger.save(str(model), 'classifier')
criterion = nn.CrossEntropyLoss().cuda()

# writer
writer = SummaryWriter(args.logdir)

# eval flatness
torch.backends.cudnn.benchmark = True
for i in range(args.start, args.end):
    ckpt_file = os.path.join(args.ckptdir, 'iter-'+str(i*1000)+'.pth.tar')
    if os.path.isfile(ckpt_file):
        logger.save("=> loading checkpoint '{}'".format(ckpt_file))
        checkpoint = torch.load(ckpt_file)
        state_dict = checkpoint['model']
    else:
        logger.save("=> no checkpoint found at '{}'".format(ckpt_file))
    dlossT, daccT, dlossV, daccV = deltaLossAcc(train_list, test_list, model, criterion, state_dict, args.noise_std, args.repeat)
    writer.add_scalar('delatAcc/train', daccT, i*1000)
    writer.add_scalar('deltaLoss/train', dlossT, i*1000)
    writer.add_scalar('delatAcc/test', daccV, i*1000)
    writer.add_scalar('deltaLoss/test', dlossV, i*1000)
    logger.save('Model:%d, Test [dacc: %.2f, dloss: %.6f], Train [dacc: %.2f, dloss: %.6f]' \
            % (i*1000, daccV, dlossV, daccT, dlossT))
writer.close()
