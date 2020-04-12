import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from tensorboardX import SummaryWriter

from svhn import SVHN
from vgg import vgg11
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--maxiter', type=int, default=int(2e5+1))
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--list-size', type=int, default=5000)
parser.add_argument('--update-noise', type=int, default=10)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--datadir', type=str, default='/home/wjf/datasets/SVHN/train25000_test70000')
parser.add_argument('--logdir', type=str, default='logs/GLD_diag')

args = parser.parse_args()
logger = LogSaver(args.logdir)
logger.save(str(args), 'args')

# data
dataset = SVHN(args.datadir)
logger.save(str(dataset), 'dataset')
train_list = dataset.getTrainList(args.list_size, True)
test_list = dataset.getTestList(1000, True)

# model
start_iter = 0
model = vgg11().cuda()
logger.save(str(model), 'classifier')
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
logger.save(str(optimizer), 'optimizer')

if args.resume:
    checkpoint = torch.load(args.resume)
    start_iter = checkpoint['iter']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}'".format(args.resume))

# writer
writer = SummaryWriter(args.logdir)

# get number of total param
n_param = 0
for name, param in model.named_parameters():
    n_param += len(param.view(-1))
logger.save(str(n_param), 'n_param')

# optimization
torch.backends.cudnn.benchmark = True
for i in range(start_iter, args.maxiter):
    # eval noise
    if i % args.update_noise == 0:
        eval_list = dataset.getTrainBatchList(args.batch_size, 100, True)
        std_dict, diag2 = evalCovDiag(eval_list, model, criterion, optimizer)

    # train
    model.train()
    optimizer.zero_grad()
    loss_train, acc_train = 0, 0
    for x,y in train_list:
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        acc_train += accuracy(out, y).item()
        loss_train += loss.detach().item()

    for name, param in model.named_parameters():
        param.grad.data /= len(train_list)
        param.grad.data += torch.randn(param.size()).cuda() * std_dict[name]
    optimizer.step()
    acc_train /= len(train_list)
    loss_train /= len(train_list)

    # evaluate
    if i % 100 == 0 or i <= 100:
        writer.add_scalar('noise/norm2', diag2, i)
        writer.add_scalar('acc/train', acc_train, i)
        writer.add_scalar('loss/train', loss_train, i)

        acc, loss = 0, 0
        for x,y in test_list:
            out = model(x)
            acc += accuracy(out, y).item()
            loss += criterion(out, y).item()
        acc /= len(test_list)
        loss /= len(test_list)

        writer.add_scalar('acc/test', acc, i)
        writer.add_scalar('loss/test', loss, i)
        writer.add_scalar('acc/diff', acc_train-acc, i)

        logger.save('Iter:%d, Test [acc: %.2f, loss: %.4f], Train [acc: %.2f, loss: %.4f]' \
                % (i, acc, loss, acc_train, loss_train))

    if i % 1000 == 0:
        state = {'iter':i, 'model':model.state_dict(), 'optimizer':optimizer.state_dict()}
        torch.save(state, args.logdir+'/iter-'+str(i)+'.pth.tar')

writer.close()
