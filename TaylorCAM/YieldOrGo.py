"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import copy
import os
# import cPickle as pickle
import pickle
import random
import numpy as np
import torchvision

from relation_network import RN, CNN_MLP, Pool
import torch
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP', 'Pool'], default='RN',
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--pre-relational', action='store_true', default=False,
                    help='Adds pre-relational layers')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--save-all', action='store_true', default=False,
                    help='save each epoch')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str, help='resume from model stored')
parser.add_argument('--data-dir', type=str, default="Data/", help='Data directory')
parser.add_argument('--saved-model-dir', type=str, default="Saved_Models/", help='Saved model directory')
parser.add_argument('--name', type=str, default="TEST_RN_yield_or_go", help='Saved model directory')
parser.add_argument('--gelu', action='store_true', default=False, help='use gelu as act func')
parser.add_argument('--num_outputs', type=int, default=2, help='number of outputs')

args = parser.parse_args()
print(not args.no_cuda, torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")

# args.resume = "epoch_3_way_RN_02.pth"

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'CNN_MLP':
    model = CNN_MLP(args, device).to(device)
elif args.model == 'RN':
    model = RN(args, device).to(device)
elif args.model == 'Pool':
    model = Pool(args, device).to(device)

bs = args.batch_size
input_img = torch.FloatTensor(bs, 3, 75, 75).to(device)
input_img_unscaled = torch.FloatTensor(bs, 3, 300, 300).to(device)
label = torch.LongTensor(bs).to(device)

input_img = Variable(input_img)
input_img_unscaled = Variable(input_img_unscaled)
label = Variable(label)

best_acc = 0
best_epoch = 0
best_model_wts = copy.deepcopy(model.state_dict())
hist = []


def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs * i:bs * (i + 1)]))
    ans = torch.from_numpy(np.asarray(data[1][bs * i:bs * (i + 1)]))

    input_img_unscaled.data.resize_(img.size()).copy_(img)
    input_img.copy_(torch.nn.functional.interpolate(input_img_unscaled, 75))
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    ans = [e[1] for e in data]
    return (img, ans)


def train(epoch, rel):
    model.to(device).train()

    random.shuffle(rel)

    rel = cvt_data_axis(rel)

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        torchvision.transforms.functional.to_pil_image(input_img_unscaled[0]).show()
        accuracy_rel = model.train_(input_img.to(device), None, label.to(device))

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] Relations accuracy: {:.0f}%'.format(
                    epoch, batch_idx * bs * 2, len(rel[0]) * 2, \
                    100. * batch_idx * bs / len(rel[0]), accuracy_rel))


def test(epoch, rel, best_acc, best_epoch, best_model_wts):
    model.to(device).eval()

    rel = cvt_data_axis(rel)

    accuracy_rels = []
    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        accuracy_rels.append(model.test_(input_img.to(device), None, label.to(device)))

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    if epoch > -1:
        hist.append((epoch, accuracy_rel))
    print('\n Test set: Relation accuracy: {:.0f}%\n'.format(
        accuracy_rel))

    if accuracy_rel > best_acc:
        best_acc = accuracy_rel
        best_epoch = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
    return best_acc, best_epoch, best_model_wts


def load_data():
    print('loading data...')
    dirs = args.data_dir
    filename = os.path.join(dirs, 'yield-or-go.pickle')
    with open(filename, 'rb') as f:
        train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    print('processing data...')

    for img, ans in train_datasets:
        img = np.swapaxes(img, 0, 2)
        rel_train.append((img, ans))

    for img, ans in test_datasets:
        img = np.swapaxes(img, 0, 2)
        rel_test.append((img, ans))

    return (rel_train, rel_test)


rel_train, rel_test = load_data()

try:
    os.makedirs(args.saved_model_dir)
except:
    print('directory {} already exists'.format(args.saved_model_dir))

if args.resume:
    filename = os.path.join(args.saved_model_dir, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename, map_location=None if args.cuda else torch.device('cpu'))
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

for epoch in range(1, args.epochs + 1):
    train(epoch, rel_train)
    best_acc, best_epoch, best_model_wts = test(epoch, rel_test, best_acc, best_epoch, best_model_wts)
    if args.save_all:
        model.save_model(epoch, args.name)
model.load_state_dict(best_model_wts)
model.save_model(0, args.name)
print("best")
test(-1, rel_test, best_acc, best_epoch, best_model_wts)
try:
    os.makedirs("./Results/ExplainRN/{}/".format(args.name))
except:
    pass
file_object = open('Results/ExplainRN/{}/RN_hist_{}.txt'.format(args.name, args.name), 'w')
file_object.write(str(hist))
file_object.close()
