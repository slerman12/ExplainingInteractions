"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import copy
import os
import pickle
import random
import numpy as np
import torchvision
from PIL import Image
from PIL import ImageDraw
from relation_network import RN, CNN_MLP, Pool
import torch
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLEVR Explain')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP', 'Pool'], default='RN', help='model type')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for testing')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--pre-relational', action='store_true', default=False, help='Adds pre-relational layers')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--data-dir', type=str, default="Data/", help='Data directory')
parser.add_argument('--saved-model-dir', type=str, default="Saved_Models", help='Saved model directory')
parser.add_argument('--resume', type=str, default="model_00.pth", help='resume from model stored')
parser.add_argument('--name', type=str, default="visuals_and_stats", help='Name')
parser.add_argument('--use-norel', action='store_true', default=False, help='use non-relational questions only')
parser.add_argument('--gelu', action='store_true', default=False, help='use gelu as act func')
parser.add_argument('--num_outputs', type=int, default=10, help='number of outputs')
parser.add_argument('--sigmoid', action='store_true', default=False, help='use sigmoid as act func')
parser.add_argument('--tanh', action='store_true', default=False, help='use tanh as act func')

# random.seed(10)
args = parser.parse_args()
print(not args.no_cuda, torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'CNN_MLP':
    model = CNN_MLP(args, device)
elif args.model == 'RN':
    model = RN(args, device)
elif args.model == 'Pool':
    model = Pool(args, device)

model = model.to(device)
input_img = torch.FloatTensor(args.batch_size, 3, 75, 75).to(device)
input_img_unscaled = torch.FloatTensor(args.batch_size, 3, 300, 300).to(device)
input_qst = torch.FloatTensor(args.batch_size, 11).to(device)
label = torch.LongTensor(args.batch_size).to(device)

input_img = Variable(input_img)
input_img_unscaled = Variable(input_img_unscaled)
input_qst = Variable(input_qst)
label = Variable(label)


def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][args.batch_size * i:args.batch_size * (i + 1)]))
    qst = torch.from_numpy(np.asarray(data[1][args.batch_size * i:args.batch_size * (i + 1)]))
    ans = torch.from_numpy(np.asarray(data[2][args.batch_size * i:args.batch_size * (i + 1)]))

    input_img_unscaled.data.resize_(img.size()).copy_(img)
    input_img.copy_(torch.nn.functional.interpolate(input_img_unscaled, 75))
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img, qst, ans)


def explain(data, zero_diags=True, name="rel"):
    model.eval()

    data = cvt_data_axis(data)

    tensor_data(data, 1)
    interaction_effects = model.interaction_effects(input_img, input_qst)
    # zero_diags=False
    if zero_diags:
        zero_diag = interaction_effects.reshape(interaction_effects.shape[:-2] +
                                                (interaction_effects.shape[3] * interaction_effects.shape[4],))
        zero_diag = interaction_effects.reshape(zero_diag.shape[0],
                                                zero_diag.shape[1] * zero_diag.shape[2],
                                                zero_diag.shape[3])
        zero_diag = torch.triu(zero_diag, diagonal=1)
        interaction_effects = zero_diag.reshape(interaction_effects.shape)

    height_scaler = int(input_img_unscaled.shape[2] / interaction_effects.shape[1])
    width_scaler = int(input_img_unscaled.shape[3] / interaction_effects.shape[2])

    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return list(reversed(out))

    qst_count = torch.zeros(3, dtype=torch.int).to("cpu")
    accuracy = model.test_(input_img, input_qst, label, return_raw=True)
    max_interaction_effects = []
    min_interaction_effects = []
    med_interaction_effects = []
    mean_interaction_effects = []
    for i in range(interaction_effects.shape[0]):
        topk = 100
        for k in range(topk):
            max_interaction_effects.append(torch.max(interaction_effects[i]).item())
            min_interaction_effects.append(torch.min(interaction_effects[i]).item())
            med_interaction_effects.append(torch.median(interaction_effects[i]).item())
            mean_interaction_effects.append(torch.mean(interaction_effects[i]).item())
            top_interaction = torch.argmax(interaction_effects[i])
            top_interaction = unravel_index(top_interaction, interaction_effects[i].shape)
            interaction_effects[i][top_interaction] = 0
            border_size = 2
            border_color = 0
            pad = 18
            padded_image = torch.nn.ConstantPad2d(pad, 1)(input_img_unscaled[i])
            heights = []
            widths = []
            for yx in range(0, len(top_interaction), 2):
                heights.append((top_interaction[yx] * height_scaler,
                                (top_interaction[yx] + 1) * height_scaler + 2 * pad))
                widths.append((top_interaction[yx + 1] * width_scaler,
                               (top_interaction[yx + 1] + 1) * width_scaler + 2 * pad))
            heatmap = copy.deepcopy(padded_image)
            for h, w in zip(heights, widths):
                heatmap[:, h[0]:h[0] + border_size, w[0]:w[1]] = border_color
                heatmap[:, h[1] - border_size:h[1], w[0]:w[1]] = border_color
                heatmap[:, h[0]:h[1], w[0]:w[0] + border_size] = border_color
                heatmap[:, h[0]:h[1], w[1] - border_size:w[1]] = border_color
            img = torchvision.transforms.functional.to_pil_image(padded_image.to("cpu"))
            h_img = torchvision.transforms.functional.to_pil_image(heatmap.to("cpu"))
            res = Image.blend(img, h_img, 0.5)
            draw = ImageDraw.Draw(res)
            draw.line([(sum(w) / 2., sum(h) / 2.) for w, h in zip(widths, heights)], fill=(255, 0, 0), width=2)
            q = torch.argmax(input_qst[i, -3:])
            qst_count += input_qst[i, -3:].to("cpu").int()
            c_map = {
                0: "blue",  # bl
                1: "yellow",  # ye
                2: "orange",  # or
                3: "pink",  # pi
                4: "green",  # gr
                5: "purple"   # pu
            }[int(torch.argmax(input_qst[i, :6]))]
            correct_incorrect = "correct" if accuracy[i] > 0 else "incorrect"
            try:
                os.makedirs("./Results/ExplainRN/{}/{}/question_{}/{}".format(name, correct_incorrect, q, i))
            except:
                pass
            try:
                os.makedirs("./Results/ExplainRN/{}/{}_raw/question_{}/{}".format(name, correct_incorrect, q, i))
            except:
                pass
            res.resize((300, 300)).save("./Results/ExplainRN/{}/{}/question_{}/{}/{}_qst_{}_lbl_{}.png".format(
                name, correct_incorrect, q, i, qst_count[q], c_map, label[i].item()), "PNG")
            img.resize((300, 300)).save("./Results/ExplainRN/{}/{}_raw/question_{}/{}/{}_qst_{}_lbl_{}.png".format(
                name, correct_incorrect, q, i, qst_count[q], c_map, label[i].item()), "PNG")

    accuracy = accuracy.sum() * 100. / len(accuracy.tolist())
    print('\n Interaction set: Accuracy: {:.0f}%\n'.format(accuracy))
    print("Average max interaction effect: ", np.mean(max_interaction_effects))
    print("Average min interaction effect: ", np.mean(min_interaction_effects))
    print("Average median interaction effect: ", np.mean(med_interaction_effects))
    print("Average interaction effect: ", np.mean(mean_interaction_effects))
    file_object = open("./Results/ExplainRN/{}/stats.txt".format(name), 'w')
    file_object.write('\n Interaction set: Accuracy: {:.0f}%\n'.format(accuracy))
    file_object.write("Average max interaction effect: {}\n".format(np.mean(max_interaction_effects)))
    file_object.write("Average min interaction effect: {}\n".format(np.mean(min_interaction_effects)))
    file_object.write("Average median interaction effect: {}\n".format(np.mean(med_interaction_effects)))
    file_object.write("Average interaction effect: {}\n".format(np.mean(mean_interaction_effects)))
    file_object.close()


def test(rel, norel):
    model.to(device).eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []
    for batch_idx in range(len(rel[0]) // args.batch_size):
        tensor_data(rel, batch_idx)
        accuracy_rels.append(model.test_(input_img.to(device), input_qst.to(device), label.to(device)))

        tensor_data(norel, batch_idx)
        accuracy_norels.append(model.to(device).test_(input_img.to(device), input_qst.to(device), label.to(device)))

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)

    print('\n Test set: Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%\n'.format(
        accuracy_rel, accuracy_norel))
    file_object = open("./Results/ExplainRN/{}/stats_all.txt".format(args.name), 'w')
    file_object.write('\n Test set: Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%\n'.format(
        accuracy_rel, accuracy_norel))
    file_object.close()



def load_data():
    print('loading data...')
    dirs = args.data_dir
    filename = os.path.join(dirs, 'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
        train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(relations[0], relations[1]):
            rel_train.append((img, qst, ans))
        for qst, ans in zip(norelations[0], norelations[1]):
            norel_train.append((img, qst, ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(relations[0], relations[1]):
            rel_test.append((img, qst, ans))
        for qst, ans in zip(norelations[0], norelations[1]):
            norel_test.append((img, qst, ans))

    random.shuffle(rel_train)
    random.shuffle(rel_test)
    random.shuffle(norel_train)
    random.shuffle(norel_test)

    return (rel_train, rel_test, norel_train, norel_test)


rel_train, rel_test, norel_train, norel_test = load_data()

filename = os.path.join(args.saved_model_dir, args.resume)
print(filename, os.path.isfile(filename))
if os.path.isfile(filename):
    print('==> loading checkpoint {}'.format(filename))
    checkpoint = torch.load(filename, map_location=None if args.cuda else torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print('==> loaded checkpoint {}'.format(filename))

explain(norel_test if args.use_norel else rel_test, name=args.name)
# test(rel_test, norel_test)
