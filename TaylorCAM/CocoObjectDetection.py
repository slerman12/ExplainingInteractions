from __future__ import print_function
from __future__ import division
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
import time
import os
import torch.utils.data as data
from PIL import Image, ImageDraw
import utils
from Data.coco import CocoDataset

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Interactional Multi-Object Detection COCO')
parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--num-workers', type=int, default=9, help='number of workers to use')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--use-trained', type=str, help='resume from model stored')
parser.add_argument('--data-dir', type=str, default="Data", help='Data directory')
parser.add_argument('--saved-model-dir', type=str, default="Saved_Models", help='Saved model directory')
parser.add_argument('--name', type=str, default="TEST_PC_Formal", help='Name of run')
parser.add_argument('--model-name', type=str, default="resnet", help='Name of model')
parser.add_argument('--use-train-data', action='store_true', default=True, help='use training dataset')
parser.add_argument('--avg-pool', action='store_true', default=False, help='do average pooling')

args = parser.parse_args()
print(not args.no_cuda, torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")
args.num_workers = args.num_workers if args.cuda else 0
args.batch_size = args.batch_size if args.cuda else 64
args.epochs = args.epochs if args.cuda else 1
args.use_train_data = args.use_train_data if args.cuda else False
args.use_trained = args.use_trained if args.cuda else "TEST_PC_Formal.pth "

# Models to choose from [resnet, resnetcoco, alexnet, vgg, squeezenet, densenet, inception]
model_name = args.model_name

# Number of classes in the dataset
num_classes = 2

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                all_true = labels["all_true"]
                labels = labels["labels"]
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    outputs = outputs.double()
                    preds = outputs.argmax(1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                for i in range(preds.shape[0]):
                    running_corrects += 1 if preds[i] == labels[i].data else 0

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class Compress(nn.Module):
    def __init__(self, out_channels=64):
        super(Compress, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(nn.Conv2d(2048, 1024, 2, stride=2, padding=2),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(1024),
                                  )
        self.layers = nn.Sequential(nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, self.out_channels),
                                    )

    def forward(self, x):
        x = self.conv(x)
        _x = x.permute(0, 2, 3, 1)
        _x = _x.reshape(-1, _x.shape[-1])
        _x = self.layers(_x).view(x.shape[0], x.shape[2], x.shape[3], self.out_channels)
        return _x.permute(0, 3, 1, 2)


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet" or model_name == "resnetcoco":
        """ Resnet18
        """
        if model_name == "resnetcoco":
            # Create the model
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=use_pretrained,
                                                             pretrained_backbone=use_pretrained)
            resnet = model.backbone.body

            # Check for all FrozenBN layers
            bn_to_replace = []
            for name, module in resnet.named_modules():
                if isinstance(module, torchvision.ops.misc.FrozenBatchNorm2d):
                    # print('adding ', name)
                    bn_to_replace.append(name)

            # Iterate all layers to change
            for layer_name in bn_to_replace:
                # Check if name is nested
                *parent, child = layer_name.split('.')
                # Nested
                if len(parent) > 0:
                    # Get parent modules
                    m = resnet.__getattr__(parent[0])
                    for p in parent[1:]:
                        m = m.__getattr__(p)
                    # Get the FrozenBN layer
                    orig_layer = m.__getattr__(child)
                else:
                    m = resnet.__getattr__(child)
                    orig_layer = copy.deepcopy(m)  # deepcopy, otherwise you'll get an infinite recusrsion
                # Add your layer here
                in_channels = orig_layer.weight.shape[0]
                bn = nn.BatchNorm2d(in_channels)
                with torch.no_grad():
                    bn.weight = nn.Parameter(orig_layer.weight)
                    bn.bias = nn.Parameter(orig_layer.bias)
                    bn.running_mean = orig_layer.running_mean
                    bn.running_var = orig_layer.running_var
                m.__setattr__(child, bn)

            # Fix the bn1 module to load the state_dict
            resnet.bn1 = resnet.bn1.bn1

            # Create reference model and load state_dict
            reference = models.resnet50()
            reference.load_state_dict(resnet.state_dict(), strict=False)
            model_ft = reference
        else:
            model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        compress = True
        if args.avg_pool:
            if compress:
                c = Compress()
                model_ft.avgpool = nn.Sequential(
                    c,
                    model_ft.avgpool
                )
                num_ftrs = c.out_channels
            else:
                num_ftrs = model_ft.fc.in_features
        else:
            if compress:
                c = Compress()
                model_ft.avgpool = nn.Sequential(
                    c,
                    Flatten()
                )
                num_ftrs = c.out_channels * 5 * 5
            else:
                model_ft.avgpool = Identity()
                num_ftrs = 2048 * 7 * 7
        # TODO test without pooling, compare to pooling, avg max interaction effect / performance / relational reasoning
        lay = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(256, num_classes),
            nn.Softmax()
        )
        model_ft.fc = lay
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=not args.use_trained)


# Data augmentation and normalization for training
# Just normalization for validation
def data_transforms(mode):
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


print("Initializing Datasets and Dataloaders...")


def get_loader(root, json, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              )
    return data_loader, coco


# Create training and validation dataloaders
dataloaders = {m: get_loader(root=args.data_dir + "/" + (m if args.use_train_data else "val") + "2017",
                             json=args.data_dir + "/annotations/instances_{}2017.json".format(
                                 m if args.use_train_data else "val"),
                             transform=data_transforms(m if args.use_train_data else "val"),
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers) for m in (['train', 'val'] if args.use_trained and args.use_train_data else ['val'] if args.use_trained else ['train', 'val'])}
datasets = {m: dataloaders[m][1] for m in dataloaders}
dataloaders = {m: dataloaders[m][0] for m in dataloaders}
val_size = int(0.5 * len(dataloaders["val"]))
test_size = len(dataloaders["val"]) - val_size
val_dataset, test_dataset = torch.utils.data.random_split(dataloaders["val"], [val_size, test_size])
dataloaders["val"] = val_dataset.dataset
dataloaders["test"] = test_dataset.dataset
datasets["test"] = datasets["val"]
if not args.use_train_data:
    dataloaders["train"] = test_dataset.dataset
    datasets["train"] = datasets["test"]

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

if not args.use_trained:
    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=args.epochs,
                                 is_inception=(model_name == "inception"))
    try:
        os.makedirs("./Results/ExplainCOCO/{}/".format(args.name))
    except:
        pass
    file_object = open('Results/ExplainCOCO/{}/COCO_hist_{}.txt'.format(args.name, args.name), 'w')
    file_object.write(str(hist))
    file_object.close()

    torch.save(model_ft.state_dict(), '{}/{}.pth'.format(args.saved_model_dir, args.name))
else:
    filename = os.path.join(args.saved_model_dir, args.use_trained)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename, map_location=None if args.cuda else torch.device('cpu'))
        model_ft.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))
    else:
        print("checkpoint not found")


def get_accuracy(imgs, lbls, return_raw=False):
    outputs = model_ft.to(device)(imgs.to(device)).to(device).double()
    preds = outputs.argmax(1).to("cpu")
    corrects = []
    for i in range(preds.shape[0]):
        corrects.append(1 if preds[i] == lbls[i].data else 0)
    if return_raw:
        return corrects
    else:
        return sum(corrects) / preds.shape[0]


def downstream(_x):
    if args.avg_pool:
        _x = nn.AdaptiveAvgPool2d((1, 1))(_x)
    _x = torch.flatten(_x.to("cpu"))
    _x = model_ft.eval().fc.to("cpu")(_x)
    print(_x)
    return _x.max()


def get_hessian(_x):
    fc = model_ft.fc[0].to(device)
    b = fc.bias.to(device)
    W = fc.weight.to(device)
    t_0 = torch.exp(-fc(_x.to(device)))
    t_1 = (torch.ones(b.shape[0]).to(device) + (2 * t_0))
    t_2 = (t_0 / t_1)
    return ((2 * (
                W.to(device).T[None, :, :] * ((t_0.to(device) * t_0.to(device)) / (t_1.to(device) * t_1.to(device)))[:,
                                             None, :]).to("cpu").matmul(W.to("cpu"))) -
            (W.to(device).T[None, :, :] * t_2.to(device)[:, None, :]).to("cpu").matmul(W.to("cpu")))


def get_interaction_effects(img):
    layers = nn.ModuleList(list(model_ft.children())[:-2] + [list(model_ft.children())[-2][0]]).eval()
    feature_extractor = nn.Sequential(*layers)
    features = feature_extractor(img.to(device))
    mb = features.shape[0]
    n_channels = features.shape[1]
    height, width = features.shape[2], features.shape[3]
    hessian = torch.zeros((mb, n_channels, height, width, n_channels, height, width))
    for b in range(mb):
        hessian[b] = torch.autograd.functional.hessian(downstream, features[b].to("cpu"), create_graph=False,
                                                       strict=True)
    hessian = hessian.view(mb, n_channels, height * width, n_channels, height * width).to(device)

    features = features.permute(0, 2, 3, 1).reshape(mb, height * width, n_channels).to(device)
    return utils.TaylorCAM(features, hessian, 0, 2, 4, 1, 3).reshape(mb, height, width, height, width)


def explain(imgs, lbls, zero_diags=True, name="coco"):
    model_ft.eval()
    interaction_effects = get_interaction_effects(imgs)

    if zero_diags:
        zero_diag = interaction_effects.reshape(interaction_effects.shape[:-2] +
                                                (interaction_effects.shape[3] * interaction_effects.shape[4],))
        zero_diag = interaction_effects.reshape(zero_diag.shape[0],
                                                zero_diag.shape[1] * zero_diag.shape[2],
                                                zero_diag.shape[3])
        zero_diag = torch.triu(zero_diag, diagonal=1)
        interaction_effects = zero_diag.reshape(interaction_effects.shape)

    height_scaler = int(imgs.shape[2] / interaction_effects.shape[1])
    width_scaler = int(imgs.shape[3] / interaction_effects.shape[2])

    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return list(reversed(out))


    all_true = lbls["all_true"]
    lbls = lbls["labels"]

    accuracy = get_accuracy(imgs, lbls, return_raw=True)
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
            border_size = 4
            line_size = border_size
            border_color = torch.tensor([230,224,176])[:, None, None]
            line_color = (250,128,114)
            pad = 16
            padded_image = torch.nn.ConstantPad2d(pad, 1)(imgs[i])
            heights = []
            widths = []
            for yx in range(0, len(top_interaction), 2):
                heights.append((top_interaction[yx] * height_scaler,
                                (top_interaction[yx] + 1) * height_scaler + 2 * pad))
                widths.append((top_interaction[yx + 1] * width_scaler,
                               (top_interaction[yx + 1] + 1) * width_scaler + 2 * pad))
            heatmap = copy.deepcopy(padded_image)
            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            pi = inv_normalize(padded_image)
            hm = inv_normalize(heatmap)
            for h, w in zip(heights, widths):
                hm[:, h[0]:h[0] + border_size, w[0]:w[1]] = border_color
                hm[:, h[1] - border_size:h[1], w[0]:w[1]] = border_color
                hm[:, h[0]:h[1], w[0]:w[0] + border_size] = border_color
                hm[:, h[0]:h[1], w[1] - border_size:w[1]] = border_color
            img = torchvision.transforms.functional.to_pil_image(pi)
            h_img = torchvision.transforms.functional.to_pil_image(hm)
            # res = Image.blend(img, h_img, 0.5)
            res = h_img
            draw = ImageDraw.Draw(res)
            draw.line([(sum(w) / 2., sum(h) / 2.) for w, h in zip(widths, heights)], fill=line_color, width=line_size)
            # res = res.crop((pad, pad, res.size[0] - pad, res.size[1] - pad))
            correct_incorrect = "correct" if accuracy[i] > 0 else "incorrect"
            try:
                os.makedirs("./Results/ExplainCOCO/{}/{}/{}/label_{}_{}/{}".format(args.name, name, correct_incorrect, lbls[i].data, all_true[i], i))
            except:
                pass
            try:
                os.makedirs("./Results/ExplainCOCO/{}/{}/{}_raw/label_{}_{}/{}".format(args.name, name, correct_incorrect, lbls[i].data, all_true[i], i))
            except:
                pass
            res.resize((300, 300)).save("./Results/ExplainCOCO/{}/{}/{}/label_{}_{}/{}/{}.png".format(
                args.name, name, correct_incorrect, lbls[i].data, all_true[i], i, k, "PNG"))
            img.resize((300, 300)).save("./Results/ExplainCOCO/{}/{}/{}_raw/label_{}_{}/{}/{}.png".format(
                args.name, name, correct_incorrect, lbls[i].data, all_true[i], i, k, "PNG"))

    accuracy = sum(accuracy) * 100. / len(accuracy)
    print('\n Interaction set: Accuracy: {:.0f}%\n'.format(accuracy))
    print("Average max interaction effect: ", np.mean(max_interaction_effects))
    print("Average min interaction effect: ", np.mean(min_interaction_effects))
    print("Average median interaction effect: ", np.mean(med_interaction_effects))
    print("Average interaction effect: ", np.mean(mean_interaction_effects))
    file_object = open("./Results/ExplainCOCO/{}/{}/stats.txt".format(args.name, name), 'w')
    file_object.write('\n Interaction set: Accuracy: {:.0f}%\n'.format(accuracy))
    file_object.write("Average max interaction effect: {}\n".format(np.mean(max_interaction_effects)))
    file_object.write("Average min interaction effect: {}\n".format(np.mean(min_interaction_effects)))
    file_object.write("Average median interaction effect: {}\n".format(np.mean(med_interaction_effects)))
    file_object.write("Average interaction effect: {}\n".format(np.mean(mean_interaction_effects)))
    file_object.close()


def test(model, phase):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels["labels"]
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs).double()
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)

        running_loss += loss.item() * inputs.size(0)
        for i in range(preds.shape[0]):
            running_corrects += 1 if preds[i] == labels[i].data else 0

    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects / len(dataloaders[phase].dataset)
    print('{} Test Loss: {:.4f} Test Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    try:
        os.makedirs("./Results/ExplainCOCO/{}/".format(args.name))
    except:
        pass
    file_object = open('Results/ExplainCOCO/{}/COCO_acc_{}.txt'.format(args.name, args.name), 'w')
    file_object.write('{} Test Loss: {:.4f} Test Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    file_object.close()


explain_imgs, explain_labels = next(iter(dataloaders["train" if args.use_train_data else "val"]))
print(explain_imgs.shape)
explain(explain_imgs, explain_labels, name=str(datasets["train" if args.use_train_data else "val"].class_names) + "_1")
explain_imgs, explain_labels = next(iter(dataloaders["train" if args.use_train_data else "val"]))
explain(explain_imgs, explain_labels, name=str(datasets["train" if args.use_train_data else "val"].class_names) + "_2")
explain_imgs, explain_labels = next(iter(dataloaders["train" if args.use_train_data else "val"]))
explain(explain_imgs, explain_labels, name=str(datasets["train" if args.use_train_data else "val"].class_names) + "_3")
test(model_ft, "test" if args.use_train_data else "val")
