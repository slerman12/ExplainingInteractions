import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import utils


class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x


class FCOutputModel(nn.Module):
    def __init__(self, activation, args):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, args.num_outputs)
        self.activation =activation

    def forward(self, x):
        x = self.fc2(x)
        x = self.activation(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x)


class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name
        self.args = args

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def test_(self, input_img, input_qst, label, return_raw=False):
        output = self(input_img, input_qst)
        if len(output.shape) == 1 and len(input_img.shape) == 4:
            output = output.unsqueeze(0)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data)
        if return_raw:
            return correct
        accuracy = correct.sum() * 100. / len(label)
        return accuracy.item()

    def save_model(self, epoch, name):
        torch.save(self.state_dict(), '{}/{}_{:02d}.pth'.format(self.args.saved_model_dir, name, epoch))


class RN(BasicModel):
    def __init__(self, args, device):
        super(RN, self).__init__(args, 'RN')

        self.conv = ConvInputModel()

        qst_size = 11
        if args.num_outputs == 2:
            qst_size = 0

        if args.pre_relational:
            self.g_pre_fc1 = nn.Linear(24 + 2 + qst_size, 128)
            self.g_pre_fc3 = nn.Linear(128, 32)
            self.g_in_size = (32 + 2) * 2 + qst_size
        else:
            self.g_in_size = (24 + 2) * 2 + qst_size

        # TODO adapt to input size
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear(self.g_in_size, 256)
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2).to(device)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2).to(device)
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i / 5 - 2) / 2., (i % 5 - 2) / 2.]

        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2).to(device)
        self.coord_tensor = Variable(self.coord_tensor)
        # TODO adapt to input size
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        if args.gelu:
            self.activation = nn.GELU()
        elif args.sigmoid:
            self.activation = nn.Sigmoid()
        elif args.tanh:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        self.fcout = FCOutputModel(self.activation, self.args)
        self.device = device

    def forward(self, img, qst):
        x = self.conv(img)  ## x = (64 x 24 x 5 x 5)

        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        num_components = x.size()[2] * x.size()[3]
        if qst is not None:
            qst_size = qst.size()[1]
        else:
            qst_size = 0
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb, n_channels, num_components).permute(0, 2, 1)

        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)

        # add question everywhere
        if qst is not None:
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, num_components, 1)

        if self.args.pre_relational:
            if qst is not None:
                feature_vecs = torch.cat([x_flat, qst], dim=-1)
            else:
                feature_vecs = x_flat
            feature_vecs = feature_vecs.view(-1, n_channels + 2 + qst_size)
            g_pre = self.activation(self.g_pre_fc1(feature_vecs))
            # g_pre = nn.ReLU()(self.g_pre_fc2(g_pre))
            g_pre = self.activation(self.g_pre_fc3(g_pre).view(mb, num_components, 32))
            x_flat = g_pre

            # add coordinates
            x_flat = torch.cat([x_flat, self.coord_tensor], 2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+11)
        x_i = x_i.repeat(1, num_components, 1, 1)  # (64x25x25x26)
        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+11)
        if qst is not None:
            qst = torch.unsqueeze(qst, 2)
            x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, num_components, 1)  # (64x25x25x26+11)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)  # (64x25x25x2*26+11)

        # reshape for passing through network
        x_ = x_full.view(mb * num_components ** 2, (32 + 2) * 2 + qst_size if self.args.pre_relational else 2 * (n_channels + 2) + qst_size)

        x_ = self.g_fc1(x_)
        x_ = self.activation(x_)
        x_ = self.g_fc2(x_)
        x_ = self.activation(x_)
        x_ = self.g_fc3(x_)
        x_ = self.activation(x_)
        x_ = self.g_fc4(x_)
        x_ = self.activation(x_)

        # reshape again and sum
        x_g = x_.view(mb, num_components ** 2, 256)

        x_g = x_g.sum(1).squeeze()

        """f"""
        x_f = self.f_fc1(x_g)
        x_f = self.activation(x_f)

        return self.fcout(x_f)

    def interaction_effects(self, img, qstn):
        x = self.conv(img)  ## x = (64 x 24 x 5 x 5)

        mb = x.size()[0]
        n_channels = x.size()[1]
        height = x.size()[2]
        width = x.size()[3]
        num_components = height * width
        if qstn is not None:
            qst_size = qstn.size()[1]
        else:
            qst_size = 0

        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb, n_channels, num_components).permute(0, 2, 1)

        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor[:mb]], 2)

        # add question everywhere
        if qstn is not None:
            qstn = torch.unsqueeze(qstn, 1)
            qstn = qstn.repeat(1, num_components, 1)

        if self.args.pre_relational:
            if qstn is not None:
                feature_vecs = torch.cat([x_flat, qstn], dim=-1)
            else:
                feature_vecs = x_flat
            feature_vecs = feature_vecs.view(-1, n_channels + 2 + qst_size)
            g_pre = self.activation(self.g_pre_fc1(feature_vecs))
            # g_pre = nn.ReLU()(self.g_pre_fc2(g_pre))
            g_pre = self.activation(self.g_pre_fc3(g_pre).view(mb, num_components, 32))
            x_flat = g_pre

            # add coordinates
            x_flat = torch.cat([x_flat, self.coord_tensor], 2)

        def downstream(x_, qst):
            self.to("cpu")
            # cast all pairs against each other
            x_i = torch.unsqueeze(x_, 0)  # (64x1x25x26+11)
            x_i = x_i.repeat(num_components, 1, 1)  # (64x25x25x26)
            x_j = torch.unsqueeze(x_, 1)  # (64x25x1x26+11)
            if qst is not None:
                qst = torch.unsqueeze(qst, 1)
                x_j = torch.cat([x_j, qst], 2)
            x_j = x_j.repeat(1, num_components, 1)  # (64x25x25x26+11)

            # concatenate all together
            x_full = torch.cat([x_i, x_j], 2)  # (25x25x2*26+11)

            # reshape for passing through network
            x_ = x_full.view(num_components ** 2, (32 + 2 if self.args.pre_relational else n_channels + 2) * 2 + qst_size)

            x_ = self.g_fc1(x_)
            x_ = self.activation(x_)
            x_ = self.g_fc2(x_)
            x_ = self.activation(x_)
            x_ = self.g_fc3(x_)
            x_ = self.activation(x_)
            x_ = self.g_fc4(x_)
            x_ = self.activation(x_)

            # reshape again and sum
            x_g = x_.view(num_components ** 2, 256)
            x_g = x_g.sum(0).squeeze()

            x_f = self.f_fc1(x_g)
            x_f = self.activation(x_f)

            # Return max proba
            return self.fcout(x_f).max().to("cpu")

        hessian = torch.zeros((mb, num_components, 32 + 2 if self.args.pre_relational else n_channels + 2, num_components, 32 + 2 if self.args.pre_relational else n_channels + 2))
        for b in range(mb):
            t = time.time()
            hessian[b] = torch.autograd.functional.hessian(lambda _x: downstream(_x.to("cpu"),
                                                                                 qstn[b].to("cpu") if qstn is not None else None),
                                                           x_flat[b], create_graph=False, strict=True)
            print("Hessian:", b+1, "/", mb, "Time: ", time.time() - t, "s")
        gradcam = False
        if gradcam:
            jacobian = torch.zeros((mb, num_components, n_channels + 2))
            for b in range(mb):
                jacobian[b] = torch.autograd.functional.jacobian(lambda _x: downstream(_x, qstn[b] if qstn is not None else None), x_flat[b],
                                                                 create_graph=False, strict=True)
            alpha_c_k = jacobian.mean(1)
            gc = x_flat * nn.ReLU()(alpha_c_k[:, None, :])
            gc = gc.sum(-1)
            outs = torch.zeros(mb, gc.shape[-1], gc.shape[-1])
            for i in range(mb):
                outs[i] = torch.diag(gc[i])
            return outs.reshape(mb, height, width, height, width)
        self.to(self.device)
        return utils.TaylorCAM(x_flat.to(self.device), hessian.to(self.device), 0, 1, 3, 2, 4).reshape(mb, height, width, height, width)


class CNN_MLP(BasicModel):
    def __init__(self, args, device):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv = ConvInputModel()
        self.fc1 = nn.Linear(5 * 5 * 24 + 11, 256)  # question concatenated to all

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        if args.gelu:
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.fcout = FCOutputModel(self.activation, self.args)

    def forward(self, img, qst):
        x = self.conv(img)  ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)

        x_ = torch.cat((x, qst), 1)  # Concat question

        x_ = self.fc1(x_)
        x_ = self.activation(x_)

        return self.fcout(x_)

    def interaction_effects(self, img, qstn):
        x_flat = self.conv(img)

        mb = x_flat.size()[0]
        n_channels = x_flat.size()[1] - 2
        height = x_flat.size()[2]
        width = x_flat.size()[3]
        num_components = height * width

        x_flat = x_flat.view(-1, n_channels + 2, num_components).permute(0, 2, 1)

        def downstream(x_, qst):
            x_ = x_.unsqueeze(0)
            qst = qst.unsqueeze(0)
            x_ = x_.reshape(x_.size(0), -1)
            x_ = torch.cat((x_, qst), 1)  # Concat question
            x_ = self.fc1(x_)
            x_ = self.activation(x_)
            return self.fcout(x_).max()

        hessian = torch.zeros((mb, num_components, n_channels + 2, num_components, n_channels + 2))
        for b in range(mb):
            t = time.time()
            hessian[b] = torch.autograd.functional.hessian(lambda _x: downstream(_x, qstn[b]), x_flat[b],
                                                           create_graph=False, strict=True)
            print("Hessian:", b+1, "/", mb, "Time: ", time.time() - t, "s")
        return utils.TaylorCAM(x_flat, hessian, 0, 1, 3, 2, 4).reshape(mb, height, width, height, width)


class Pool(BasicModel):
    def __init__(self, args, device):
        super(Pool, self).__init__(args, 'Pool')

        self.conv = ConvInputModel()

        self.g_pre_fc1 = nn.Linear(24 + 2 + 11, 256)
        self.g_pre_fc2 = nn.Linear(256, 256)
        self.g_pre_fc3 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2).to(device)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2).to(device)
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i / 5 - 2) / 2., (i % 5 - 2) / 2.]

        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2).to(device)
        self.coord_tensor = Variable(self.coord_tensor)
        # TODO adapt to input size
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        if args.gelu:
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.fcout = FCOutputModel(self.activation, self.args)

    def forward(self, img, qst):
        x = self.conv(img)  ## x = (64 x 24 x 5 x 5)

        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        num_components = x.size()[2] * x.size()[3]
        qst_size = qst.size()[1]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb, n_channels, num_components).permute(0, 2, 1)

        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)

        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1, num_components, 1)

        feature_vecs = torch.cat([x_flat, qst], dim=-1).view(-1, n_channels + 2 + qst_size)
        g_pre = nn.ReLU()(self.g_pre_fc1(feature_vecs))
        g_pre = nn.ReLU()(self.g_pre_fc2(g_pre))
        g_pre = nn.ReLU()(self.g_pre_fc3(g_pre))

        # reshape and sum
        x_g = g_pre.view(mb, num_components, 256)
        x_g = x_g.sum(1).squeeze()

        """f"""
        x_f = self.f_fc1(x_g)
        x_f = self.activation(x_f)

        return self.fcout(x_f)

    def interaction_effects(self, img, qstn):
        x = self.conv(img)  ## x = (64 x 24 x 5 x 5)

        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        height = x.size()[2]
        width = x.size()[3]
        num_components = height * width
        qst_size = qstn.size()[1]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb, n_channels, num_components).permute(0, 2, 1)

        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)

        # add question everywhere
        qst = torch.unsqueeze(qstn, 1)
        qst = qst.repeat(1, num_components, 1)

        feature_vecs = torch.cat([x_flat, qst], dim=-1).view(-1, n_channels + 2 + qst_size)
        g_pre = nn.ReLU()(self.g_pre_fc1(feature_vecs))
        g_pre = nn.ReLU()(self.g_pre_fc2(g_pre))
        x_flat = nn.ReLU()(self.g_pre_fc3(g_pre)).view(mb, num_components, 256)

        def downstream(x_):
            x_g = x_.sum(0).squeeze()
            x_f = self.f_fc1(x_g)
            x_f = self.activation(x_f)
            return self.fcout(x_f).max()

        hessian = torch.zeros((mb, num_components, 256, num_components, 256))
        for b in range(mb):
            t = time.time()
            hessian[b] = torch.autograd.functional.hessian(downstream, x_flat[b], create_graph=False, strict=True)
            print("Hessian:", b+1, "/", mb, "Time: ", time.time() - t, "s")
        return utils.TaylorCAM(x_flat, hessian, 0, 1, 3, 2, 4).reshape(mb, height, width, height, width)
