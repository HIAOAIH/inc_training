import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torchvision import transforms
from torchvision.models.resnet import ResNet, BasicBlock

from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from dataset_with_class import SingleClassData


class RevisedResNet(ResNet):
    def __init__(self, out_features=0):
        self.out_features = out_features
        super(RevisedResNet, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.fc = nn.Linear(512, self.out_features, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, classify=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).view(-1, 512)
        if classify:
            return x
        else:
            return self.sigmoid(self.fc(x))

    def append_weights(self, num):
        with torch.no_grad():
            fc = nn.Linear(512, self.out_features + num, True)
            fc.weight[:self.out_features] = self.fc.weight
            fc.bias[:self.out_features] = self.fc.bias
            self.out_features += num
            self.fc = fc


class Exemplar(SingleClassData):
    def __init__(self, transform, data, target):
        super(Exemplar, self).__init__(transform, data, target)

    def store_mean_of_exemplar(self, moe):
        self.moe = moe


class ICaRL(object):
    def __init__(self, args):
        self.discriminator = RevisedResNet(args.class_num)
        self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=args.lr, momentum=args.momentum)
        self.criterion = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.class_num = args.class_num
        self.exemplars = dict()
        self.K = args.k

        self.batch_size = args.batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def classify(self, x):
        answer = torch.zeros(x.shape[0]).type(torch.LongTensor)
        for i in range(x.shape[0]):
            feature = self.discriminator(x[i], True)
            out = 10000
            class_label = 0
            for label, exemplar in self.exemplars:
                if out > self.mse_loss(feature, exemplar.moe):
                    out = self.mse_loss(feature, exemplar.moe)
                    class_label = label
            answer[i] = class_label
        return answer

    def update_representation(self, epoch, x, num):
        self.discriminator.append_weights(num)
        self.class_num += num
        self.discriminator.train()

        data_loader = DataLoader(ConcatDataset(x + list(self.exemplars.values())), shuffle=True, batch_size=self.batch_size)
        for i in range(epoch):
            for _, (x, y) in enumerate(data_loader):
                self.d_optimizer.zero_grad()
                features = self.discriminator(x, classify=False)
                y_one_hot = torch.zeros(features.shape[0], self.class_num)
                y_one_hot[torch.arange(features.shape[0]), y] = 1
                loss = self.distillation_loss(features[:, :-num], y_one_hot[:, :-num]) + \
                       self.classification_loss(features[:, -num:], y_one_hot[:, -num:])
                loss.backward()
                self.d_optimizer.step()

    def reduce_exemplar_set(self, exemplar_num):
        for key in self.exemplars:
            self.exemplars[key].data = self.exemplars[key].data[:exemplar_num]
            self.exemplars[key].targets = self.exemplars[key].targets[:exemplar_num]
            # is it necessary to recalculate moe when reduce exemplar set?
            self.calculate_mean_of_exemplar(self.exemplars[key])

    def calculate_mean_of_exemplar(self, exemplar):
        dl = DataLoader(exemplar, shuffle=False, batch_size=exemplar.data.shape[0])
        data = dl.__iter__().__next__()[0]
        moe = self.discriminator(data, classify=False).mean(dim=0)
        exemplar.store_mean_of_exemplar(moe)
        return moe

    def calculate_sum_of_exemplar(self, exemplar):
        dl = DataLoader(exemplar, shuffle=False, batch_size=exemplar.data.shape[0])
        data = dl.__iter__().__next__()[0]
        moe = self.discriminator(data, classify=False).sum(dim=0)
        return moe

    def construct_exemplar_set(self, input_data, added_class_num, exemplar_num):
        for i in range(added_class_num):
            # input_data의 정답 label을 저장
            label = int(input_data[i].targets[0])
            exemplar = Exemplar(self.transform, [], [])
            mu = self.calculate_mean_of_exemplar(input_data[i])

            for j in range(exemplar_num):
                dl = DataLoader(input_data[i], shuffle=False, batch_size=self.batch_size)
                soe = self.calculate_sum_of_exemplar(exemplar) if j != 0 else 0
                p = 100000
                tmp = 0
                for index, (x, y) in enumerate(dl):
                    out = (self.discriminator(x) + soe) / (index + 1)
                    if self.mse_loss(mu, out) < p:
                        tmp = index
                exemplar.data = input_data[i].data[tmp:tmp + 1] if j == 0 \
                    else np.concatenate((exemplar.data, input_data[i].data[tmp:tmp + 1]))
                input_data[i].data = np.concatenate((input_data[i].data[:tmp], input_data[i].data[tmp+1:]))
                input_data[i].targets = input_data[i].targets[:-1]
            exemplar.targets = input_data[i].targets[:exemplar_num]
            self.exemplars[label] = exemplar
            self.calculate_mean_of_exemplar(self.exemplars[label])

    def distillation_loss(self, x, y):
        # TODO: 미리 구해둔 q와 y를 곱해야 함
        # y = y * q
        loss = self.criterion(x, y)
        return loss

    def classification_loss(self, x, y):
        loss = self.criterion(x, y)
        return loss

    def train(self, x, num):
        # TODO
        self.update_representation(70, x, num)
        exemplar_num = self.K // self.class_num
        self.reduce_exemplar_set(exemplar_num)
        self.construct_exemplar_set(x, num, exemplar_num)

    def test(self, image):
        # TODO
        self.discriminator.eval()
        self.classify(image)
        return
