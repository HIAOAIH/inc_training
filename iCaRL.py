import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torchvision import transforms
# from torchvision.models.resnet import ResNet, BasicBlock

from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from resnet import RevisedResNet, ResidualBlock
from dataset_with_class import SingleClassData


# class RevisedResNet(ResNet):
#     def __init__(self, out_features=0):
#         self.out_features = out_features
#         super(RevisedResNet, self).__init__(BasicBlock, [3, 4, 6, 3])
#         self.fc = nn.Linear(512, self.out_features, False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, classify=False):
#         x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
#         x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
#         x = self.avgpool(x).view(-1, 512)
#         if classify:
#             return x
#         else:
#             return self.sigmoid(self.fc(x))
#
#     def append_weights(self, num):
#         with torch.no_grad():
#             fc = nn.Linear(512, self.out_features + num, False)
#             fc.weight[:self.out_features] = self.fc.weight.data
#             # fc.bias[:self.out_features] = self.fc.bias
#             self.out_features += num
#             self.fc = fc


class Exemplar(SingleClassData):
    def __init__(self, transform, data, target):
        super(Exemplar, self).__init__(transform, data, target)


class ICaRL(object):
    def __init__(self, args):
        self.discriminator = RevisedResNet(ResidualBlock, [2, 2, 2], args.init_class_num)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.BCELoss(reduction='sum')
        self.mse_loss = nn.MSELoss()
        self.class_num = args.init_class_num
        self.exemplars = dict()
        self.K = args.k
        self.use_gpu = args.use_gpu
        self.discriminator = self.discriminator.cuda() if self.use_gpu else self.discriminator

        self.batch_size = args.batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def classify(self, x):
        answer = torch.zeros(x.shape[0]).type(torch.LongTensor)
        answer = answer.cuda() if self.use_gpu else answer
        for i in range(x.shape[0]):
            with torch.no_grad():
                feature = self.discriminator(x[i:i+1], True)# .requires_grad_(False)
            out = 10000
            class_label = 0
            for label, exemplar in self.exemplars.items():
                exemplar.moe = exemplar.moe.cuda() if self.use_gpu else exemplar.moe
                norm = self.mse_loss(feature.data, exemplar.moe)
                if out > norm:
                    out = norm
                    class_label = label
            answer[i] = class_label
        return answer

    def update_representation(self, epoch, train_data, num):
        print('updating representation')
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.discriminator.append_weights(num)
        self.discriminator.cuda() if self.use_gpu else self.discriminator
        self.class_num += num

        dataset = ConcatDataset(train_data + list(self.exemplars.values()))
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        q = torch.zeros(len(dataset), self.class_num)
        q = q.cuda() if self.use_gpu else q
        for _, (index, x, y) in enumerate(data_loader):
            if self.use_gpu:
                x = x.cuda()
            g = self.discriminator(x, classify=False)
            q[index] = g.data

        for i in range(epoch):
            if i == 49:
                self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr/5, weight_decay=self.weight_decay)
            elif i == 63:
                self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr/25, weight_decay=self.weight_decay)

            for _, (index, x, y) in enumerate(data_loader):
                if self.use_gpu:
                    x = x.cuda()

                self.d_optimizer.zero_grad()
                features = self.discriminator(x, classify=False)
                y_one_hot = torch.zeros(features.shape[0], self.class_num)
                y_one_hot[torch.arange(features.shape[0]), y] = 1
                y_one_hot = y_one_hot.cuda() if self.use_gpu else y_one_hot

                # loss = self.distillation_loss(features[:, :-num], y_one_hot[:, :-num]) + \
                if self.class_num == num:
                    loss = self.classification_loss(features[:, -num:], y_one_hot[:, -num:])
                else:
                    loss = self.distillation_loss(features[:, :-num], q[index, :-num]) + \
                           self.classification_loss(features[:, -num:], y_one_hot[:, -num:])
                loss.backward()
                self.d_optimizer.step()
            if i % 10 == 9:
                print('process: {} / {} ({:.2f}%)'.format((i + 1), epoch, 100 * (i + 1) / epoch))
        print('update is done')

    def calculate_mean_of_exemplar(self, exemplar):
        dl = DataLoader(exemplar, shuffle=False, batch_size=exemplar.data.shape[0])
        data = dl.__iter__().__next__()[1]
        data = data.cuda() if self.use_gpu else data
        with torch.no_grad():
            moe = self.discriminator(data, classify=True).mean(dim=0)
        exemplar.store_mean_of_exemplar(moe.data)
        return moe

    def calculate_sum_of_exemplar(self, exemplar):
        dl = DataLoader(exemplar, shuffle=False, batch_size=exemplar.data.shape[0])
        data = dl.__iter__().__next__()[1]
        data = data.cuda() if self.use_gpu else data
        with torch.no_grad():
            soe = self.discriminator(data, classify=True).sum(dim=0)
        return soe

    def reduce_exemplar_set(self, exemplar_num):
        for key in self.exemplars:
            self.exemplars[key].data = self.exemplars[key].data[:exemplar_num]
            self.exemplars[key].targets = self.exemplars[key].targets[:exemplar_num]

            # isn't it necessary to recalculate moe when reduce exemplar set?
            # self.calculate_mean_of_exemplar(self.exemplars[key])

    def construct_exemplar_set(self, input_data, added_class_num, exemplar_num):
        print('constructing exemplar sets')
        for i in range(added_class_num):
            label = int(input_data[i].targets[0])
            exemplar = SingleClassData(self.transform, [], [])

            # mu is not a mean of exemplar because input is not exemplar. but can get with the same way
            mu = self.calculate_mean_of_exemplar(input_data[i])
            # TODO: torch.argmin() 사용할 수 있는지 확인

            for j in range(exemplar_num):
                dl = DataLoader(input_data[i], shuffle=False, batch_size=self.batch_size)
                soe = self.calculate_sum_of_exemplar(exemplar) if j != 0 else 0
                p = 100000
                tmp = 0
                for _, (index, x, y) in enumerate(dl):
                    x = x.cuda() if self.use_gpu else x
                    out = (self.discriminator(x, classify=True) + soe) / (_ + 1)
                    for k in range(out.shape[0]):
                        if self.mse_loss(mu, out[k]) < p:
                            p = self.mse_loss(mu, out[k])
                            tmp = int(index[k])
                exemplar.data = input_data[i].data[tmp:tmp + 1] if j == 0 \
                    else np.concatenate((exemplar.data, input_data[i].data[tmp:tmp + 1]))
                exemplar.targets = input_data[i].targets[0:1] if j == 0 else torch.cat([exemplar.targets, input_data[i].targets[0:1]])
                input_data[i].data = np.concatenate((input_data[i].data[:tmp], input_data[i].data[tmp+1:]))
                input_data[i].targets = input_data[i].targets[:-1]

            self.exemplars[label] = exemplar
            self.calculate_mean_of_exemplar(self.exemplars[label])

            print('process: {} / {} ({:.2f}%)'.format(i + 1, added_class_num, 100 * (i + 1) / added_class_num))

    def distillation_loss(self, x, y):
        loss = self.criterion(x, y)
        return loss

    def classification_loss(self, x, y):
        loss = self.criterion(x, y)
        return loss

    def train(self, x, num):
        print('training {} classes'.format(self.class_num + num))
        self.discriminator.train()
        self.update_representation(70, x, num)
        self.discriminator.eval()
        exemplar_num = self.K // self.class_num
        print('there will be {} exemplars in each class'.format(exemplar_num))
        if self.class_num != num:
            self.reduce_exemplar_set(exemplar_num)
        self.construct_exemplar_set(x, num, exemplar_num)

    def test(self, eval_data):
        concat_dataset = ConcatDataset(eval_data)
        total_num = len(concat_dataset)
        data_loader = DataLoader(concat_dataset, shuffle=True, batch_size=self.batch_size)
        correct = 0
        for i, (index, x, y) in enumerate(data_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            label = self.classify(x)
            correct += label.eq(y).long().cpu().sum() if self.use_gpu else label.eq(y).long().sum()

        print("Accuracy: {}/{} ({:.2f}%)".format(correct, total_num, 100. * correct / total_num))

