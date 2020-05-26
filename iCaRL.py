import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from dataset_with_class import SingleClassData
from RevisedResNet import RevisedResNet


class Exemplar(SingleClassData):
    def __init__(self, transform, data, target):
        super(Exemplar, self).__init__(transform, data, target)


class ICaRL(object):
    def __init__(self, args):
        self.discriminator = RevisedResNet(args.init_class_num)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        self.criterion = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.class_num = args.init_class_num
        self.exemplars = dict()
        self.K = args.k
        self.use_gpu = args.use_gpu

        self.device_num = args.device_num
        self.discriminator = self.discriminator.cuda(self.device_num) if self.use_gpu else self.discriminator

        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    def classify(self, x):
        answer = torch.zeros(x.shape[0]).type(torch.long)
        answer = answer.cuda(self.device_num) if self.use_gpu else answer
        for i in range(x.shape[0]):
            with torch.no_grad():
                feature = self.discriminator(x[i:i+1], True)
            out = 10000
            class_label = 0
            for label, exemplar in self.exemplars.items():
                exemplar.moe = exemplar.moe.cuda(self.device_num) if self.use_gpu else exemplar.moe
                norm = self.mse_loss(feature.data, exemplar.moe)
                if out > norm:
                    out = norm
                    class_label = label
            answer[i] = class_label
        return answer

    def update_representation(self, train_epoch, train_data, num):
        print('updating representation')

        self.class_num += num
        if self.class_num != num:
            self.discriminator.eval()
            net = copy.deepcopy(self.discriminator)
            net.eval()

        self.discriminator.append_weights(num)
        self.discriminator.cuda(self.device_num) if self.use_gpu else self.discriminator
        self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        dataset = ConcatDataset(train_data + list(self.exemplars.values()))
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        self.discriminator.train()
        for i in range(train_epoch):
            if i == train_epoch * 7 // 10:
                self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=self.lr/5, weight_decay=self.weight_decay, momentum=0.9)
            elif i == train_epoch * 8 // 10:
                self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=self.lr/25, weight_decay=self.weight_decay, momentum=0.9)
            elif i == train_epoch * 9 // 10:
                self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=self.lr/125, weight_decay=self.weight_decay, momentum=0.9)

            for _, (index, x, y) in enumerate(data_loader):
                if self.use_gpu:
                    x = x.cuda(self.device_num)

                self.d_optimizer.zero_grad()
                features = self.discriminator(x, classify=False)
                y_one_hot = torch.zeros(features.shape[0], self.class_num)
                y_one_hot[torch.arange(features.shape[0]), y] = 1
                y_one_hot = y_one_hot.cuda(self.device_num) if self.use_gpu else y_one_hot

                if self.class_num != num:
                    with torch.no_grad():
                        q = net(x, classify=False).data
                    y_one_hot[..., :-num] = q
                loss = self.criterion(features, y_one_hot)

                loss.backward()
                self.d_optimizer.step()
                if _ % 10 == 0:
                    print('process: {} / {}, loss: {:.4f}'.format((i + 1), train_epoch, loss.item()))

        print('update is done')

    def calculate_mean_of_exemplar(self, exemplar):
        dl = DataLoader(exemplar, shuffle=False, batch_size=exemplar.data.shape[0])
        data = dl.__iter__().__next__()[1]
        data = data.cuda(self.device_num) if self.use_gpu else data
        with torch.no_grad():
            moe = self.discriminator(data, classify=True).mean(dim=0)
        exemplar.store_mean_of_exemplar(moe.data)
        return moe

    def calculate_sum_of_exemplar(self, exemplar):
        dl = DataLoader(exemplar, shuffle=False, batch_size=exemplar.data.shape[0])
        data = dl.__iter__().__next__()[1]
        data = data.cuda(self.device_num) if self.use_gpu else data
        with torch.no_grad():
            soe = self.discriminator(data, classify=True).sum(dim=0)
        return soe

    def reduce_exemplar_set(self, exemplar_num):
        for key in self.exemplars:
            self.exemplars[key].data = self.exemplars[key].data[:exemplar_num]
            self.exemplars[key].targets = self.exemplars[key].targets[:exemplar_num]

            self.calculate_mean_of_exemplar(self.exemplars[key])

    def construct_exemplar_set(self, input_data, added_class_num, exemplar_num):
        print('constructing exemplar sets')
        for i in range(added_class_num):
            single_class_data = copy.deepcopy(input_data[i])
            single_class_data.transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
            label = int(single_class_data.targets[0])
            exemplar = SingleClassData(self.transform, [], [])

            # mu is not a mean of exemplar because input is not exemplar. but can get with the same way
            mu = self.calculate_mean_of_exemplar(single_class_data)

            for j in range(exemplar_num):
                dl = DataLoader(single_class_data, shuffle=False, batch_size=self.batch_size)
                soe = self.calculate_sum_of_exemplar(exemplar) if j != 0 else 0
                p = 100000
                tmp = 0
                for _, (index, x, y) in enumerate(dl):
                    x = x.cuda(self.device_num) if self.use_gpu else x
                    out = (self.discriminator(x, classify=True) + soe) / (_ + 1)
                    for k in range(out.shape[0]):
                        if self.mse_loss(mu, out[k]) < p:
                            p = self.mse_loss(mu, out[k])
                            tmp = int(index[k])
                exemplar.data = single_class_data.data[tmp:tmp + 1] if j == 0 \
                    else np.concatenate((exemplar.data, single_class_data.data[tmp:tmp + 1]))
                exemplar.targets = single_class_data.targets[0:1] if j == 0\
                    else torch.cat([exemplar.targets, single_class_data.targets[0:1]])
                single_class_data.data = np.concatenate((single_class_data.data[:tmp], single_class_data.data[tmp+1:]))
                single_class_data.targets = single_class_data.targets[:-1]

            self.exemplars[label] = exemplar
            self.calculate_mean_of_exemplar(self.exemplars[label])

            print('process: {} / {} ({:.2f}%)'.format(i + 1, added_class_num, 100 * (i + 1) / added_class_num))

    def train(self, x):
        num = len(x)
        print('training {} classes'.format(self.class_num + num))
        self.update_representation(self.train_epoch, x, num)
        self.discriminator.eval()
        exemplar_num = self.K // self.class_num
        print('there will be {} exemplars in each class'.format(exemplar_num))
        if self.class_num != num:
            self.reduce_exemplar_set(exemplar_num)
        self.construct_exemplar_set(x, num, exemplar_num)

    def test(self, eval_data, eval=True):
        concat_dataset = ConcatDataset(eval_data)
        total_num = len(concat_dataset)
        data_loader = DataLoader(concat_dataset, shuffle=True, batch_size=self.batch_size)
        correct = 0

        t = 'test' if eval else 'train'
        confusion_matrix = torch.zeros(self.class_num, self.class_num).type(torch.long)

        with torch.no_grad():
            for i, (index, x, y) in enumerate(data_loader):
                if self.use_gpu:
                    x, y = x.cuda(self.device_num), y.cuda(self.device_num)
                label = self.classify(x)

                for n, m in zip(y.view(-1, 1), label.view(-1, 1)):
                    confusion_matrix[n, m] += 1

                correct += label.eq(y).long().cpu().sum() if self.use_gpu else label.eq(y).long().sum()
            confusion_matrix = confusion_matrix.numpy()
            df_cm = pd.DataFrame(confusion_matrix, index=[i for i in range(self.class_num)],
                                 columns=[i for i in range(self.class_num)])
            plt.xlabel('real label')
            plt.ylabel('classification result')
            plt.figure(figsize=(7 * self.class_num // 10, 5 * self.class_num // 10))
            sn.heatmap(df_cm, annot=True)
            plt.savefig('./confusion_matrix/icarl/' + t + '_' + str(self.class_num) + '_heatmap.png', dpi=300)

        if eval:
            print("Using test data. Accuracy: {}/{} ({:.2f}%)".format(correct, total_num, 100. * correct / total_num))
        else:
            print("Using train data. Accuracy: {}/{} ({:.2f}%)".format(correct, total_num, 100. * correct / total_num))

