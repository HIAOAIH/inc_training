import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from resnet import RevisedResNet
from dataset_with_class import SingleClassData

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset


class BicLayer(nn.Module):
    def __init__(self):
        super(BicLayer, self).__init__()
        self.alpha = torch.ones(1)
        self.beta = torch.zeros(1)

    def forward(self, x):
        return self.alpha * x + self.beta

    def cuda(self, device=0):
        self.alpha = self.alpha.cuda(device)
        self.beta = self.beta.cuda(device)

    def train(self, train=True):
        self.alpha.requires_grad_(train)
        self.beta.requires_grad_(train)

    def eval(self):
        self.train(train=False)

    def get_params(self):
        return [self.alpha, self.beta]


class LargeScale(object):
    def __init__(self, args):
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.momentum = 0.9
        self.weight_decay = args.weight_decay
        self.device_num = args.device_num
        self.trained_class_num = args.init_class_num
        self.use_gpu = args.use_gpu

        self.net = RevisedResNet(self.trained_class_num)
        if self.trained_class_num != 0:
            self.net.load_state_dict(torch.load('./' + args.network_dir + '/ls/large_scale_to_' + str(self.trained_class_num) + '.pt'))
        self.bias_layers = []

        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.exemplars = {}
        self.K = 2000

    def bias_forward(self, x, classes_per_train):
        output = []
        for i in range(len(self.bias_layers)):
            output.append(self.bias_layers[i](x[:, i * classes_per_train: (i + 1) * classes_per_train]))
        return torch.cat(output, dim=1)

    def train(self, train_data, test_data):
        added_class_num = len(train_data)
        previous_net = None
        if self.trained_class_num != 0:
            previous_net = copy.deepcopy(self.net)
            previous_net.eval()
            previous_net.cuda(self.device_num)

        self.net.append_weights(added_class_num)
        self.net.cuda(self.device_num)

        opt_stage1 = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler_stage1 = optim.lr_scheduler.MultiStepLR(opt_stage1, milestones=[100, 150, 200], gamma=0.1)

        # self.alpha = torch.randn(1).requires_grad_(False)
        # self.beta = torch.randn(1).requires_grad_(False)
        self.bias_layers.append(BicLayer())
        self.bias_layers[-1].cuda(self.device_num)
        print('BiC layers')
        for layer in self.bias_layers:
            print(layer.get_params())

        # opt_stage2 = optim.SGD(self.bias_layers[-1].get_params(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        opt_stage2 = optim.Adam(self.bias_layers[-1].get_params(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler_stage2 = optim.lr_scheduler.MultiStepLR(opt_stage2, milestones=[100, 150, 200], gamma=0.1)

        exemplar_train_set = []
        exemplar_validation_set = []

        if len(self.exemplars) != 0:
            exemplar_size = len(self.exemplars.values().__iter__().__next__().targets)
            validation_size = exemplar_size // 10

            for item in self.exemplars.values():
                validation_index = random.sample(range(exemplar_size), validation_size)
                train_index = list(set(range(exemplar_size)) - set(validation_index))

                validation = copy.deepcopy(item)
                train = copy.deepcopy(item)

                train.data = train.data[train_index]
                train.targets = train.targets[train_index]
                train.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

                validation.data = validation.data[validation_index]
                validation.targets = validation.targets[validation_index]
                validation.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

                exemplar_train_set.append(train)
                exemplar_validation_set.append(validation)

        else:
            validation_size = 50

        new_train_set = []
        new_validation_set = []

        for n in range(len(train_data)):
            validation_index = random.sample(range(500), validation_size)
            train_index = list(set(range(500)) - set(validation_index))

            validation = copy.deepcopy(train_data[n])
            train = copy.deepcopy(train_data[n])

            train.data = train.data[train_index]
            train.targets = train.targets[train_index]

            validation.data = validation.data[validation_index]
            validation.targets = validation.targets[validation_index]

            new_train_set.append(train)
            new_validation_set.append(validation)

        train_data_loader = DataLoader(ConcatDataset(new_train_set + exemplar_train_set), shuffle=True,
                                       batch_size=self.batch_size)
        validation_data_loader = DataLoader(ConcatDataset(new_validation_set + exemplar_validation_set), shuffle=True,
                                            batch_size=self.batch_size)
        # train_data의 일부는 훈련에, 일부는 validation(bias correction)에 사용

        # self.alpha = self.alpha.cuda(self.device_num)
        # self.beta = self.beta.cuda(self.device_num)

        for epoch in range(self.train_epoch):
            # train data
            self.net.train()
            self.bias_layers[-1].eval()

            # self.alpha.requires_grad = False
            # self.beta.requires_grad = False
            for i, (index, images, targets) in enumerate(train_data_loader):
                images, targets = images.cuda(self.device_num), targets.cuda(self.device_num)
                opt_stage1.zero_grad()
                # opt_stage2.zero_grad()

                features = self.net(images)

                # alpha_mat = torch.ones(features.shape)
                # alpha_mat[:, -added_class_num:] *= alpha

                # beta_mat = torch.zeros(features.shape)
                # beta_mat[:, -added_class_num:] += beta
                #
                # alpha_mat, beta_mat = alpha_mat.cuda(self.device_num), beta_mat.cuda(self.device_num)
                #
                # features = alpha_mat * features + beta_mat
                # features[:, -added_class_num:] = out

                # features[:, -added_class_num:] = alpha * features[:, -added_class_num:] + beta
                # features = features * alpha + beta

                # TODO: stage 1 훈련시에도 alpha, beta 계산을 해야 하는지?
                output = self.bias_forward(features, added_class_num)
                classification_loss = self.CELoss(output, targets)

                # out = self.alpha * features.clone()[:, -added_class_num:] + self.beta
                # classification_loss = self.CELoss(torch.cat([features[:, :-added_class_num], out], dim=1), targets)

                if previous_net is not None:
                    with torch.no_grad():
                        old_features = previous_net(images)
                        old_output = self.bias_forward(old_features, added_class_num)
                        pi_hat = self.softmax(old_output / 2)
                    # old_exp_features = old_features.exp() ** 0.5
                    # old_exp_softmax = old_exp_features / old_exp_features.sum(dim=1).view(-1, 1)

                    # exp_features = features[..., :self.trained_class_num].exp() ** 0.5
                    # exp_softmax = exp_features / exp_features.sum(dim=1).view(-1, 1)
                    pi = self.log_softmax(output[..., :self.trained_class_num] / 2)

                    # distilling_loss = (-old_exp_softmax * exp_softmax.log()).sum()
                    distilling_loss = - (pi_hat * pi).sum(dim=1).mean()

                    loss = 2 * (self.trained_class_num / (self.trained_class_num + added_class_num)) * distilling_loss + (
                            added_class_num / (self.trained_class_num + added_class_num)) * classification_loss
                else:
                    loss = classification_loss

                loss.backward()
                opt_stage1.step()
                if i % 10 == 0:
                    print('process: {} / {}, loss: {:.4f}, classification_loss: {:.4f}'.format(
                        (epoch + 1), self.train_epoch, loss.item(), classification_loss.item()))
            scheduler_stage1.step(epoch)

            # if epoch % 10 == 9:
            #     self.test(test_data)

            self.net.eval()
            self.bias_layers[-1].train()
            # self.alpha.requires_grad = True
            # self.beta.requires_grad = True
        # for epoch in range(self.train_epoch):
            # validation data
            if len(self.bias_layers) > 1 and epoch > 149:
                for i, (index, images, targets) in enumerate(validation_data_loader):
                    images, targets = images.cuda(self.device_num), targets.cuda(self.device_num)
                    # opt_stage1.zero_grad()
                    opt_stage2.zero_grad()

                    features = self.net(images)
                    # features = alpha_mat * features + beta_mat
                    # out = self.alpha * features.clone()[:, -added_class_num:] + self.beta
                    output = self.bias_forward(features, added_class_num)
                    bias_loss = self.CELoss(output, targets)
                    # bias_loss = self.CELoss(torch.cat([features[:, :-added_class_num], out], dim=1), targets)

                    # features[:, -added_class_num:] = alpha * features[:, -added_class_num:] + beta
                    # bias_loss = self.CELoss(features, targets)
                    bias_loss.backward()
                    opt_stage2.step()
                    print('process: {} / {}, bias loss: {:.4f}'.format((epoch + 1), self.train_epoch, bias_loss.item()))
                scheduler_stage2.step(epoch)

            if epoch % 10 == 9:
                print('process: {} / {}, BiC layers: '.format((epoch + 1), self.train_epoch))
                for layer in self.bias_layers:
                    print(layer.get_params())
                self.test(test_data)

        self.trained_class_num += added_class_num

        exemplar_num = self.K // self.trained_class_num

        self.reduce_exemplar_set(exemplar_num)
        self.construct_exemplar_set(train_data, exemplar_num)

    # test 코드
    def test(self, test_data, save_conf_mat=False):
        self.net.eval()
        self.bias_layers[-1].eval()

        # dataset = ConcatDataset(test_data)
        # data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
        # total = len(dataset)
        total_correct = 0
        total_dataset_len = 0

        # confusion_matrix = torch.zeros(self.trained_class_num, self.trained_class_num).type(torch.long)

        with torch.no_grad():
            for i in range(len(test_data) // 20):
                dataset = ConcatDataset(test_data[20 * i: 20 * (i + 1)])
                data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
                dataset_len = len(dataset)
                correct = 0

                for _, (index, x, y) in enumerate(data_loader):
                    if self.use_gpu:
                        x, y = x.cuda(self.device_num), y.cuda(self.device_num)
                    output = self.net(x)
                    # TODO: 10: added_class_num -> 따로 저장해둬야
                    output = self.bias_forward(output, 20)
                    label = self.softmax(output).argmax(dim=1)

                    # if save_conf_mat:
                    #     for n, m in zip(y.view(-1, 1), label.view(-1, 1)):
                    #         confusion_matrix[n, m] += 1

                    correct += label.eq(y).long().cpu().sum() if self.use_gpu else label.eq(y).long().sum()
                print("Accuracy: {}/{} ({:.2f}%)".format(correct, dataset_len, 100. * correct / dataset_len))

                total_correct += correct
                total_dataset_len += dataset_len

            # if save_conf_mat:
            #     confusion_matrix = confusion_matrix.numpy()
            #     df_cm = pd.DataFrame(confusion_matrix, index=[i for i in range(self.trained_class_num)],
            #                          columns=[i for i in range(self.trained_class_num)])
            #     plt.xlabel('real label')
            #     plt.ylabel('classification result')
            #     plt.figure(figsize=(7 * self.trained_class_num // 10, 5 * self.trained_class_num // 10))
            #     sn.heatmap(df_cm, annot=True)
            #     plt.savefig('./confusion_matrix/ls/' + str(self.trained_class_num) + '_heatmap.png', dpi=300)

        print("Total Accuracy: {}/{} ({:.2f}%)".format(total_correct, total_dataset_len, 100. * total_correct / total_dataset_len))

    # exemplar
    def construct_exemplar_set(self, train_data, exemplar_num):
        added_class_num = len(train_data)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

        for n in range(len(train_data)):
            train_data[n].transform = transform

            dl = torch.utils.data.DataLoader(train_data[n], batch_size=len(train_data[n].targets), shuffle=False)
            data = dl.__iter__().__next__()
            mean = torch.mean(data[1], dim=0)

            distance = []
            for i in range(len(data[1])):  # data가 (index, x, y일 경우)
                distance.append(self.MSELoss(mean, data[1][i]))

            distance = torch.tensor(distance)
            max_value = torch.max(distance)
            index_list = []

            for i in range(exemplar_num):
                index = torch.argmin(distance)
                index_list.append(int(index))
                distance[index] = max_value

            train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

            exemplar = SingleClassData(train_transforms, [], [])
            exemplar.data = train_data[n].data[index_list]
            exemplar.targets = train_data[n].targets[index_list]
            label = train_data[n].targets[0]
            self.exemplars[label] = exemplar

            print('process: {} / {} ({:.2f}%)'.format(n + 1, added_class_num, 100 * (n + 1) / added_class_num))

    def reduce_exemplar_set(self, exemplar_num):
        for key in self.exemplars:
            self.exemplars[key].data = self.exemplars[key].data[:exemplar_num]
            self.exemplars[key].targets = self.exemplars[key].targets[:exemplar_num]

