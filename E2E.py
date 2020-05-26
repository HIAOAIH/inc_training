import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from torchvision import transforms
from RevisedResNet import RevisedResNet
from dataset_with_class import SingleClassData

from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset


class EndToEnd(object):
    def __init__(self, args):
        self.train_epoch = args.train_epoch
        self.lr = args.lr
        self.t = 1 / 2
        self.batch_size = args.batch_size
        self.use_gpu = args.use_gpu
        self.K = args.k

        self.weight_decay = args.weight_decay
        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()

        self.NLLLoss = nn.NLLLoss()
        self.softmax = nn.Softmax()

        self.task_size_per_phase = [0]
        self.trained_class_num = args.init_class_num
        self.discriminator = RevisedResNet(args.init_class_num)
        if self.use_gpu:
            self.discriminator = self.discriminator.cuda(self.device_num)

        self.optimizer = optim.SGD(self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.exemplars = {}
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

        self.device_num = args.device_num

    # def cross_distilled_loss(self, epoch, x, y, t):
    #     loss = self.CELoss(x, y)
    #     for i in range(len(self.task_size_per_phase) - 2):
    #         q = self.softmax(x[self.task_size_per_phase[i]:self.task_size_per_phase[i + 1]])
    #         q = torch.log(torch.pow(q, t))
    #         loss += self.NLLLoss(q, y[self.task_size_per_phase[i]:self.task_size_per_phase[i + 1]])
    #
    #     # TODO: 함수로 만들지 않고 train 안에서 backward 할 수 있도록
    #     #  이건 gradient가 아니네........
    #     # eed to add gradient noise
    #     # torch.distributions.normal.Normal(loc, scale, validate_args=None)
    #     # torch.distributions.normal.Normal
    #     # scale (float or Tensor) – standard deviation of the distribution (often referred to as sigma)
    #
    #     # d = torch.distributions.normal.Normal(0., 0.3 / (2 + epoch) ** 0.55)
    #     # (1 + t)여야 하지만, epoch이 0부터 시작하므로
    #
    #     return loss

    def train(self, train_data):
        added_class_num = len(train_data)
        self.discriminator.append_weights(added_class_num)
        self.task_size_per_phase.append(added_class_num)
        self.trained_class_num += added_class_num

        self.discriminator.train()

        self.optimizer = optim.SGD(self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        dataset = ConcatDataset(train_data + list(self.exemplars.values()))
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
        for epoch in range(self.train_epoch):
            distribution = torch.distributions.normal.Normal(0., 0.3 / (2 + epoch) ** 0.55)
            for _, (index, x, y) in enumerate(data_loader):
                if self.use_gpu:
                    x, y = x.cuda(self.device_num), y.cuda(self.device_num)

                self.optimizer.zero_grad()
                q = self.discriminator(x)
                loss = self.CELoss(q, y)
                for i in range(len(self.task_size_per_phase) - 2):
                    q_dist = self.softmax(x[self.task_size_per_phase[i]:self.task_size_per_phase[i + 1]])
                    q_dist = torch.log(torch.pow(q_dist, self.t))
                    loss += self.NLLLoss(q_dist, y[self.task_size_per_phase[i]:self.task_size_per_phase[i + 1]])

                loss.backward()
                for p in self.discriminator.parameters():
                    if p.grad is not None:
                        p.grad += distribution.sample(p.grad.shape).cuda(self.device_num)
                self.optimizer.step()
                if _ % 10 == 0:
                    print('process: {} / {}, loss: {:.4f}'.format((epoch + 1), self.train_epoch, loss.item()))

            self.scheduler.step(epoch)

        exemplar_num = self.K // self.trained_class_num
        if self.trained_class_num != added_class_num:
            self.reduce_exemplar_set(exemplar_num)
        self.construct_exemplar_set(train_data, exemplar_num)

    def fine_tune(self):
        return

    def test(self, test_data):
        self.discriminator.eval()
        dataset = ConcatDataset(test_data)
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
        total = len(dataset)
        correct = 0

        confusion_matrix = torch.zeros(self.trained_class_num, self.trained_class_num).type(torch.long)

        with torch.no_grad():
            for _, (index, x, y) in enumerate(data_loader):
                if self.use_gpu:
                    x, y = x.cuda(self.device_num), y.cuda(self.device_num)
                output = self.discriminator(x)
                label = self.softmax(output).argmax(dim=1)

                for n, m in zip(y.view(-1, 1), label.view(-1, 1)):
                    confusion_matrix[n, m] += 1

                correct += label.eq(y).long().cpu().sum() if self.use_gpu else label.eq(y).long().sum()
                confusion_matrix = confusion_matrix.numpy()
                df_cm = pd.DataFrame(confusion_matrix, index=[i for i in range(self.trained_class_num)],
                                     columns=[i for i in range(self.trained_class_num)])
                plt.xlabel('real label')
                plt.ylabel('classification result')
                plt.figure(figsize=(7 * self.trained_class_num // 10, 5 * self.trained_class_num // 10))
                sn.heatmap(df_cm, annot=True)
                plt.savefig('./confusion_matrix/e2e/' + str(self.trained_class_num) + '_heatmap.png', dpi=300)

            print("Accuracy: {}/{} ({:.2f}%)".format(correct, total, 100. * correct / total))

    def construct_exemplar_set(self, train_data, exemplar_num):
        added_class_num = len(train_data)
        for n in range(len(train_data)):
            train_data[n].transform = self.transform
            dl = torch.utils.data.Dataloader(train_data[n], batch_size=len(train_data[n].targets), shuffle=False)
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
            exemplar = SingleClassData(self.transform, [], [])
            exemplar.data = train_data[n].data[index_list]
            exemplar.targets = train_data[n].targets[index_list]
            label = train_data[n].targets[0]
            self.exemplars[label] = exemplar

            print('process: {} / {} ({:.2f}%)'.format(n + 1, added_class_num, 100 * (n + 1) / added_class_num))

        # TODO: index_list를 data와 target의 index로 사용하여 dataset 만들기

    def reduce_exemplar_set(self, exemplar_num):
        for key in self.exemplars:
            self.exemplars[key].data = self.exemplars[key].data[:exemplar_num]
            self.exemplars[key].targets = self.exemplars[key].targets[:exemplar_num]
