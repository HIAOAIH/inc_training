import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import ResNet, BasicBlock


class RevisedResNet(ResNet):
    def __init__(self, out_features=100):
        self.out_features = out_features
        super(RevisedResNet, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.fc = nn.Linear(512, self.out_features, True)

    def forward(self, x, classify=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).view(-1, 512)
        # classify 할 경우에는 마지막 fc를 계산하지 않은 feature map을 반환
        if classify:
            return x
        else:
            return self.fc(x)

    def append_weights(self, num):
        fc = nn.Linear(512, self.out_features + num, True)
        fc.weight[:self.out_features] = self.fc.weight
        fc.bias[:self.out_features] = self.fc.bias
        self.out_features += num
        self.fc = fc


class Exemplar(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def store_mean_of_exemplar(self, moe):
        self.moe = moe


class ICaRL(object):
    def __init__(self, args):
        self.discriminator = RevisedResNet(args.class_num)
        self.d_optimizer = optim.SGD(lr=args.lr, momentum=args.momentum)
        self.class_num = args.class_num
        # exemplars는 dictionary로 저장
        self.exemplars = dict()
        self.K = args.k

    def classify(self, x):
        feature = self.discriminator(x, False)
        out = 100000
        class_label = 0

        for label, exemplar in self.exemplars:
            if out > (feature - exemplar.moe) ** 2:
                out = (feature - exemplar.moe) ** 2
                class_label = label
        return class_label

    def update_representation(self, epoch, x, num):
        # representation을 update할 때는 우선 마지막 fc에 num개의 weight를 추가
        self.discriminator.append_weights(num)
        # class_num을 새로 추가할 클래스 수 만큼 증가시킴
        self.class_num += num
        self.discriminator.train()
        for _ in range(epoch):
            self.d_optimizer.zero_grad()
            loss = self.classification_loss() + self.distillation_loss()
            loss.backward()
            self.d_optimizer.step()
        return

    def reduce_exemplar_set(self, exemplar_num):
        for key in self.exemplars:
            # 데이터셋의 데이터 형태 확인할 것
            self.exemplars[key].data = self.exemplars[key].data[:exemplar_num]

    def construct_exemplar_set(self, exemplar_num):
        # exemplar set 만든 후 dictionary에 추가하기
        return

    def classification_loss(self, ):
        return

    def distillation_loss(self, ):
        return

    def train(self, x, num):
        self.update_representation(100, x, num)
        exemplar_num = self.K // self.class_num # 각 클래스의 exemplar 수
        self.reduce_exemplar_set(exemplar_num)
        self.construct_exemplar_set(exemplar_num)

    def test(self, ):
        return
