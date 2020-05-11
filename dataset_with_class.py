import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import datasets, transforms


class SingleClassData(VisionDataset):
    def __init__(self, transform, data, targets):
        super(SingleClassData, self).__init__(root='./data', transform=transform)
        self.transform = transform
        self.data = data
        self.targets = targets

    def store_mean_of_exemplar(self, moe):
        self.moe = moe

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return index, img, target

    def __len__(self):
        return len(self.targets)


def dataset_with_class(train=True):
    if train:
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(15),
                                        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    dataset = datasets.CIFAR100('./data', train=train, transform=transform, download=True)
    data_list = []
    for i in range(100):
        index = torch.tensor(dataset.targets) == i
        data = dataset.data[index]
        targets = torch.tensor(dataset.targets)[index]
        single_class_data = SingleClassData(transform, data, targets)
        data_list.append(single_class_data)
    return data_list
