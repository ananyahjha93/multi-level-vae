import torch
import random

from torchvision import datasets
from utils import transform_config
from torch.utils.data import Dataset


class MNIST_Paired(Dataset):
    def __init__(self, root='mnist', download=True, train=True, transform=transform_config):
        self.mnist = datasets.MNIST(root=root, download=download, train=train, transform=transform)

        self.data_dict = {}

        for i in range(self.__len__()):
            image, label = self.mnist.__getitem__(i)

            try:
                self.data_dict[label.item()]
            except KeyError:
                self.data_dict[label.item()] = []
            self.data_dict[label.item()].append(image)

    def __len__(self):
        return self.mnist.__len__()

    def __getitem__(self, index):
        image, label = self.mnist.__getitem__(index)

        # return another image of the same class randomly selected from the data dictionary
        # this is done to simulate pair-wise labeling of data
        return image, random.SystemRandom().choice(self.data_dict[label.item()]), label
