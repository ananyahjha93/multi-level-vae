import torch
import torch.nn as nn
from collections import OrderedDict

from itertools import cycle
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import transform_config, reparameterize


class Encoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Encoder, self).__init__()

        self.linear_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(in_features=784, out_features=500, bias=True)),
            ('tan_h_1', nn.Tanh())
        ]))

        # style
        self.style_mu = nn.Linear(in_features=500, out_features=style_dim, bias=True)
        self.style_logvar = nn.Linear(in_features=500, out_features=style_dim, bias=True)

        # class
        self.class_mu = nn.Linear(in_features=500, out_features=class_dim, bias=True)
        self.class_logvar = nn.Linear(in_features=500, out_features=class_dim, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.linear_model(x)

        style_latent_space_mu = self.style_mu(x)
        style_latent_space_logvar = self.style_logvar(x)

        class_latent_space_mu = self.class_mu(x)
        class_latent_space_logvar = self.class_logvar(x)

        return style_latent_space_mu, style_latent_space_logvar, class_latent_space_mu, class_latent_space_logvar


class Decoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Decoder, self).__init__()

        self.linear_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(in_features=style_dim + class_dim, out_features=500, bias=True)),
            ('tan_h_1', nn.Tanh()),

            ('linear_2', nn.Linear(in_features=500, out_features=784, bias=True)),
            ('sigmoid_final', nn.Sigmoid())
        ]))

    def forward(self, style_latent_space, class_latent_space):
        x = torch.cat((style_latent_space, class_latent_space), dim=1)

        x = self.linear_model(x)
        x = x.view(x.size(0), 1, 28, 28)

        return x


class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=z_dim, out_features=256, bias=True)),
            ('fc_1_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_2', nn.Linear(in_features=256, out_features=256, bias=True)),
            ('fc_2_bn', nn.BatchNorm1d(num_features=256)),
            ('LeakyRelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_3', nn.Linear(in_features=256, out_features=num_classes, bias=True))
        ]))

    def forward(self, z):
        x = self.fc_model(z)

        return x


if __name__ == '__main__':
    """
    test network outputs
    """
    encoder = Encoder(10, 10)
    decoder = Decoder(10, 10)

    classifier = Classifier(z_dim=16, num_classes=10)

    mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
    loader = cycle(DataLoader(mnist, batch_size=64, shuffle=True, num_workers=0, drop_last=True))

    image_batch, labels_batch = next(loader)

    style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))

    style_reparam = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
    class_reparam = reparameterize(training=True, mu=class_mu, logvar=class_logvar)

    reconstructed_image = decoder(style_reparam, class_reparam)

    style_classifier_pred = classifier(style_reparam)
    class_classifier_pred = classifier(class_reparam)

    print(reconstructed_image.size())
    print(style_classifier_pred.size())
    print(class_classifier_pred.size())
