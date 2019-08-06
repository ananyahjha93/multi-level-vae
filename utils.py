import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Compose, ToTensor


# compose a transform configuration
transform_config = Compose([ToTensor()])


def accumulate_group_evidence(class_mu, class_logvar, labels_batch, is_cuda):
    """
    :param class_mu: mu values for class latent embeddings of each sample in the mini-batch
    :param class_logvar: logvar values for class latent embeddings for each sample in the mini-batch
    :param labels_batch: class labels of each sample (the operation of accumulating class evidence can also
        be performed using group labels instead of actual class labels)
    :param is_cuda:
    :return:
    """
    unique_labels = torch.unique(labels_batch)
    n_unique_labels = len(unique_labels)
    if is_cuda:
        var_inv_matrix = torch.zeros((class_logvar.size()[1], n_unique_labels)).cuda()
        mu_matrix = torch.zeros((class_mu.size()[1], n_unique_labels)).cuda()
    else:
        var_inv_matrix = torch.zeros((class_logvar.size()[1], n_unique_labels))
        mu_matrix = torch.zeros((class_mu.size()[1], n_unique_labels))

    # convert logvar to variance for calculations
    class_var = class_logvar.exp_()
    class_var[class_var==float(0)] = 1e-6
    class_var_inv = 1/class_var

    # calculate var inverse for each group using group vars
    for i in range(n_unique_labels):
        var_inv_matrix[:, i] = torch.sum(class_var_inv[labels_batch == unique_labels[i]], 0)
    # invert var inverses to calculate mu and return value
    var_matrix = 1 / var_inv_matrix

    # calculate mu for each group
    for i in range(n_unique_labels):
        mu_matrix[:, i] = torch.sum(class_mu[labels_batch == unique_labels[i]]*
                                    class_var_inv[labels_batch == unique_labels[i]], 0)

    # multiply group var with sums calculated above to get mu for the group
    mu_matrix *= var_matrix

    # replace individual mu and logvar values for each sample with group mu and logvar
    group_mu = torch.FloatTensor(class_mu.size(0), class_mu.size(1))
    group_var = torch.FloatTensor(class_var.size(0), class_var.size(1))

    if is_cuda:
        group_mu = group_mu.cuda()
        group_var = group_var.cuda()

    for i in range(n_unique_labels):
        label = unique_labels[i]
        num_element = torch.sum(labels_batch == label)
        group_mu[labels_batch == label] = mu_matrix[:, i].unsqueeze(0).repeat(num_element, 1)
        group_var[labels_batch == label] = var_matrix[:, i].unsqueeze(0).repeat(num_element, 1)

    # remove 0 from var before taking log
    group_var[group_var == float(0)] = 1e-6

    # convert group vars into logvars before returning
    return Variable(group_mu, requires_grad=True), Variable(torch.log(group_var), requires_grad=True)


def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def group_wise_reparameterize(training, mu, logvar, labels_batch, cuda):
    unique_labels = torch.unique(labels_batch)
    n_unique_labels = len(unique_labels)
    eps_dict = {}

    # generate only 1 eps value per group label
    for label in unique_labels:
        if cuda:
            eps_dict[label.item()] = torch.cuda.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)
        else:
            eps_dict[label.item()] = torch.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)

    if training:
        # multiply std by correct eps and add mu
        std = logvar.mul(0.5).exp_()
        reparameterized_var = Variable(std.data.new(std.size()))
        for i in range(n_unique_labels):
            label = unique_labels[i]
            labels_true = labels_batch == label
            reparameterized_var[labels_true] = std[labels_true].mul(Variable(eps_dict[label.item()])).add(
                mu[labels_true])

        return reparameterized_var
    else:
        return mu


def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()


def imshow_grid(images, shape=[2, 8], name='default', save=False):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    if save:
        plt.savefig('reconstructed_images/' + str(name) + '.png')
        plt.clf()
    else:
        plt.show()
