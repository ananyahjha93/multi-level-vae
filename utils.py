import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms as transforms
import pdb
import numpy as np
from torchvision import datasets
from my_mnist import *
transform_config = transforms.Compose([transforms.ToTensor()])

def accumulate_group_evidence(FLAGS,class_mu, class_logvar, labels_batch, is_cuda):
    # convert logvar to variance for calculations
    content_mu = []
    content_inv_logvar = []
    list_groups_labels = []
    sizes_group = []
    groups = (labels_batch).unique()
    # calculate var inverse for each group using group vars
    for _, g in enumerate(groups):
        group_label = g.item()
        samples_group = labels_batch.eq(group_label).nonzero().squeeze()

        if samples_group.numel()>0:
            inv_group_logvar =  - class_logvar[samples_group,:]
            # multiply by inverse variance
            inv_group_var = torch.exp(inv_group_logvar)
            group_mu = class_mu[samples_group,:] * inv_group_var

            if samples_group.numel()>1:
                group_mu = group_mu.sum(0,keepdim=True)
                inv_group_logvar = torch.logsumexp(inv_group_logvar,
                                            dim=0,keepdim=True)
            else:
                group_mu = group_mu[None,:]
                inv_group_logvar = inv_group_logvar[None,:]

            content_mu.append(group_mu)
            content_inv_logvar.append(inv_group_logvar)
            list_groups_labels.append(group_label)
            sizes_group.append(samples_group.numel())

    content_mu = torch.cat(content_mu,dim=0)
    content_inv_logvar = torch.cat(content_inv_logvar, dim=0)
    sizes_group = torch.FloatTensor(sizes_group)
    # inverse log variance
    content_logvar =  - content_inv_logvar
    # multiply group var with group log variance
    content_mu = content_mu * torch.exp(content_logvar)
    return content_mu, content_logvar, list_groups_labels, sizes_group

def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = torch.zeros_like(std).normal_()
        return eps.mul(std).add_(mu)
    else:
        return mu

def group_wise_reparameterize(training, mu, logvar, labels_batch,
                            list_groups_labels, sizes_group, cuda):
    eps_dict = {}
    batch_size = labels_batch.size(0)
    # generate only 1 eps value per group label
    for i, g in enumerate(list_groups_labels):
        if cuda:
            eps_dict[g] = torch.cuda.FloatTensor(1, logvar.size(1)).normal_()
        else:
            eps_dict[g] = torch.FloatTensor(1, logvar.size(1)).normal_()

    if training:
        std = logvar.mul(0.5).exp_()
    else:
        std =torch.zeros_like(logvar)

    content_samples = []
    indexes = []
    sizes = []
    # multiply std by correct eps and add mu
    for i, g in enumerate(list_groups_labels):
        samples_group = labels_batch.eq(g).nonzero().squeeze()
        size_group = samples_group.numel()
        assert size_group == sizes_group[i]
        if size_group > 0:

            reparametrized = std[i][None,:] * eps_dict[g] + mu[i][None,:]
            group_content_sample = reparametrized.repeat((size_group,1))
            content_samples.append(group_content_sample)
            if size_group == 1:
                samples_group = samples_group[None]
            indexes.append(samples_group)
            size_group = torch.ones(size_group) * size_group
            sizes.append(size_group)

    content_samples = torch.cat(content_samples,dim=0)
    indexes = torch.cat(indexes)
    sizes = torch.cat(sizes)

    return content_samples, indexes, sizes

def group_wise_reparameterize_each(training, mu, logvar, labels_batch,
                            list_groups_labels, sizes_group, cuda):
    eps_dict = {}
    batch_size = labels_batch.size(0)

    if training:
        std = logvar.mul(0.5).exp_()
    else:
        std =torch.zeros_like(logvar)

    content_samples = []
    indexes = []
    sizes = []
    # multiply std by correct eps and add mu
    for i, g in enumerate(list_groups_labels):
        samples_group = labels_batch.eq(g).nonzero().squeeze()
        size_group = samples_group.numel()
        assert size_group == sizes_group[i]
        if size_group > 0:
            if cuda:
                eps = torch.cuda.FloatTensor(size_group, std.size(1)).normal_()
            else:
                eps = torch.FloatTensor(size_group, std.size(1)).normal_()
            group_content_sample = std[i][None,:] * eps + mu[i][None,:]
            content_samples.append(group_content_sample)
            if size_group == 1:
                samples_group = samples_group[None]
            indexes.append(samples_group)
            size_group = torch.ones(size_group) * size_group
            sizes.append(size_group)

    content_samples = torch.cat(content_samples,dim=0)
    indexes = torch.cat(indexes)
    sizes = torch.cat(sizes)

    return content_samples, indexes, sizes

def weights_init(layer):
    r"""Apparently in Chainer Lecun normal initialisation was the default one
    """
    if isinstance(layer, nn.Linear):
        lecun_normal_(layer.bias)
        lecun_normal_(layer.weight)

def lecun_normal_(tensor, gain=1):

    import math
    r"""Adapted from https://pytorch.org/docs/0.4.1/_modules/torch/nn/init.html#xavier_normal_
    """
    dimensions = tensor.size()
    if len(dimensions) == 1:  # bias
        fan_in = tensor.size(0)
    elif len(dimensions) == 2:  # Linear
        fan_in = tensor.size(1)
    else:
        num_input_fmaps = tensor.size(1)
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size

    std = gain * math.sqrt(1.0 / (fan_in))
    with torch.no_grad():
        return tensor.normal_(0, std)

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
