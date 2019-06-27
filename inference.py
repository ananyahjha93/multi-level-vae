import os
import argparse
import numpy as np
from itertools import cycle

import torch
import random
import pickle
from torchvision import datasets
from torch.autograd import Variable
from alternate_data_loader import MNIST_Paired
from utils import accumulate_group_evidence, reparameterize, group_wise_reparameterize

import matplotlib.pyplot as plt
from utils import transform_config
from networks import Encoder, Decoder
from torch.utils.data import DataLoader

from mpl_toolkits.axes_grid1 import ImageGrid

parser = argparse.ArgumentParser()

parser.add_argument('--reference_data', type=str, default='fixed', help="generate output using random digits or fixed reference")
parser.add_argument('--accumulate_evidence', type=int, default=0, help="accumulate class evidence before producing swapped images")
parser.add_argument('--cuda', type=bool, default=False, help="run the following code on a GPU")
parser.add_argument('--batch_size', type=int, default=256, help="batch size for training")
parser.add_argument('--image_size', type=int, default=28, help="height and width of the image")
parser.add_argument('--num_channels', type=int, default=1, help="number of channels in images")
parser.add_argument('--fixed_var', type=int, default=1)
parser.add_argument('--sigmoid', type=int, default=0)
parser.add_argument('--style_dim', type=int, default=10, help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=10, help="dimension of common factor latent space")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes on which the data set trained")
# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder', help="model save for decoder")
parser.add_argument('--end_epoch', type=int, default=400)

FLAGS = parser.parse_args()

if __name__ == '__main__':
    """
    model definitions
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)

    folder = 'checkpoints_%d'%(FLAGS.batch_size)
    monitor = torch.load(os.path.join(folder, 'monitor_e%d'%(FLAGS.end_epoch -1)))

    print(folder)
    checks = np.arange(0,FLAGS.end_epoch+5,5)
    checks[1:] -= 1
    monitor[0,0] =  - np.inf
    best_elbo = monitor[checks,0].argmax()

    #DEBUG
    best_elbo=-1
    print(checks[best_elbo],monitor[checks[best_elbo],0],monitor[checks,0].max(), best_elbo)
    FLAGS.encoder_save = folder + '/encoder_e%d' % checks[best_elbo]
    FLAGS.decoder_save = folder + '/decoder_e%d' % checks[best_elbo]

    FLAGS.batch_size = 256
    encoder.load_state_dict(
        torch.load(FLAGS.encoder_save, map_location=lambda storage, loc: storage))
    decoder.load_state_dict(
        torch.load(FLAGS.decoder_save, map_location=lambda storage, loc: storage))

    if FLAGS.cuda:
        encoder.cuda()
        decoder.cuda()

    if not os.path.exists('reconstructed_images'):
        os.makedirs('reconstructed_images')

    # load data set and create data loader instance
    print('Loading MNIST paired dataset...')
    paired_mnist = MNIST_Paired(root='mnist', download=True, train=False, transform=transform_config)
    loader = cycle(DataLoader(paired_mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    image_array = []
    for i in range(0, 11):
        image_array.append([])

    # add a blank image in the top row
    image_array[0].append(np.zeros((28, 28, 3), dtype=np.float32))

    sampled_classes = list(paired_mnist.data_dict.keys())
    sampled_classes.sort()

    if FLAGS.reference_data == 'random':
        count = 1
        # fill the top row and first column of the grid with reference images
        for class_sample in sampled_classes:
            class_image = random.SystemRandom().choice(paired_mnist.data_dict[class_sample])
            class_image = np.transpose(class_image.numpy(), (1, 2, 0))
            class_image = np.concatenate((class_image, class_image, class_image), axis=2)

            # add image in the top row
            image_array[0].append(class_image)

            # add image in the first column
            image_array[count].append(class_image)

            count += 1

    elif FLAGS.reference_data == 'fixed':
        with open("reference_images.pkl", "rb") as fp:
            reference_images = pickle.load(fp)

        count = 1
        # fill the top row and first column of the grid with reference images
        for i in range(0, 10):
            # add image in the top row
            image_array[0].append(reference_images[i])

            # add image in the first column
            image_array[count].append(reference_images[i])

            count += 1

    # get class specified label for entire top row at once
    specified_factor_images = []
    for i in range(1, 11):
        specified_factor_images.append(image_array[0][i][:, :, 0])

    specified_factor_images = np.asarray(specified_factor_images)
    specified_factor_images = np.expand_dims(specified_factor_images, axis=3)
    specified_factor_images = np.transpose(specified_factor_images, (0, 3, 1, 2))
    specified_factor_images = torch.FloatTensor(specified_factor_images)
    specified_factor_images = specified_factor_images.contiguous()

    if FLAGS.accumulate_evidence:
        # sample a big batch, accumulate evidence and use that for class embeddings
        image_batch, _, labels_batch = next(loader)
        _, __, class_mu, class_logvar = encoder(image_batch)
        content_mu, content_logvar, list_g, sizes_group = accumulate_group_evidence(FLAGS,
                class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
            )

    # generate all possible combinations using the encoder and decoder architecture in the grid
    for row in range(1, 11):
        style_image = image_array[row][0]
        style_image = np.transpose(style_image, (2, 0, 1))
        style_image = torch.FloatTensor(style_image)
        style_image = style_image.contiguous()
        style_image = style_image[0, :, :]
        style_image = style_image.view(1, 1, 28, 28)
        style_mu, style_logvar, _, _ = encoder(Variable(style_image))
        style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)

        for col in range(1, 11):
            if not FLAGS.accumulate_evidence:
                class_image = image_array[col][0]
                class_image = np.transpose(class_image, (2, 0, 1))
                class_image = torch.FloatTensor(class_image)
                class_image = class_image.contiguous()
                class_image = class_image[0, :, :]
                class_image = class_image.view(1, 1, 28, 28)
                _, _, class_mu, class_logvar = encoder(Variable(class_image))
                specified_factor_temp = reparameterize(training=True, mu=class_mu, logvar=class_logvar)
            else:
                group_index = list_g.index(col-1)
                mu_group = content_mu[group_index,:]
                log_var_group = content_logvar[group_index,:]
                std = torch.exp(log_var_group)**0.5
                eps = torch.FloatTensor(mu_group.size()).normal_()
                group_content_sample = std * eps + mu_group
                specified_factor_temp = group_content_sample

            specified_factor_temp = specified_factor_temp.view(1, FLAGS.class_dim)

            mu_x, logvar_x = decoder(style_latent_embeddings, specified_factor_temp)
            if FLAGS.sigmoid:
                mu_x = torch.sigmoid(mu_x)
            reconstructed_x = mu_x.permute(0, 2, 3, 1).detach().numpy()[0]
            reconstructed_x = np.concatenate((reconstructed_x, reconstructed_x, reconstructed_x), axis=2)
            image_array[row].append(reconstructed_x)

    # plot
    image_array = np.asarray(image_array)
    print(image_array.shape)

    fig = plt.figure(1)

    grid = ImageGrid(fig, 111, nrows_ncols=[11, 11], axes_pad=0.)

    temp_list = []

    for i in range(0, 11):
        for j in range(0, 11):
            temp_list.append(image_array[i][j])

    for i in range(121):
        grid[i].axis('off')
        im = temp_list[i]
        im[im<0] = 0
        im[im>1] = 1
        grid[i].imshow(im)
    plt.show()
    # plt.savefig('reconstructed_images/inference.png', bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.clf()
