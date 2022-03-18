import os
import argparse
from itertools import cycle

import torch
from sklearn.manifold import TSNE
from torch.autograd import Variable

import matplotlib.pyplot as plt
from alternate_data_loader import MNIST_Paired
from torch.utils.data import DataLoader
from utils import transform_config
from networks import Encoder, Decoder
from utils import group_wise_reparameterize, accumulate_group_evidence, reparameterize, group_wise_reparameterize_each

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--cuda', type=bool, default=False, help="run the following code on a GPU")
parser.add_argument('--accumulate_evidence', type=int, default=0, help="accumulate class evidence before producing swapped images")

parser.add_argument('--batch_size', type=int, default=64, help="batch size for training")
parser.add_argument('--image_size', type=int, default=28, help="height and width of the image")

parser.add_argument('--num_test_samples', type=int, default=10000, help="number of test samples")
parser.add_argument('--style_dim', type=int, default=10, help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=10, help="dimension of common factor latent space")


# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder_0.1_var_reparam', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder_0.1_var_reparam', help="model save for decoder")


FLAGS = parser.parse_args()

if __name__ == '__main__':
    """
    model definitions
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)

    encoder.load_state_dict(
        torch.load(FLAGS.encoder_save, map_location=lambda storage, loc: storage))
    decoder.load_state_dict(
        torch.load(FLAGS.decoder_save, map_location=lambda storage, loc: storage))

    """
    variable definition
    """
    z_space = torch.FloatTensor(1, FLAGS.style_dim)

    '''
    test
    '''
    # load data set and create data loader instance
    print('Loading MNIST paired dataset...')
    paired_mnist = MNIST_Paired(root='mnist', download=True, train=False, transform=transform_config)
    loader = cycle(DataLoader(paired_mnist, batch_size=FLAGS.num_test_samples, shuffle=True, num_workers=0, drop_last=True))

    image_batch, _, labels_batch = next(loader)

    style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))

    if FLAGS.accumulate_evidence:
        content_mu, content_logvar, list_g, sizes_group = accumulate_group_evidence(FLAGS, class_mu, class_logvar, labels_batch, FLAGS.cuda
            )
        class_latent_embeddings, indexes, _ = group_wise_reparameterize_each(
                        training=True, mu=content_mu, logvar=content_logvar, labels_batch=labels_batch, list_groups_labels=list_g,
                        sizes_group=sizes_group, cuda=FLAGS.cuda)
        style_latent_embeddings, indexes, _ =group_wise_reparameterize_each(
                        training=True,mu=style_mu, logvar=style_logvar,
                        labels_batch=labels_batch,list_groups_labels=list_g,
                        sizes_group=sizes_group, cuda=FLAGS.cuda)
        reindexed_indexes = labels_batch[indexes]
    else:
        class_latent_embeddings = reparameterize(training=True, mu=class_mu, logvar=class_logvar)
        style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
        reindexed_indexes = labels_batch


    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    vis_data = pca.fit_transform(class_latent_embeddings.data.numpy())
    # vis_data = TSNE(n_components=2, verbose=1, perplexity=30.0, n_iter=1000).fit_transform(class_latent_embeddings.data.numpy())
    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    fig, ax = plt.subplots(1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.scatter(vis_x, vis_y, marker='.', c=reindexed_indexes.numpy(), cmap=plt.cm.get_cmap("jet", 10),alpha=0.5)
    ax.set_title('Content PCA')
    plt.axis('off')
    plt.clim(-0.5, 9.5)
    plt.show()

    pca = PCA(n_components=2)
    vis_data = pca.fit_transform(style_latent_embeddings.data.numpy())
    # vis_data = TSNE(n_components=2, verbose=1, perplexity=30.0, n_iter=1000).fit_transform(style_latent_embeddings.data.numpy())
    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    fig, ax = plt.subplots(1)
    ax.set_title('Style PCA')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.scatter(vis_x, vis_y, marker='.', c=reindexed_indexes.numpy(), cmap=plt.cm.get_cmap("jet", 10),alpha=0.5)
    plt.axis('off')
    plt.clim(-0.5, 9.5)
    plt.show()
