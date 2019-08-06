import os
import numpy as np
from itertools import cycle

import torch
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils import weights_init
from utils import transform_config
from networks import Encoder, Decoder
from torch.utils.data import DataLoader
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence


def training_procedure(FLAGS):
    """
    model definition
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    encoder.apply(weights_init)

    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder.apply(weights_init)

    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        encoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.encoder_save)))
        decoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.decoder_save)))

    """
    variable definition
    """
    X = torch.FloatTensor(FLAGS.batch_size, 1, FLAGS.image_size, FLAGS.image_size)

    '''
    add option to run on GPU
    '''
    if FLAGS.cuda:
        encoder.cuda()
        decoder.cuda()

        X = X.cuda()

    """
    optimizer definition
    """
    auto_encoder_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    """
    training
    """
    if torch.cuda.is_available() and not FLAGS.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # load_saved is false when training is started from 0th iteration
    if not FLAGS.load_saved:
        with open(FLAGS.log_file, 'w') as log:
            log.write('Epoch\tIteration\tReconstruction_loss\tStyle_KL_divergence_loss\tClass_KL_divergence_loss\n')

    # load data set and create data loader instance
    print('Loading MNIST dataset...')
    mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
    loader = cycle(DataLoader(mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    # initialize summary writer
    writer = SummaryWriter()

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('')
        print('Epoch #' + str(epoch) + '..........................................................................')

        for iteration in range(int(len(mnist) / FLAGS.batch_size)):
            # load a mini-batch
            image_batch, labels_batch = next(loader)

            # set zero_grad for the optimizer
            auto_encoder_optimizer.zero_grad()

            X.copy_(image_batch)

            style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(X))
            grouped_mu, grouped_logvar = accumulate_group_evidence(
                class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
            )

            # kl-divergence error for style latent space
            style_kl_divergence_loss = FLAGS.kl_divergence_coef * (
                    - 0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
            )
            style_kl_divergence_loss /= (FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size)
            style_kl_divergence_loss.backward(retain_graph=True)

            # kl-divergence error for class latent space
            class_kl_divergence_loss = FLAGS.kl_divergence_coef * (
                    - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
            )
            class_kl_divergence_loss /= (FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size)
            class_kl_divergence_loss.backward(retain_graph=True)

            # reconstruct samples
            """
            sampling from group mu and logvar for each image in mini-batch differently makes
            the decoder consider class latent embeddings as random noise and ignore them 
            """
            style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
            class_latent_embeddings = group_wise_reparameterize(
                training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
            )

            reconstructed_images = decoder(style_latent_embeddings, class_latent_embeddings)

            reconstruction_error = FLAGS.reconstruction_coef * mse_loss(reconstructed_images, Variable(X))
            reconstruction_error.backward()

            auto_encoder_optimizer.step()

            if (iteration + 1) % 50 == 0:
                print('')
                print('Epoch #' + str(epoch))
                print('Iteration #' + str(iteration))

                print('')
                print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                print('Style KL-Divergence loss: ' + str(style_kl_divergence_loss.data.storage().tolist()[0]))
                print('Class KL-Divergence loss: ' + str(class_kl_divergence_loss.data.storage().tolist()[0]))

            # write to log
            with open(FLAGS.log_file, 'a') as log:
                log.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                    epoch,
                    iteration,
                    reconstruction_error.data.storage().tolist()[0],
                    style_kl_divergence_loss.data.storage().tolist()[0],
                    class_kl_divergence_loss.data.storage().tolist()[0]
                ))

            # write to tensorboard
            writer.add_scalar('Reconstruction loss', reconstruction_error.data.storage().tolist()[0],
                              epoch * (int(len(mnist) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('Style KL-Divergence loss', style_kl_divergence_loss.data.storage().tolist()[0],
                              epoch * (int(len(mnist) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('Class KL-Divergence loss', class_kl_divergence_loss.data.storage().tolist()[0],
                              epoch * (int(len(mnist) / FLAGS.batch_size) + 1) + iteration)

        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
            torch.save(encoder.state_dict(), os.path.join('checkpoints', FLAGS.encoder_save))
            torch.save(decoder.state_dict(), os.path.join('checkpoints', FLAGS.decoder_save))
