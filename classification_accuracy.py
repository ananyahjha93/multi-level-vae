import os
import argparse
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable

from torch.utils.data import DataLoader
from utils import transform_config
from networks import Encoder, Decoder, Classifier
from utils import weights_init, accumulate_group_evidence, reparameterize, group_wise_reparameterize

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--cuda', type=bool, default=False, help="run the following code on a GPU")
parser.add_argument('--accumulate_evidence', type=str, default=False, help="accumulate class evidence before producing swapped images")

parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
parser.add_argument('--image_size', type=int, default=28, help="height and width of the image")
parser.add_argument('--num_channels', type=int, default=1, help="number of channels in the image")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes on which the data set trained")

parser.add_argument('--num_test_samples', type=int, default=10000, help="number of test samples")
parser.add_argument('--num_train_samples', type=int, default=60000, help="number of train samples")

parser.add_argument('--initial_learning_rate', type=float, default=0.0001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")

parser.add_argument('--style_dim', type=int, default=10, help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=10, help="dimension of common factor latent space")

# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder_1_var_reparam', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder_1_var_reparam', help="model save for decoder")

parser.add_argument('--end_iteration', type=int, default=100000, help="flag to indicate the final epoch of training")


FLAGS = parser.parse_args()

if __name__ == '__main__':
    """
    model definitions
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)

    encoder.load_state_dict(
        torch.load(os.path.join('checkpoints', FLAGS.encoder_save), map_location=lambda storage, loc: storage))
    decoder.load_state_dict(
        torch.load(os.path.join('checkpoints', FLAGS.decoder_save), map_location=lambda storage, loc: storage))

    # class labels variable
    X = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size)
    class_labels = torch.LongTensor(FLAGS.batch_size)

    # test
    if torch.cuda.is_available() and not FLAGS.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # load data set and create data loader instance
    print('Loading MNIST dataset...')
    mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
    loader = cycle(DataLoader(mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    style_classifier = Classifier(z_dim=FLAGS.style_dim, num_classes=FLAGS.num_classes)
    style_classifier.apply(weights_init)

    class_classifier = Classifier(z_dim=FLAGS.class_dim, num_classes=FLAGS.num_classes)
    class_classifier.apply(weights_init)

    cross_entropy_loss = nn.CrossEntropyLoss()

    style_classifier_optimizer = optim.Adam(
        list(style_classifier.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    class_classifier_optimizer = optim.Adam(
        list(class_classifier.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    if FLAGS.cuda:
        encoder.cuda()
        decoder.cuda()
        style_classifier.cuda()
        class_classifier.cuda()

        X = X.cuda()
        class_labels = class_labels.cuda()

    count = 0

    # training
    for i in range(0, FLAGS.end_iteration):
        image_batch, labels_batch = next(loader)

        class_labels.copy_(labels_batch)
        X.copy_(image_batch)

        style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(X))
        style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)

        if FLAGS.accumulate_evidence:
            grouped_mu, grouped_logvar = accumulate_group_evidence(
                class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
            )
            class_latent_embeddings = group_wise_reparameterize(
                training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
            )
        else:
            class_latent_embeddings = reparameterize(training=True, mu=class_mu, logvar=class_logvar)

        style_classifier_optimizer.zero_grad()

        # Style
        style_classifier_pred = style_classifier(style_latent_embeddings)

        style_classification_error = cross_entropy_loss(style_classifier_pred, Variable(class_labels))
        style_classification_error.backward(retain_graph=True)

        _, style_classifier_pred = torch.max(style_classifier_pred, 1)
        style_classifier_accuracy = (style_classifier_pred.data == class_labels).sum().item() / FLAGS.batch_size

        style_classifier_optimizer.step()

        class_classifier_optimizer.zero_grad()

        # Class
        class_classifier_pred = class_classifier(class_latent_embeddings)

        class_classification_error = cross_entropy_loss(class_classifier_pred, Variable(class_labels))
        class_classification_error.backward()

        _, class_classifier_pred = torch.max(class_classifier_pred, 1)
        class_classifier_accuracy = (class_classifier_pred.data == class_labels).sum().item() / FLAGS.batch_size

        class_classifier_optimizer.step()

        if count % 100 == 0:
            print('Count: ' + str(count))
            print('Style classifier accuracy: ' + str(style_classifier_accuracy))
            print('Class classifier accuracy: ' + str(class_classifier_accuracy))
            print('\n')

        count += 1

    # load data set and create data loader instance
    print('Loading MNIST dataset...')
    mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
    loader = cycle(DataLoader(mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    total_style_classifier_accuracy = 0.
    total_class_classifier_accuracy = 0.

    for i in range(0, FLAGS.num_train_samples // FLAGS.batch_size):
        image_batch, labels_batch = next(loader)

        class_labels.copy_(labels_batch)
        X.copy_(image_batch)

        style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(X))
        style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)

        if FLAGS.accumulate_evidence:
            grouped_mu, grouped_logvar = accumulate_group_evidence(
                class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
            )
            class_latent_embeddings = group_wise_reparameterize(
                training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
            )
        else:
            class_latent_embeddings = reparameterize(training=True, mu=class_mu, logvar=class_logvar)

        style_classifier_pred = style_classifier(style_latent_embeddings)
        style_classification_error = cross_entropy_loss(style_classifier_pred, Variable(class_labels))

        _, style_classifier_pred = torch.max(style_classifier_pred, 1)
        style_classifier_accuracy = (style_classifier_pred.data == class_labels).sum().item() / FLAGS.batch_size

        class_classifier_pred = class_classifier(class_latent_embeddings)
        class_classification_error = cross_entropy_loss(class_classifier_pred, Variable(class_labels))

        _, class_classifier_pred = torch.max(class_classifier_pred, 1)
        class_classifier_accuracy = (class_classifier_pred.data == class_labels).sum().item() / FLAGS.batch_size

        total_style_classifier_accuracy += style_classifier_accuracy
        total_class_classifier_accuracy += class_classifier_accuracy

    print('Style classifier train accuracy: ' + str(total_style_classifier_accuracy / (FLAGS.num_train_samples // FLAGS.batch_size)))
    print('Class classifier train accuracy: ' + str(total_class_classifier_accuracy / (FLAGS.num_train_samples // FLAGS.batch_size)))
    print('\n')

    # load data set and create data loader instance
    print('Loading MNIST dataset...')
    mnist = datasets.MNIST(root='mnist', download=True, train=False, transform=transform_config)
    loader = cycle(DataLoader(mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    total_style_classifier_accuracy = 0.
    total_class_classifier_accuracy = 0.

    for i in range(0, FLAGS.num_test_samples // FLAGS.batch_size):
        image_batch, labels_batch = next(loader)

        class_labels.copy_(labels_batch)
        X.copy_(image_batch)

        style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(X))
        style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)

        if FLAGS.accumulate_evidence:
            grouped_mu, grouped_logvar = accumulate_group_evidence(
                class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
            )
            class_latent_embeddings = group_wise_reparameterize(
                training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
            )
        else:
            class_latent_embeddings = reparameterize(training=True, mu=class_mu, logvar=class_logvar)

        style_classifier_pred = style_classifier(style_latent_embeddings)
        style_classification_error = cross_entropy_loss(style_classifier_pred, Variable(class_labels))

        _, style_classifier_pred = torch.max(style_classifier_pred, 1)
        style_classifier_accuracy = (style_classifier_pred.data == class_labels).sum().item() / FLAGS.batch_size

        class_classifier_pred = class_classifier(class_latent_embeddings)
        class_classification_error = cross_entropy_loss(class_classifier_pred, Variable(class_labels))

        _, class_classifier_pred = torch.max(class_classifier_pred, 1)
        class_classifier_accuracy = (class_classifier_pred.data == class_labels).sum().item() / FLAGS.batch_size

        total_style_classifier_accuracy += style_classifier_accuracy
        total_class_classifier_accuracy += class_classifier_accuracy

    print('Style classifier test accuracy: ' + str(total_style_classifier_accuracy / (FLAGS.num_test_samples // FLAGS.batch_size)))
    print('Class classifier test accuracy: ' + str(total_class_classifier_accuracy / (FLAGS.num_test_samples // FLAGS.batch_size)))
