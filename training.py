import os
import numpy as np
from itertools import cycle

import torch
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from torch.distributions import Normal, Bernoulli
from utils import weights_init
from utils import transform_config
from networks import Encoder, Decoder
from torch.utils.data import DataLoader
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence, group_wise_reparameterize_each
from torch.nn import functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

def process(FLAGS, X, labels_batch, encoder, decoder):

    style_mu, style_logvar, class_mu, class_logvar = encoder(X.cuda())

    content_mu, content_logvar, list_g, sizes_group = \
            accumulate_group_evidence(FLAGS,class_mu, class_logvar,
                                    labels_batch, FLAGS.cuda)
    style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)

    class_latent_embeddings, indexes, sizes = group_wise_reparameterize_each(
        training=True, mu=content_mu, logvar=content_logvar, labels_batch=labels_batch, list_groups_labels=list_g, sizes_group=sizes_group, cuda=FLAGS.cuda)

    # kl-divergence error for style latent space
    style_kl_divergence_loss = 0.5 * ( - 1 - style_logvar[indexes,:] + style_mu[indexes,:].pow(2) + style_logvar[indexes,:].exp()).sum()
    # kl-divergence error for class latent space
    class_kl_divergence_loss = 0.5 * ( - 1 - content_logvar + content_mu.pow(2) + content_logvar.exp()).sum()

    # reconstruct samples
    #reorder by the same order as class_latent_embeddings
    mu_x, logvar_x = decoder(style_latent_embeddings[indexes,:], class_latent_embeddings)
    scale_x = (torch.exp(logvar_x) + 1e-12)**0.5
    scale_x = scale_x.view(X.size(0),784)
    # create normal distribution on output pixel
    mu_x = mu_x.view(X.size(0),784)
    prob_x = Normal(mu_x,scale_x)
    logp_batch = prob_x.log_prob(X[indexes,:].view(X.size(0),784)).sum(1)

    reconstruction_proba = logp_batch.sum(0)
    n_groups = content_mu.size(0)
    elbo = (reconstruction_proba - style_kl_divergence_loss - class_kl_divergence_loss) / n_groups

    return elbo, reconstruction_proba / n_groups, style_kl_divergence_loss/ n_groups, class_kl_divergence_loss / n_groups

def eval(FLAGS, valid_loader, encoder, decoder):

    elbo_epoch = 0
    rec_loss = 0
    kl_style = 0
    kl_content = 0
    for it, (image_batch, labels_batch) in enumerate(valid_loader):

        X = image_batch.cuda().detach().clone()
        elbo, reconstruction_proba, style_kl_divergence_loss,class_kl_divergence_loss = process(FLAGS, X, labels_batch, encoder, decoder)
        elbo_epoch += elbo
        rec_loss += reconstruction_proba
        kl_style += style_kl_divergence_loss
        kl_content += class_kl_divergence_loss

    elbo_epoch /= (it + 1)
    rec_loss /= (it + 1)
    kl_style /= (it + 1)
    kl_content /= (it + 1)

    return torch.FloatTensor([elbo_epoch, rec_loss, kl_style, kl_content])

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
        encoder.load_state_dict(torch.load(os.path.join(savedir, FLAGS.encoder_save)))
        decoder.load_state_dict(torch.load(os.path.join(savedir, FLAGS.decoder_save)))

    '''
    add option to run on GPU
    '''
    if FLAGS.cuda:
        encoder.cuda()
        decoder.cuda()

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

    savedir = 'checkpoints_%d' % (FLAGS.batch_size)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # load_saved is false when training is started from 0th iteration
    if not FLAGS.load_saved:
        with open(FLAGS.log_file, 'w') as log:
            log.write('Epoch\tIteration\tReconstruction_loss\tStyle_KL_divergence_loss\tClass_KL_divergence_loss\n')

    # load data set and create data loader instance
    print('Loading MNIST dataset...')
    mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
    # Creating data indices for training and validation splits:
    dataset_size = len(mnist)
    indices = list(range(dataset_size))
    split = 10000
    np.random.seed(0)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_mnist, val_mnist = torch.utils.data.random_split(mnist, [dataset_size-split,split])

    # Creating PT data samplers and loaders:
    weights_train = torch.ones(len(mnist))
    weights_test = torch.ones(len(mnist))
    weights_train[val_mnist.indices] = 0
    weights_test[train_mnist.indices] = 0
    counts = torch.zeros(10)
    for i in range(10):
        idx_label = mnist.targets[train_mnist.indices].eq(i)
        counts[i] = idx_label.sum()
    max = float(counts.max())
    sum_counts = float(counts.sum())
    for i in range(10):
        idx_label = mnist.targets[train_mnist.indices].eq(i).nonzero().squeeze()
        weights_train[train_mnist.indices[idx_label]] = (sum_counts / counts[i])

    train_sampler = SubsetRandomSampler(train_mnist.indices)
    valid_sampler = SubsetRandomSampler(val_mnist.indices)
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}
    loader = DataLoader(mnist,batch_size=FLAGS.batch_size,
            sampler=train_sampler,**kwargs)
    valid_loader = DataLoader(mnist,batch_size=FLAGS.batch_size,
            sampler=valid_sampler,**kwargs)
    monitor = torch.zeros(FLAGS.end_epoch - FLAGS.start_epoch,4)
    # initialize summary writer
    writer = SummaryWriter()

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('')
        print('Epoch #' + str(epoch) + '..........................................................................')
        elbo_epoch = 0
        term1_epoch = 0
        term2_epoch = 0
        term3_epoch = 0
        for it, (image_batch, labels_batch) in enumerate(loader):
            # set zero_grad for the optimizer
            auto_encoder_optimizer.zero_grad()

            X = image_batch.cuda().detach().clone()
            elbo, reconstruction_proba, style_kl_divergence_loss, class_kl_divergence_loss = process(FLAGS, X, labels_batch,
                                                encoder, decoder)
            (-elbo).backward()
            auto_encoder_optimizer.step()
            elbo_epoch += elbo
            term1_epoch += reconstruction_proba
            term2_epoch += style_kl_divergence_loss
            term3_epoch += class_kl_divergence_loss

        print("Elbo epoch %.2f" % (elbo_epoch / (it + 1)))
        print("Rec. Proba %.2f" % (term1_epoch / (it + 1)))
        print("KL style %.2f" % (term2_epoch / (it + 1)))
        print("KL content %.2f" % (term3_epoch / (it + 1)))
        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
            monitor[epoch,:]=eval(FLAGS, valid_loader, encoder, decoder)
            torch.save(encoder.state_dict(), os.path.join(savedir, FLAGS.encoder_save +'_e%d'%epoch))
            torch.save(decoder.state_dict(), os.path.join(savedir, FLAGS.decoder_save +'_e%d'%epoch))
            print("VAL elbo %.2f" % (monitor[epoch,0]))
            print("VAL Rec. Proba %.2f" % (monitor[epoch,1]))
            print("VAL KL style %.2f" % (monitor[epoch,2]))
            print("VAL KL content %.2f" % (monitor[epoch,3]))

            torch.save(monitor, os.path.join(savedir, 'monitor_e%d'%epoch))
