"""
sorce:
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
"""

import argparse
import os
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator

os.makedirs("images", exist_ok=True)

hiperparameters = {
    'n_epochs': 200,
    'batch_size': 128,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'n_cpu': 8,
    'latent_dim': 100,
    'img_size': 28,
    'channels': 1,
    'n_critic': 5,
    'clip_value': 0.01,
    'sample_interval': 1000
}

cuda = True if torch.cuda.is_available() else False

img_shape = (hiperparameters['channels'], hiperparameters['img_size'], hiperparameters['img_size'])

# Initialize generator and discriminator
generator = Generator(img_shape = img_shape, latent_dim = hiperparameters['latent_dim'])
discriminator = Discriminator(img_shape = img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=hiperparameters['batch_size'],
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=hiperparameters['lr'])
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=hiperparameters['lr'])

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(hiperparameters['n_epochs']):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], hiperparameters['latent_dim']))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-hiperparameters['clip_value'], hiperparameters['clip_value'])

        # Train the generator every n_critic iterations
        if i % hiperparameters['n_critic'] == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, hiperparameters['n_epochs'], batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % hiperparameters['sample_interval'] == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1