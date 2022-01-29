from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from misc import weights_init, save_individual_images
from models import Generator
from models import Discriminator
from models import mahalanobis_loss


from itertools import cycle

import pretty_errors

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
#dataroot = "office-31-intact//amazon//images//"
#dataroot_tgt = "office-31-intact//dslr//images//"
#dataroot = "celebs//"

# Batch size during training
batch_size = 128

image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 1500
lr = 0.0002
beta1 = 0.5
ngpu = 4

src = "amazon"
tgt = "dslr"

#src_obj = tgt_obj = "ring_binder"

print('available objs are {}'.format(os.listdir("office-31//amazon//images//")))


for obj in os.listdir("office-31//amazon//images//"):

    if os.path.isfile('generated_images//' + obj + '_images.png'):
        print('skipping {} as its already translated.'.format(obj))
        continue
    else:
        print('starting to create intermediate of {} between the domains {} and {}'.format(obj, src, tgt))

        src_obj = obj
        tgt_obj = obj

        dataroot = "office-31//amazon//images//" + src_obj
        dataroot_tgt = "office-31//dslr//images//" + tgt_obj

        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        dataset_tgt = dset.ImageFolder(root=dataroot_tgt,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True)
        dataloader_tgt = torch.utils.data.DataLoader(dataset_tgt, batch_size=batch_size,
                                                 shuffle=True)
        print('finished loading the datasets.')
        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Create the generator
        netG = Generator(ngpu).to(device)
        netG.apply(weights_init)
        # Create the Discriminator
        netD = Discriminator(ngpu).to(device)
        netD.apply(weights_init)

        # Create the Discriminator
        netD_tgt = Discriminator(ngpu).to(device)
        netD_tgt.apply(weights_init)

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(256, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerD_tgt = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            # for i, data in enumerate(dataloader, 0):
            for i, (data, data_tgt) in enumerate(zip(dataloader, cycle(dataloader_tgt)), 0):


                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (1.5) Update D_tgt network: maximize log(D_tgt(x)) + log(1 - D_tgt(G(z)))
                ###########################
                ## Train with all-real batch
                netD_tgt.zero_grad()
                # Format batch
                real_cpu = data_tgt[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = netD_tgt(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real_tgt = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real_tgt.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD_tgt(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake_tgt = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake_tgt.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real_tgt + errD_fake_tgt
                # Update D
                optimizerD_tgt.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                m_loss = mahalanobis_loss(real_cpu.cpu(), netG(noise).cpu())

                output = netD(fake).view(-1)
                output_tgt = netD_tgt(fake).view(-1)
                # Calculate G's loss based on this output
                errG = (criterion(output, label)+criterion(output_tgt, label))/2
                #errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

        # Grab a batch of real images from the dataloader
        real_batch_src = next(iter(dataloader))
        real_batch_tgt = next(iter(dataloader_tgt))
        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,3,1)
        plt.axis("off")
        plt.title("Real Source Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch_src[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        #Plot the real images
        plt.subplot(1,3,2)
        plt.axis("off")
        plt.title("Real Target Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch_tgt[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,3,3)
        plt.axis("off")
        plt.title("Fake/Transferable Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        plt.show()
        plt.savefig('generated_images//' + src_obj + '_images.png')

        fake = netG(fixed_noise).detach().cpu()
        save_individual_images(src + '_' + tgt + '_fake_dataset//'+src_obj+'//', fake)
