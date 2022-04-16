from __future__ import print_function
#%matplotlib inline
import argparse
import os
import shutil
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from misc import weights_init, save_individual_images, get_particular_class, get_same_index, AddGaussianNoise, compute_gradient_penalty
from models import Generator
from models import Discriminator
from models import mahalanobis_loss


from itertools import cycle
import pretty_errors

parser = argparse.ArgumentParser()
parser.add_argument("--which_class", type=int, default=0, help="generate fake samples of this class.")
opt = parser.parse_args()
print(opt)


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
num_epochs = 200
lr = 5e-5
lr_g = 1e-7
beta1 = 0.5
ngpu = 4
# Loss weight for gradient penalty
lambda_gp = 10

category = opt.which_class

print('generating fake data for label {}'.format(category))

dataroot_amazon = "datasets//office-31-intact//amazon//images//"
dataroot_dslr = "datasets//office-31-intact//dslr//images//"
dataroot_webcam = "datasets//office-31-intact//webcam//images//"

transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #AddGaussianNoise(0., 1.)
])

dataset_amazon = datasets.ImageFolder(root=dataroot_amazon,
                           transform=transform)
dataset_dslr = datasets.ImageFolder(root=dataroot_dslr,
                           transform=transform)
dataset_webcam = datasets.ImageFolder(root=dataroot_webcam,
                           transform=transform)


train_set_amazon, test_set_amazon = torch.utils.data.random_split(dataset_amazon, [int(len(dataset_amazon)*0.8), len(dataset_amazon)-int(len(dataset_amazon)*0.8)])
train_set_dslr, test_set_dslr = torch.utils.data.random_split(dataset_dslr, [int(len(dataset_dslr)*0.8), len(dataset_dslr)-int(len(dataset_dslr)*0.8)])
train_set_webcam, test_set_webcam = torch.utils.data.random_split(dataset_webcam, [int(len(dataset_webcam)*0.8), len(dataset_webcam)-int(len(dataset_webcam)*0.8)])

dataset = train_set_amazon
dataset_tgt = train_set_dslr

dataset.targets = torch.tensor(dataset.targets)
idx = get_same_index(dataset.targets, category)
dataset.targets= dataset.targets[idx]
dataset.data = dataset.data[idx]


dataset_tgt.labels = torch.tensor(dataset_tgt.labels)
idx = get_same_index(dataset_tgt.labels, category)
dataset_tgt.labels = dataset_tgt.labels[idx]
dataset_tgt.data = dataset_tgt.data[idx]


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
criterion_b = nn.MSELoss()
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(512, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD_tgt = optim.Adam(netD_tgt.parameters(), lr=lr_g, betas=(beta1, 0.999))
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
        output_real = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output_fake = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        #gradient_penalty = compute_gradient_penalty(netD, real_cpu.data, fake.data)
        # lambda_gp * gradient_penalty
        D_loss = -torch.mean(output_real) + torch.mean(output_fake)


        D_loss.backward()
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
        # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        G_loss = -torch.mean(output)
        # Calculate gradients for G
        G_loss.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     D_loss.item(), G_loss.item()))



        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                try:
                    shutil.rmtree('generated_images//office_31_amazon_to_webcam//'+str(category) + '//')
                except:
                    pass
                os.makedirs('generated_images//office_31_amazon_to_webcam//'+str(category) + '//')
                save_individual_images('generated_images//office_31_amazon_to_webcam//'+str(category) + '//', fake)
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
plt.savefig('generated_images//demo_images.png')
