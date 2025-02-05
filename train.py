import time
import argparse
from progress.bar import IncrementalBar

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import MyDataset
from model.generator import UnetGenerator
from model.discriminator import ConditionalDiscriminator
from model.criterion import GeneratorLoss, DiscriminatorLoss
from model.utils import Logger, initialize_weights


def menu():
    parser = argparse.ArgumentParser(prog = 'top', description='Train Pix2Pix')
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
    args = parser.parse_args()
    return args
    

def train_epoch(dataloader, epoch, args, generator, discriminator, g_criterion, d_criterion, g_optimizer, d_optimizer):
    bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(dataloader))
    for x, real in dataloader:
        x = x.to(device)
        real = real.to(device)

        # Generator`s loss
        fake = generator(x)
        fake_pred = discriminator(fake, x)
        g_loss = g_criterion(fake, real, fake_pred)

        # Discriminator`s loss
        fake = generator(x).detach()
        fake_pred = discriminator(fake, x)
        real_pred = discriminator(real, x)
        d_loss = d_criterion(fake_pred, real_pred)

        # Generator`s params update
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Discriminator`s params update
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # add batch losses
        ge_loss += g_loss.item()
        de_loss += d_loss.item()
        bar.next()
    bar.finish()  

    return ge_loss, de_loss

if __name__ == '__main__':
    args = menu()
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    transforms = v2.Compose([v2.Resize((256,256)),
                             v2.ToTensor(),
                             v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                             ])
    # Gen and Disc models
    print('Defining models!')
    generator = UnetGenerator().to(device)
    discriminator = ConditionalDiscriminator().to(device)
    # optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # loss functions
    g_criterion = GeneratorLoss(alpha=100)
    d_criterion = DiscriminatorLoss()

    # dataset and dataloader
    dataset = MyDataset(root=args.dataset, transform=transforms, download=True, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print('Start of training process!')
    logger = Logger(filename=args.dataset)
    for epoch in range(args.epochs):
        ge_loss=0.
        de_loss=0.
        start = time.time()

        ge_loss, de_loss = train_epoch(dataloader, epoch, args, generator, discriminator, g_criterion, d_criterion, g_optimizer, d_optimizer)

        # obttain per epoch losses
        g_loss = ge_loss/len(dataloader)
        d_loss = de_loss/len(dataloader)
        # count timeframe
        end = time.time()
        tm = (end - start)
        logger.add_scalar('generator_loss', g_loss, epoch+1)
        logger.add_scalar('discriminator_loss', d_loss, epoch+1)
        logger.save_weights(generator.state_dict(), 'generator')
        logger.save_weights(discriminator.state_dict(), 'discriminator')
        print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch+1, args.epochs, g_loss, d_loss, tm))
    logger.close()
    print('End of training process!')
