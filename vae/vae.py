import argparse
import torch
from torch._C import device
import torch.optim as optim
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os, time

import matplotlib.pyplot as plt
import math

class CVAE(nn.Module):
    def __init__(self, img_size, label_size, latent_size, hidden_size=256):
        super(CVAE, self).__init__()
        self.img_size = img_size  # (C, H, W)
        self.label_size = label_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        # Encoder.
        '''
        img   -> fc  ->                   -> fc -> mean    
                        concat -> encoder                  -> z
        label -> fc  ->                   -> fc -> logstd 
        '''
        self.enc_img_fc = nn.Linear(int(np.prod(self.img_size)), self.hidden_size)
        self.enc_label_fc = nn.Linear(self.label_size, self.hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
        )
        self.z_mean = nn.Linear(2 * self.hidden_size, self.latent_size)
        self.z_logstd = nn.Linear(2 * self.hidden_size, self.latent_size)
        # Decoder.
        '''
        latent -> fc ->
                         concat -> decoder -> reconstruction
        label  -> fc ->
        '''
        self.dec_latent_fc = nn.Linear(self.latent_size, self.hidden_size)
        self.dec_label_fc = nn.Linear(self.label_size, self.hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
            nn.Linear(2 * self.hidden_size, int(np.prod(self.img_size))), nn.Sigmoid(),
        )
        # TODO: assume the distribution of reconstructed images is a Gaussian distibution. Write the log_std here.
        self.recon_logstd = 0
        self.prior = torch.distributions.Normal(0, 1)

    def preEncode(self, batch_img, batch_label):
        batch_img = batch_img.reshape(batch_img.shape[0],-1)
        a = self.enc_img_fc(batch_img)
        b = self.enc_label_fc(batch_label)
        c = torch.cat((a,b),1)
        c = F.relu(c)
        c = self.encoder(c)
        mean = self.z_mean(c)
        logstd = self.z_logstd(c)
        return mean, logstd
    
    def getZ(self, mean, logstd):
        z = self.prior.sample(mean.shape).to(mean.device)
        z = mean + z * torch.exp(logstd)
        return z

    def encode(self, batch_img, batch_label):
        '''
        :param batch_img: a tensor of shape (batch_size, C, H, W)
        :param batch_label: a tensor of shape (batch_size, self.label_size)
        :return: a batch of latent code of shape (batch_size, self.latent_size)
        '''
        # TODO: compute latent z from images and labels
        mean, logstd = self.preEncode(batch_img, batch_label)
        z = self.getZ(mean, logstd)
        return z

    def decode(self, batch_latent, batch_label):
        '''
        :param batch_latent: a tensor of shape (batch_size, self.latent_size)
        :param batch_label: a tensor of shape (batch_size, self.label_size)
        :return: reconstructed results
        '''
        a = self.dec_latent_fc(batch_latent)
        b = self.dec_label_fc(batch_label)
        c = torch.cat((a,b),1)
        c = F.relu(c)
        c = self.decoder(c)
        c = c.reshape([-1]+list(self.img_size))
        return c

    def sample(self, batch_latent, batch_label):
        '''
        :param batch_latent: a tensor of size (batch_size, self.latent_size)
        :param batch_label: a tensor of size (batch_size, self.label_dim)
        :return: a tensor of size (batch_size, C, H, W), each value is in range [0, 1]
        '''
        with torch.no_grad():
            # TODO: get samples from the decoder.
            y = self.decode(batch_latent, batch_label)
            # y += self.prior.sample(y.shape).to(y.device)
            y = torch.clip(y,0,1)
        return y


#########################
####  DO NOT MODIFY  ####
def generate_samples(cvae, n_samples_per_class, device):
    cvae.eval()
    latent = torch.randn((n_samples_per_class * 10, cvae.latent_size), device=device)
    label = torch.eye(cvae.label_size, dtype=torch.float, device=device).repeat(n_samples_per_class, 1)
    imgs = cvae.sample(latent, label).cpu()
    label = torch.argmax(label, dim=-1).cpu()
    samples = dict(imgs=imgs, labels=label)
    return samples
#########################

def plot(cvae, n_samples_per_class, device, name):
    cvae.eval()
    latent = torch.randn((n_samples_per_class * 10, cvae.latent_size), device=device)
    label = torch.eye(cvae.label_size, dtype=torch.float, device=device).repeat(n_samples_per_class, 1)
    imgs = cvae.sample(latent, label).cpu()
    # print(imgs.shape)
    m = np.zeros((28*10,28*n_samples_per_class))
    for i in range(imgs.shape[0]):
        x = i // 10
        y = i % 10
        m[x*28:(x+1)*28,y*28:(y+1)*28] = imgs[i,0]
    plt.imsave(name,m,cmap='gray')

def KL(u1,s1):
    # print(u1.shape)
    # print(s1.shape)
    return torch.sum(-s1 + (torch.exp(2 * s1) + u1 ** 2) / 2,1)

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # Load dataset
    if args.dataset == "mnist":
        dataset = MNIST(root="../data",
                        transform=transforms.ToTensor(),  # TODO: you may want to tweak this
                        train=not args.eval,download=True)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True)
    else:
        raise NotImplementedError

    # Configure
    logdir = args.logdir if args.logdir is not None else "/tmp/cvae_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(logdir, exist_ok=True)

    label_dim = 10
    img_dim = (1, 28, 28)
    latent_dim = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae = CVAE(img_dim, label_dim, latent_dim)
    cvae.to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=args.lr)

    if not args.eval:
        for name, param in cvae.named_parameters():
            print(name, param.shape)
        prior = torch.distributions.Normal(0, 1)
        for epoch in range(args.num_epochs):
            # TODO: Training, logging, saving, visualization, etc.
            cvae.train()
            for it, data in enumerate(dataloader):
                imgs, labels = data
                imgs = imgs.to(device)
                ones = torch.sparse.torch.eye(label_dim)
                labels =  ones.index_select(0,labels)
                labels = labels.to(device)
                cvae.zero_grad()
                u, s = cvae.preEncode(imgs,labels)
                z = cvae.getZ(u, s)
                mean = cvae.decode(z,labels)
                # print(((imgs - mean) ** 2).shape)
                loss1 = torch.sum((imgs - mean) ** 2  ,(1,2,3)) / (2 * math.exp(cvae.recon_logstd))
                loss2 = KL(u, s)
                # print(loss.shape)
                loss1 = torch.sum(loss1) / loss1.shape[0]
                loss2 = torch.sum(loss2) / loss2.shape[0]
                loss = loss1 + loss2
                if it % 100 == 0:
                    print('epoch: %d, iter: %d, loss1: %f, loss2: %f'%(epoch,it,loss1.item(),loss2.item()))
                loss.backward()
                optimizer.step()
            plot(cvae, 10, device, '%d.jpg'%epoch)
            
    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path, map_location=device)
        cvae.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cvae.eval()
        samples = generate_samples(cvae, 1000, device)
        torch.save(samples, "vae_generated_samples.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["mnist"], default="mnist")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
