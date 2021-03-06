import argparse
import torch
import torch.optim as optim
import numpy as np
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import os, time


class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, img_dim: tuple, hidden_size=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.img_dim = img_dim  # (C, H, W)
        self.hidden_size = hidden_size
        # Layers
        self.latent_fc = nn.Linear(self.latent_dim, self.hidden_size)
        self.latent_bn = nn.BatchNorm1d(self.hidden_size)
        self.label_fc = nn.Linear(self.label_dim, self.hidden_size)
        self.label_bn = nn.BatchNorm1d(self.hidden_size)
        self.fc1 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.bn1 = nn.BatchNorm1d(2 * self.hidden_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, 4 * self.hidden_size)
        self.bn2 = nn.BatchNorm1d(4 * self.hidden_size)
        self.fc3 = nn.Linear(4 * self.hidden_size, int(np.prod(self.img_dim)))

        self._initialize(0., 0.02)

    def _initialize(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, z, label):
        z = nn.functional.relu(self.latent_bn(self.latent_fc(z)))
        label = nn.functional.relu(self.label_bn(self.label_fc(label)))
        latent = torch.cat([z, label], dim=-1)
        out = nn.functional.relu(self.bn1(self.fc1(latent)))
        out = nn.functional.relu(self.bn2(self.fc2(out)))
        out = torch.tanh(self.fc3(out))
        out = torch.reshape(out, (-1,) + self.img_dim)
        return out

    def sample(self, z, label):
        '''
        :param z: latent z with size (batch_size, self.latent_dim)
        :param label: one hot labels with size (batch_size, self.label_dim)
        :return: generated images with size (batch_size, C, H, W), each value is in range [0, 1]
        '''
        with torch.no_grad():
            # TODO: generated images for further evaluation.
            out = self.forward(z,label)
            out = (out + 1) / 2 
        return out


class Discriminator(nn.Module):
    def __init__(self, label_dim, img_dim, hidden_size=256):
        super(Discriminator, self).__init__()
        self.label_dim = label_dim
        self.img_dim = img_dim
        self.hidden_size = hidden_size
        # Layers
        self.img_fc = nn.Linear(int(np.prod(self.img_dim)), 4 * self.hidden_size)
        self.label_fc = nn.Linear(self.label_dim, 4 * self.hidden_size)
        self.fc2 = nn.Linear(8 * self.hidden_size, 2 * self.hidden_size)
        self.bn2 = nn.BatchNorm1d(2 * self.hidden_size)
        self.fc3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.bn3 = nn.BatchNorm1d(self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, 1)

        self._initialize(0., 0.02)

    def _initialize(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, img, label):
        img = torch.flatten(img, start_dim=1, end_dim=-1)
        x = nn.functional.leaky_relu(self.img_fc(img), 0.2)
        y = nn.functional.leaky_relu(self.label_fc(label), 0.2)
        x = torch.cat([x, y], dim=-1)
        x = nn.functional.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = nn.functional.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x)).squeeze(dim=-1)
        return x


#########################
####  DO NOT MODIFY  ####
def generate_samples(generator, n_samples_per_class, device):
    generator.eval()
    latent = torch.randn((n_samples_per_class * 10, generator.latent_dim), device=device)
    label = torch.eye(generator.label_dim, dtype=torch.float, device=device).repeat(n_samples_per_class, 1)
    imgs = generator.sample(latent, label).cpu()
    label = torch.argmax(label, dim=-1).cpu()
    samples = dict(imgs=imgs, labels=label)
    return samples
##########################

def plot(gen, n_samples_per_class, device, name):
    gen.eval()
    latent = torch.randn((n_samples_per_class * 10, gen.latent_dim), device=device)
    label = torch.eye(gen.label_dim, dtype=torch.float, device=device).repeat(n_samples_per_class, 1)
    imgs = gen.sample(latent, label).cpu()
    # print(imgs.shape)
    m = np.zeros((28*10,28*n_samples_per_class))
    for i in range(imgs.shape[0]):
        x = i // 10
        y = i % 10
        m[x*28:(x+1)*28,y*28:(y+1)*28] = imgs[i,0]
    plt.imsave(name,m,cmap='gray')

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # Load dataset
    if args.dataset == "mnist":
        dataset = MNIST(root="../data",
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5],[0.5])]),  # You can tweak it.
                        train=not args.eval)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True)
    else:
        raise NotImplementedError

    # Configure
    logdir = args.logdir if args.logdir is not None else "/tmp/ebm_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(logdir, exist_ok=True)

    label_dim = 10
    img_dim = (1, 28, 28)
    latent_dim = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim, label_dim, img_dim)
    discriminator = Discriminator(label_dim, img_dim)
    generator.to(device)
    discriminator.to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    if not args.eval:
        # TODO: training, logging, saving
        rd = torch.distributions.Normal(0, 1)
        for epoch in range(args.num_epochs):
            generator.train()
            for it, data in enumerate(dataloader):
                img0, label = data
                ones = torch.sparse.torch.eye(label_dim)
                label =  ones.index_select(0,label)
                img0, label = img0.to(device), label.to(device)
                
                latent = rd.sample((img0.shape[0],latent_dim)).to(device)
                img1 = generator(latent, label)
                output0 = discriminator(img0, label)
                output1 = discriminator(img1, label)
                
                loss0 = -torch.mean(torch.log(output0+0.01))
                loss1 = -torch.mean(torch.log(1.01-output1))
                loss_d = loss0 + loss1
                d_optimizer.zero_grad()
                loss_d.backward()
                d_optimizer.step()

                latent = rd.sample((img0.shape[0],latent_dim)).to(device)
                img1 = generator(latent, label)
                output1 = discriminator(img1, label)
                loss_g = -torch.mean(torch.log(output1+0.01))

                g_optimizer.zero_grad()
                loss_g.backward()
                g_optimizer.step()

                # loss_g = -loss1
                # if loss0 < loss1:
                #     d_optimizer.zero_grad()
                #     loss_d.backward()
                #     d_optimizer.step()
                # else:
                #     g_optimizer.zero_grad()
                #     loss_g.backward()
                #     g_optimizer.step()

                if it % 100 == 0:
                    print('epoch: %d, iter: %d, loss0: %f, loss1: %f'%(epoch,it,loss0.item(),loss1.item()))
            plot(generator, 10, device, '%d.jpg'%epoch)
            if epoch % 20 == 19:
                torch.save(generator,'./model/generator-%d.npy'%(epoch))
                torch.save(discriminator,'./model/discriminator-%d.npy'%(epoch))


    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path, map_location=device)
        generator = checkpoint
        # discriminator.load_state_dict(checkpoint['d'])
        # generator.load_state_dict(checkpoint['g'])
        # d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        # g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        # Generate samples for evaluation.
        samples = generate_samples(generator, 1000, device)
        torch.save(samples, "gan_generated_samples.pt")


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
