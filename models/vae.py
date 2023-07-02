
import torch
import argparse
import numpy as np
from torch.autograd import Variable
from torchvision.datasets.folder import DatasetFolder
from torch.distributions import uniform, normal
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time
import pdb
import yaml
import pickle
class FeatsVAE(nn.Module):
    def __init__(self, x_dim, latent_dim):
        super(FeatsVAE, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(x_dim+latent_dim, 4096),
            #nn.LeakyReLU(),
            #nn.Linear(4096, 4096),
            nn.LeakyReLU())
        self.linear_mu =  nn.Sequential(
            nn.Linear(4096, latent_dim),
            nn.ReLU())
        self.linear_logvar =  nn.Sequential(
            nn.Linear(4096, latent_dim),
            nn.ReLU())
        self.model = nn.Sequential(
            nn.Linear(2*latent_dim, 4096),
            nn.LeakyReLU(),
            #nn.Linear(4096, 4096),
            #nn.LeakyReLU(),
            nn.Linear(4096, x_dim),
            #nn.Sigmoid(),
        )
        self.bn1 = nn.BatchNorm1d(x_dim)
        self.relu = nn.ReLU(inplace=True)
        self.z_dist = normal.Normal(0, 1)
        self.init_weights()


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  
        eps = torch.randn_like(std)
        # remove abnormal points
        return mu + eps*std

    def init_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Linear):
              m.weight.data.normal_(0, 0.02)
              m.bias.data.normal_(0, 0.02)

    def forward(self, x, attr):
        x = torch.cat((x, attr), dim=1)
        x = self.linear(x)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        latent_feats = self.reparameterize(mu, logvar)
        #Z = self.z_dist.sample(attr.shape).cuda() 
        concat_feats = torch.cat((latent_feats, attr), dim=1)
        recon_feats = self.model(concat_feats)
        recon_feats = self.relu(self.bn1(recon_feats))
        return mu, logvar, recon_feats
