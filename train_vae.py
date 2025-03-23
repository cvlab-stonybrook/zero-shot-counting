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

def dict_to_list(attributes):
    all_attrs = []
    all_keys = builtin_meta.PASCAL_VOC_ALL_CATEGORIES[1]
    for key in all_keys:
      all_attrs.append(attributes[key].numpy())
    return np.array(all_attrs)


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


class FeatureDataset(DatasetFolder):
    """Face Landmarks dataset."""

    def __init__(self, feats, root = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.features, self.labels = self.convert_dict_to_list(feats)
        #self.features_aug, self.labels = self.convert_dict_to_list(feats_aug)
        self.features, self.labels = feats['feats'], feats['labels']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def generate_feats(feats_vae, attributes, output_file, label_list):
    res_dict = {}
    count=0
    ind_count = 500
    feats_vae.eval()
    z_dist = normal.Normal(0, 1)
    for label in label_list:
        attr = torch.from_numpy(attributes[label].astype(np.float32)).cuda()
        attr = attr.repeat(ind_count, 1)
        Z = z_dist.sample((ind_count, 512)).cuda()
        concat_feats = torch.cat((Z, attr), dim=1)
        feats = feats_vae.model(concat_feats)
        feats = feats_vae.relu(feats_vae.bn1(feats))
        res_dict[label] = feats.data.cpu().numpy()
    np.save(output_file, res_dict)
  
def train_vae(feature_loader, feats_vae, attributes):
    optimizer = torch.optim.Adam(feats_vae.parameters(), lr=0.001)
    #for ep in range(10):
    for ep in range(100):
      loss_recon_all = 0
      loss_kl_all = 0
      for idx, (data, label) in enumerate(feature_loader):
        data = data.cuda()
        #weight = weight.cuda() / torch.sum(weight)
        #attr = np.array([attributes[l].numpy() for l in (label)])
        attr = attributes[label].astype(np.float32)
        attr = torch.from_numpy(attr).cuda()
        mu, logvar, recon_feats = feats_vae(data, attr)
        recon_loss = ((recon_feats - data)**2).mean(1)
        recon_loss = torch.mean(recon_loss)
        #kl_loss = -0.5*torch.sum(1+logvar-logvar.exp()-mu.pow(2)) / data.shape[0]
        kl_loss = (1+logvar-logvar.exp()-mu.pow(2)).sum(1)
        kl_loss = -0.5*torch.mean(kl_loss)
        L_vae = recon_loss+kl_loss*0.01
        optimizer.zero_grad()
        L_vae.backward()
        optimizer.step()
        loss_recon_all += recon_loss.item()
        loss_kl_all += kl_loss.item()
      print('Ep: %d   Recon Loss: %f   KL Loss: %f'%(ep, loss_recon_all/(idx+1), loss_kl_all/(idx+1)))
    return feats_vae



if __name__ == '__main__':
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    torch.cuda.manual_seed(666)

    parser = argparse.ArgumentParser(
        description='few-shot Evaluation script')
    parser.add_argument('--save_dir', default='.', type=str, help='Directory to save the result csv')
    params = parser.parse_args()
    out_file = os.path.join(params.save_dir, 'train_data_ori.npy')
    cl_data_file = np.load(out_file, allow_pickle=True).item()
    feature_dataset = FeatureDataset(cl_data_file)
    feature_loader = torch.utils.data.DataLoader(feature_dataset, shuffle=True, pin_memory=True, drop_last=False, batch_size=256)
    attributes = np.load('./fsc_attr.npy')
    feats_vae = FeatsVAE(1024, 512).cuda()
    feats_vae = train_vae(feature_loader, feats_vae, attributes)
    vae_file = os.path.join(params.save_dir, 'fsc_vae_feats.npy')
    generate_feats(feats_vae, attributes, vae_file, range(147))
