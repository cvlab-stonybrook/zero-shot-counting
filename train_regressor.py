import argparse
import datetime
import random
import time
from pathlib import Path
import pdb
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from torchvision.utils import save_image
from config import cfg
import util.misc as utils
from loss import get_loss
from FSC147_dataset import build_dataset, batch_collate_fn
from engine import evaluate, train_one_epoch, visualization
from models import build_model
from models.regressor import get_regressor
from torchvision.datasets.folder import DatasetFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class PatchDataset(DatasetFolder):
    """Face Landmarks dataset."""

    def __init__(self, feats_dir, root = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.feats_dir = feats_dir

    def __len__(self):
        return len(os.listdir(self.feats_dir))

    def __getitem__(self, idx):
        tmp = np.load(os.path.join(self.feats_dir, '%d.npy'%idx), allow_pickle=True).item()
        corr_feats = tmp['corr_feats']
        feats = tmp['feats']
        errs = tmp['errs']
        tmp_feats = np.repeat(feats[np.newaxis,:,:,:], corr_feats.shape[0], axis=0)
        concat_feats = np.concatenate((tmp_feats, corr_feats[:,np.newaxis,:,:]), 1)
        return torch.from_numpy(concat_feats), torch.from_numpy(errs)

def train_err(args):
    print(args)
    device = torch.device(cfg.TRAIN.device)
    # fix the seed for reproducibility
    seed = cfg.TRAIN.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    regressor = get_regressor(cfg)
    regressor.to(device)
    regressor.train()
    optimizer = torch.optim.AdamW(regressor.parameters(), lr=1e-5, weight_decay=5e-4)

    # define dataset
    patch_dataset = PatchDataset('patch_feats')
    patch_loader = torch.utils.data.DataLoader(patch_dataset, shuffle=True, pin_memory=True, drop_last=False, batch_size=1)
    loss_avg = 0
    for ep in range(5):
      for idx, (patch_feats, errs) in enumerate(patch_loader):
        errs = errs.to(torch.float32).cuda().view(-1)
        pred = regressor(patch_feats[0].cuda()).view(-1)
        loss = ((errs - pred)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
        if idx % 200 == 0:
          print('Ep: %d, Idx: %d, Loss: %.4f'%(ep, idx, (loss_avg/200)))
          loss_avg = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Class Agnostic Object Counting in PyTorch"
    )
    parser.add_argument(
        "--cfg",
        default="config/bmnet+_fsc147.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    #cfg.merge_from_list(args.opts)

    cfg.DIR.output_dir = os.path.join(cfg.DIR.snapshot, cfg.DIR.exp)
    if not os.path.exists(cfg.DIR.output_dir):
        os.mkdir(cfg.DIR.output_dir)

    cfg.TRAIN.resume = os.path.join(cfg.DIR.output_dir, cfg.TRAIN.resume)
    cfg.VAL.resume = os.path.join(cfg.DIR.output_dir, cfg.VAL.resume)

    with open(os.path.join(cfg.DIR.output_dir, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    train_err(cfg)
