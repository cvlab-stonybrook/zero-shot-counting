# ------------------------------------------------------------------------
# Training code for bilinear similarity network (BMNet and BMNet+)
# --cfg: path for configuration file
# ------------------------------------------------------------------------
import argparse
import datetime
import random
import time
import json
import copy
from pathlib import Path
import pdb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from torchvision.utils import save_image
from config import cfg 
import util.misc as utils
from loss import get_loss
from FSC147_dataset import build_dataset, batch_collate_fn, random_aug_boxes, get_image_classes
from engine import evaluate, train_one_epoch, visualization 
from models import build_model
from torch.distributions import uniform, normal
from models.regressor import get_regressor
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from models.vae import FeatsVAE
import pickle5 as pickle

def select_feats_vae_imgnet(vae_feature, patches, model):
    patch_feature = model.backbone(patches)
    tmp_patch = model.EPF_extractor.avgpool(patch_feature).flatten(1) 
    dist = (tmp_patch - vae_feature)**2
    dist = dist.sum(1)
    return dist.argsort()[:10]

def select_feats_vae(vae_feature, patches, model):
    patch_feature = model.backbone(patches)
    tmp_patch = model.EPF_extractor.avgpool(patch_feature).flatten(1) 
    dist = (tmp_patch - vae_feature)**2
    dist = dist.sum(1)
    return dist.argsort()[:100]

def prepare_data(img_path, anno):
    img = Image.open(img_path)
    w, h = img.size
    gtcount = len(anno['points'])
    boxes = np.array(anno['box_examples_coordinates'])
    boxes = random_aug_boxes(boxes, img.size[1], img.size[0])
    query_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    patches = []
    scale_embedding = [] 
    scale_number = 20
    for box in boxes:
        x1, y1 = box[0].astype(np.int32)
        x2, y2 = box[2].astype(np.int32)
        #x1,y1,x2,y2 = np.array(box).astype(np.int32)
        patch = img.crop((x1, y1, x2, y2))
        patches.append(query_transform(patch))
        scale = (x2 - x1) / w * 0.5 + (y2 -y1) / h * 0.5
        scale = scale // (0.5 / scale_number)
        scale = scale if scale < scale_number - 1 else scale_number - 1
        scale_embedding.append(0)
    patches = torch.stack(patches, dim=0)
    main_transform = transforms.Compose([transforms.Resize(size=384), \
                   transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    img = main_transform(img)
    return img.unsqueeze(0), patches.unsqueeze(0), torch.tensor(scale_embedding).unsqueeze(0).to(torch.int64), gtcount, boxes

def get_vae_embedding(attr_np):
    feats_vae = FeatsVAE(1024, 512).cuda()
    feats_vae.load_state_dict(torch.load('feats_vae.pth'))
    z_dist = normal.Normal(0, 1)
    ind_count = 500
    attr = torch.from_numpy(attr_np.astype(np.float32)).cuda()
    attr = attr.repeat(ind_count, 1)
    Z = z_dist.sample((ind_count, 512)).cuda()
    concat_feats = torch.cat((Z, attr), dim=1)
    feats = feats_vae.model(concat_feats)
    feats = feats_vae.relu(feats_vae.bn1(feats))
    return feats.cpu().mean(0)

def extract_corr_map(args):
    #print(args)
    device = torch.device(cfg.TRAIN.device)
    # fix the seed for reproducibility
    seed = cfg.TRAIN.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(cfg)
    model.to(device)
    model.eval()
   
    regressor = get_regressor(cfg)
    regressor.to(device)
    regressor.eval()
    regressor.load_state_dict(torch.load('regressor_model/regressor.pth'))    
 

    # define dataset
    output_dir = Path(cfg.DIR.output_dir)
    cls_dict = get_image_classes('./FSC147_384_V2/ImageClasses_FSC147.txt')
    cls_list = np.array(list(cls_dict.values()))
    cls_list = sorted(np.unique(cls_list))
    vae_feats = np.load(os.path.join(output_dir, 'fsc_vae_feats_tmp.npy'), allow_pickle=True).item()
    #vae_feats = vae_file['feats']
    #vae_labels = vae_file['labels']
    #vae_feats = np.load('./mean_proto_test.npy', allow_pickle=True).item()
    checkpoint = torch.load(cfg.VAL.resume, map_location='cpu')
    model_imgnet = copy.deepcopy(model)
    model.load_state_dict(checkpoint['model'])
    mae = 0
    mse = 0
    nae = 0
    sre = 0
    count_idx = 0
    loss_avg = 0
    errs_all = []
    #with open('FSC_multiclass_val_test_All_Boxes.pkl', 'rb') as pickle_file:
    #  annos = pickle.load(pickle_file)
    with open('FSC147_384_V2/annotation_FSC147_384.json', 'rb') as pickle_file:
      annos = json.load(pickle_file)
    imgnet_feats = np.load('imgnet_feats_clip.npy', allow_pickle=True).item()
    count_item = 0
    tmp_list = []
    train_list = [name.split('\t') for name in open('FSC147_384_V2/test.txt').read().splitlines()]
    for idxx, k in enumerate(train_list):
        img, patches1, scale_embedding, gtcount, boxes = prepare_data('./FSC147_384_V2/images_384_VarV2/%s'%k[0], annos[k[0]])
        img = img.to(device)
        scale_embedding = scale_embedding.to(device)
        patches = patches1.to(device)
        with torch.no_grad():
          ###################
          ori_features1 = model.backbone(img)
          ori_features = model.input_proj(ori_features1)
          ###################
          #vae_feature = get_vae_embedding(vae_feats[idx]).cuda().float().view(-1, 1024)
          #vae_feature = model.EPF_extractor.patch2query(vae_feature).view(1, vae_feature.shape[0], -1).permute(1, 0, 2).contiguous()
          #vae_feature = torch.from_numpy(np.array(vae_feats[idx]).mean(0)).cuda().float().view(-1, 1024)
          #vae_feature = model.EPF_extractor.patch2query(vae_feature).view(1, vae_feature.shape[0], -1).permute(1, 0, 2).contiguous()
          ###################
          img = F.interpolate(img, [384,384])
          features = model.backbone(img)
          features = model.input_proj(features)
          patches = patches.flatten(0, 1)
          cls = cls_dict[k[0]]
          label = cls_list.index(cls)
          patch_feature = model.backbone(patches) # obtain feature maps for exemplar patches
          #tmp_feature = ori_features1.flatten(2).permute(0, 2, 1)   
          #tmp_feature2 = torch.from_numpy(vae_feats[label].mean(0)).unsqueeze(0).cuda()
          #map_feature = torch.matmul(tmp_feature[0], tmp_feature2.permute(1,0))
          #save_feats = map_feature[:,0].reshape((ori_features.shape[-2:]))
          #save_image(save_feats.unsqueeze(0), 'img.png', normalize=True)
          #map_feature = torch.cdist(tmp_feature[0], tmp_feature2)[:,0]
          vae_feature = vae_feats[label]
          #vae_sel_idx = select_feats_vae_imgnet((vae_feature.mean(0)).to(device), patches, model_imgnet)
          vae_sel_idx = select_feats_vae_imgnet(torch.from_numpy(vae_feature.mean(0)).to(device), patches, model_imgnet)
          #if '6080' in k[0]:
          #    vae_sel_idx = range(48, 58)
          #vae_sel_idx = range(450)
          patch_feature2 = model.EPF_extractor(patch_feature[vae_sel_idx], scale_embedding[:, vae_sel_idx])
          if False:
            patch_feature2 = vae_feature
            patch_feature2 = model.EPF_extractor.patch2query(patch_feature2) \
                  .view(1, vae_feature.shape[0], -1) \
                  .permute(1, 0, 2) \
                  .contiguous()
          bs, batch_num_patches = scale_embedding.shape
          refined_feature, patch_feature2 = model.refiner(ori_features, patch_feature2)
          counting_feature, corr_map = model.matcher(refined_feature, patch_feature2)
          bs, c, h, w = refined_feature.shape
          feats_all = []
          if True:
            for m_idx in range(patch_feature2.shape[0]):
              counting_feature, corr_map = model.matcher(features, patch_feature2[[m_idx]])
              feats_all.append(counting_feature)
            counting_feature = torch.stack(feats_all).squeeze(1)
            #density_map = model.counter(counting_feature)
            #sel_idx = density_map.sum([1,2,3]).argmin()
            scores = regressor(counting_feature)
            sel_idx = scores.argsort(0)[:3]
            patch_feature3 = patch_feature2[sel_idx[:,0]]
            #patch_feature3 = patch_feature2[sel_idx].unsqueeze(0)
            #pdb.set_trace()
            counting_feature, corr_map = model.matcher(refined_feature, patch_feature3)
          density_map = model.counter(counting_feature)
          error = torch.abs(density_map.sum() - gtcount).item()
          errs_all.append(error)
          print('%s: gt: %d, err: %d'%(k[0], int(gtcount), int(error)))
          count_item += 1
          mae += error
          mse += error ** 2
          nae += error / gtcount
          sre += error ** 2 / gtcount
    mae = mae / count_item
    mse = mse / count_item
    nae = nae / count_item
    sre = sre / count_item
    mse = mse ** 0.5
    sre = sre ** 0.5
    print('MAE %.2f, MSE %.2f, NAE %.2f, SRE %.2f \n'%(mae, mse, nae, sre))
           
           

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

    extract_corr_map(cfg)
