import argparse, textwrap
import datetime

import os
import torchvision.transforms as tt
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import time
import json
import subprocess, atexit
import tqdm
import math
import tarfile
import warnings
import pickle
import random

import numpy as np

from tensorboardX import SummaryWriter

from torchvision.models import vgg16


from lib import datasets as db
from lib import extractor as ex
from lib import handcraft_extractor as he
from lib import metric as mt
from lib import postprocessing as pp
from lib import utils as u


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='Random seed number')
    parser.add_argument('--port', type=int, default=6006, help='Tensorboard port')
    parser.add_argument('--image_size', type=int, nargs='+', default=(256,256), help='Resize image size')
    parser.add_argument('--normalize_mean', type=float, nargs='+', default=(0.5,0.5,0.5), help='Mean in normalize')
    parser.add_argument('--normalize_std', type=float, nargs='+', default=(0.5,0.5,0.5), help='Std in normalize')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle train dataset')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers to load train dataset')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--save_interval', type=int, default=1, help='Save interval num when training ')
    parser.add_argument('--save_root', type=str, default="./jobs/capstone2020/", help='Save_root')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Load checkpoint') 
    parser.add_argument('--save_folder', '--sf', default=None, help='Set save folder name')
    parser.add_argument('--valid_interval', type=int, default=1, help='Validation interval')

    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--test', action='store_true', help='Inference model')
    parser.add_argument('--valid', action='store_true', help='Validation model')
    parser.add_argument('--valid_sample', action='store_true', help='Validation sampling model')
    parser.add_argument('--qualitative', action='store_true', help='Qualitative mode')

    parser.add_argument('--cluster', type=int, default=64, help='Set clustering number in NetVLAD')
    parser.add_argument('--pca', action='store_true', help='Use pca for dimension reduction')
    parser.add_argument('--pca_dim',  type=int, help='Reduced dimension using pca')
    parser.add_argument('--tuple', action='store_true', help='Use tuple (a,p,n) in training')
    parser.add_argument('--neg_num', type=int, default=1, help='The number of negative samples in training')
    parser.add_argument('--cen_crop',  action='store_true', help='Use center-crop')
    parser.add_argument('--topk',  type=int, default=1, help='Evaluation top k')

    parser.add_argument('--db_save',  type=str, default=None, help='Save database pkl')
    parser.add_argument('--db_load',  type=str, default=None, help='Load saved database pkl')

    parser.add_argument('--pose_estimation',  action='store_true', help='Estimate accurate pose')
    parser.add_argument('--pose_pointcloud_load',  action='store_true', help='Using saved point cloud')
    parser.add_argument('--pose_covisibility',  type=int, default=0, help='Get 2D-3D pair adjacent (n) frame')
    parser.add_argument('--pose_noniter',  action='store_true', help='Stop iterative estimation, if pnp can not estimate pose, it would immediately use top1 pose')
    parser.add_argument('--pose_cupy',  action='store_true', help='Using cupy in calculating points')
    parser.add_argument('--pose_timechecker',  action='store_true', help='Time check in pose estimation')

    parser.add_argument('--lmr_score',  type=float, default=0, help='Reranking LMR score')

    parser.add_argument('--topk_load',  type=str, default=None, help='Load topk npy')
    parser.add_argument('--topk_save',  type=str, default=None, help='Save topk npy')

    parser.add_argument('--rerank', type=int,  help='Select rerank for post processing : \n \
        0 - Local match rerank \n \
        1 - PnP rerank \n \
        ')

    parser.add_argument('--pose_ld', type=int,  help='Select local descriptor to use in the pose filter : \n \
        0 - SIFT \n \
        1 - rootSIFT \n \
        2 - D2 SS \n \
        3 - D2 MS \n \
        4 - SuperGlue \n \
        ')

    parser.add_argument('--positive_selection', type=int,  help='Positive selection method  : \n \
        0 - Thresholding \n \
        1 - Sequential \n \
        ')

    parser.add_argument('--dataset', type=int, default=0, help='Select dataset : \n \
        0 - NaverML_indoor b1\n \
        1 - NaverML_indoor 1f\n \
        ')
    parser.add_argument('--optimizer', type=int, help='Select optimizer : \n \
        0 - Adam \n \
        ')
    parser.add_argument('--scheduler', type=int, help='Select scheduler : \n \
        0 - CosineAnnealingLR \n \
        1 - StepLR \n \
        ')
    parser.add_argument('--extractor', type=int, help='Select feature extractor : \n \
        0 - D2_NetVLAD \n \
        1 - Pitts_NetVLAD \n \
        2 - APGeM \n \
        3 - APGeM_LM18 \n \
        4 - Ensemble(APGeM + APGeM_LM18 + D2_NetVLAD) \n \
        5 - Ensemble(APGeM + APGeM_LM18 + Pitts_NetVLAD) \n \
        6 - Ensemble(APGeM + APGeM_LM18) \n \
        7 - Ensemble(APGeM + D2_NetVLAD) \n \
        8 - Ensemble(APGeM + Pitts_NetVLAD) \n \
        9 - Ensemble(APGeM_LM18 + D2_NetVLAD) \n \
        10 - Ensemble(APGeM_LM18 + Pitts_NetVLAD) \n \
        ')
    parser.add_argument('--handcraft', type=int, help='Select handcraft algorithm : \n \
        0 - SIFT+VLAD \n \
        1 - rootSIFT+VLAD \n \
        ')
    parser.add_argument('--searching', type=int, default=0, help='Select searching alg : \n \
        0 - kNearestNeighbors \n \
        1 - kNearestNeighbors_GPU \n \
        2 - Diffusion \n \
        ')
    parser.add_argument('--metric', type=int, default=0, help='Select metric : \n \
        0 - LocDegThreshMetric \n \
        ')
    args = parser.parse_args()
    return args

@u.timer
def save_args(args, save_path):
    with open(os.path.join(save_path,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def arg_name_converter(args, argname, argval):

    if argname == "pose_ld":
        if argval == 0:
            return "SIFT"
        elif argval == 1:
            return "rootSIFT"
        elif argval == 2:
            return "D2 SS"
        elif argval == 3:
            return "D2 MS"
        elif argval == 4:
            return "SuperGlue"
        else:
            if args.pose_estimation is True:
                raise ValueError('Invalid arg pose_ld!')
            else:
                return "None"

    elif argname == "rerank":
        if argval == 0:
            return "Local match rerank"
        elif argval == 1:
            return "PnP rerank"
        elif argval == None:
            return "None"
        else:
            raise ValueError('Invalid arg rerank!')

    elif argname == "positive_selection":
        if argval == 0:
            return "Thresholding"
        elif argval == 1:
            return "Sequential"
        else:
            raise ValueError('Invalid arg positive_selection!')

    elif argname == "dataset":
        if argval == 0:
            return "NaverML_indoor b1"
        elif argval == 1:
            return "NaverML_indoor 1f"
        else:
            raise ValueError('Invalid arg dataset!')

    elif argname == "optimizer":
        if argval == 0:
            return "Adam"
        elif argval == None:
            return "None"
        else:
            raise ValueError('Invalid arg optimizer!')
    
    elif argname == "scheduler":
        if argval == 0:
            return "CosineAnnealingLR"
        elif argval == 1:
            return "StepLR"
        elif argval == None:
            return "None"
        else:
            raise ValueError('Invalid arg scheduler!')
 
    elif argname == "extractor":
        if argval == 0:
            return "D2_NetVLAD"
        elif argval == 1:
            return "Pitts_NetVLAD"
        elif argval == 2:
            return "APGeM"
        elif argval == 3:
            return "APGeM_LM18"
        elif argval == 4:
            return "Ensemble(APGeM + APGeM_LM18 + D2_NetVLAD)"
        elif argval == 5:
            return "Ensemble(APGeM + APGeM_LM18 + Pitts_NetVLAD)"
        elif argval == 6:
            return "Ensemble(APGeM + APGeM_LM18)"
        elif argval == 7:
            return "Ensemble(APGeM + D2_NetVLAD)"
        elif argval == 8:
            return "Ensemble(APGeM + Pitts_NetVLAD)"
        elif argval == 9:
            return "Ensemble(APGeM_LM18 + D2_NetVLAD)"
        elif argval == 10:
            return "Ensemble(APGeM_LM18 + Pitts_NetVLAD)"
        elif argval == 11:
            return "Ensemble(D2_NetVLAD + Pitts_NetVLAD)"
        elif argval == None:
            return "None"
        else:
            raise ValueError('Invalid arg extractor!')

    elif argname == "handcraft":
        if argval == 0:
            return "SIFT+VLAD"
        elif argval == 1:
            return "rootSIFT+VLAD"
        elif argval == None:
            return "None"

    elif argname == "searching":
        if argval == 0:
            return "kNearestNeighbor"
        elif argval == 1:
            return "kNearestNeighbors_GPU"
        elif argval == 2:
            return "Diffusion"
    
    elif argname == "metric":
        if argval == 0:
            return "LocDegThreshMetric"

    else:
        return str(argval)

@u.timer
def run_tensorboard( save_root, port=6006 ):

    tb_path = os.path.join( save_root, 'tensorboardX' )
    if not os.path.exists(tb_path):     os.makedirs(tb_path)
    pid = subprocess.Popen( ['tensorboard', '--logdir', tb_path, '--host', '0.0.0.0', '--port', str(port)] )    
    
    def cleanup():
    	pid.kill()

    atexit.register( cleanup )


@u.timer
def load_data(args):

    train_dataset = None
    valid_dataset = None
    test_dataset = None
    index_dataset = None

    train_loader = None
    valid_loader = None
    test_loader = None
    index_loader = None

    tlist_train = []
    tlist_train.append(db.Resize(args.image_size))
    if args.handcraft is not None:
        tlist_train.append(db.Grayscale())
    if args.cen_crop is True:
        tlist_train.append(db.CenterCrop())
    if args.extractor is not None:
        tlist_train.append(db.Normalize(mean=tuple(args.normalize_mean), std=tuple(args.normalize_std)))
        tlist_train.append(db.ToTensor())
    tform_train = tt.Compose(tlist_train)

    tlist_infer = []
    tlist_infer.append(db.Resize(args.image_size))
    if args.handcraft is not None:
        tlist_infer.append(db.Grayscale())
    if args.extractor is not None:
        tlist_infer.append(db.Normalize(mean=tuple(args.normalize_mean), std=tuple(args.normalize_std)))
        tlist_infer.append(db.ToTensor())
    tform_infer = tt.Compose(tlist_infer)


    if args.dataset == 0:
        if (args.valid is False) and (args.valid_sample is False) and (args.test is False):

            train_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                                "./NaverML_indoor/b1/train/csv/v2/train_all/train_b1.csv", \
                                                transform = tform_train, \
                                                use_tuple=args.tuple, \
                                                sel_pos=args.positive_selection, \
                                                neg_num=args.neg_num)
            
            train_loader = DataLoader(train_dataset, \
                                    batch_size=args.batch, \
                                    shuffle=args.shuffle, \
                                    num_workers=args.workers, \
                                    collate_fn=train_dataset.collate_func)
        
        elif args.test is True:
            test_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/test/", \
                                                "./NaverML_indoor/b1/test/csv/test_b1.csv", \
                                                transform = tform_infer)
            
            test_loader = DataLoader(test_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=test_dataset.collate_func)
            
            index_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                                "./NaverML_indoor/b1/train/csv/v2/train_all/train_b1.csv", \
                                                transform = tform_infer)
            
            index_loader = DataLoader(index_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=index_dataset.collate_func)
        
        else:
            train_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                                "./NaverML_indoor/b1/train/csv/v2/train_val/train_b1.csv", \
                                                transform = tform_train, \
                                                use_tuple=args.tuple, \
                                                sel_pos=args.positive_selection, \
                                                neg_num=args.neg_num)
            
            train_loader = DataLoader(train_dataset, \
                                    batch_size=args.batch, \
                                    shuffle=args.shuffle, \
                                    num_workers=args.workers, \
                                    collate_fn=train_dataset.collate_func)
 
            if (args.valid is True) and (args.valid_sample is False):
                valid_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                                    "./NaverML_indoor/b1/train/csv/v2/train_val/val_b1.csv", \
                                                    transform = tform_infer)
            elif (args.valid is False) and (args.valid_sample is True):
                valid_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                                    "./NaverML_indoor/b1/train/csv/v2/train_val/val_b1_sample10.csv", \
                                                    transform = tform_infer)
            else:
                raise ValueError('Select Valid or Valid sample !')

            valid_loader = DataLoader(valid_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=valid_dataset.collate_func)

            index_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                                "./NaverML_indoor/b1/train/csv/v2/train_val/train_b1.csv", \
                                                transform = tform_infer)
            
            index_loader = DataLoader(index_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=index_dataset.collate_func)
    elif args.dataset == 1:
        if (args.valid is False) and (args.valid_sample is False) and (args.test is False):

            train_dataset = db.Load_NMLC_indoor("./NaverML_indoor/1f/train/", \
                                                "./NaverML_indoor/1f/train/csv/v2/train_all/train_1f.csv", \
                                                transform = tform_train, \
                                                use_tuple=args.tuple, \
                                                sel_pos=args.positive_selection, \
                                                neg_num=args.neg_num)
            
            train_loader = DataLoader(train_dataset, \
                                    batch_size=args.batch, \
                                    shuffle=args.shuffle, \
                                    num_workers=args.workers, \
                                    collate_fn=train_dataset.collate_func)
        elif args.test is True:
            test_dataset = db.Load_NMLC_indoor("./NaverML_indoor/1f/test/", \
                                                "./NaverML_indoor/1f/test/csv/test_1f.csv", \
                                                transform = tform_infer)
            
            test_loader = DataLoader(test_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=test_dataset.collate_func)
            
            index_dataset = db.Load_NMLC_indoor("./NaverML_indoor/1f/train/", \
                                                "./NaverML_indoor/1f/train/csv/v2/train_all/train_1f.csv", \
                                                transform = tform_infer)
            
            index_loader = DataLoader(index_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=index_dataset.collate_func)
        else:
            train_dataset = db.Load_NMLC_indoor("./NaverML_indoor/1f/train/", \
                                                "./NaverML_indoor/1f/train/csv/v2/train_val/train_1f.csv", \
                                                transform = tform_train, \
                                                use_tuple=args.tuple, \
                                                sel_pos=args.positive_selection, \
                                                neg_num=args.neg_num)
            
            train_loader = DataLoader(train_dataset, \
                                    batch_size=args.batch, \
                                    shuffle=args.shuffle, \
                                    num_workers=args.workers, \
                                    collate_fn=train_dataset.collate_func)


            if (args.valid is True) and (args.valid_sample is False):
                valid_dataset = db.Load_NMLC_indoor("./NaverML_indoor/1f/train/", \
                                                    "./NaverML_indoor/1f/train/csv/v2/train_val/val_1f.csv", \
                                                    transform = tform_infer)
            elif (args.valid is False) and (args.valid_sample is True):
                valid_dataset = db.Load_NMLC_indoor("./NaverML_indoor/1f/train/", \
                                                    "./NaverML_indoor/1f/train/csv/v2/train_val/val_1f_sample10.csv", \
                                                    transform = tform_infer)
            else:
                raise ValueError('Select Valid or Valid sample !')
            
            valid_loader = DataLoader(valid_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=valid_dataset.collate_func)

            index_dataset = db.Load_NMLC_indoor("./NaverML_indoor/1f/train/", \
                                                "./NaverML_indoor/1f/train/csv/v2/train_val/train_1f.csv", \
                                                transform = tform_infer)
            
            index_loader = DataLoader(index_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=index_dataset.collate_func)

    return train_dataset,valid_dataset,index_dataset,test_dataset, \
        train_loader,valid_loader,index_loader,test_loader
@u.timer
def extractor(args):
    if args.extractor == 0:
        d2 = ex.D2Net(model_file="./arxiv/d2_tf.pth").dense_feature_extraction
        netvlad = ex.NetVLAD(num_clusters=args.cluster, dim=512, alpha=1.0)
        ext = ex.EmbedNet(d2, netvlad).cuda()

    elif args.extractor == 1:
        encoder = vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        for l in layers[:-5]: 
            for p in l.parameters():
                p.requires_grad = False
        base_model = nn.Sequential(*layers)
        dim = list(base_model.parameters())[-1].shape[0]
        netvlad = ex.NetVLAD(num_clusters=args.cluster, dim=dim, alpha=1.0)
        ext = ex.EmbedNet(base_model, netvlad).cuda()
        
        pittsburgh = torch.load("./arxiv/Pittsburgh_NetVLAD.pth.tar")
        pittsburgh_dict = pittsburgh['state_dict']

        ext_dict = ext.state_dict()
        for (key_t, val_t), (key_s, val_s) in zip(ext_dict.items(), pittsburgh_dict.items()):
            if key_t.split('.')[1]!=key_s.split('.')[1]:
                raise ValueError("Please check pittsburgh weight!")
            ext_dict[key_t] = val_s
        ext.load_state_dict(ext_dict)
        
    elif args.extractor == 2:
        ext = ex.create_model("resnet101_rmac", \
                              pretrained="./arxiv/Resnet-101-AP-GeM.pt", \
                              without_fc=False).cuda()

    elif args.extractor == 3:
        ext = ex.create_model("resnet101_rmac", \
                              pretrained="./arxiv/Resnet101-AP-GeM-LM18.pt", \
                              without_fc=False).cuda()

    elif args.extractor == 4:

        apgem = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet-101-AP-GeM.pt",without_fc=False).cuda()
        apgem_lm18 = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet101-AP-GeM-LM18.pt",without_fc=False).cuda()

        ckpt = torch.load("./arxiv/D2_NetVLAD_trainO_epoch024.pth.tar") # cluster 64
        d2_netvlad = ckpt['model']

        ext = ex.Ensemble3(apgem, apgem_lm18, d2_netvlad, is_backbone1=False, is_backbone2=False, is_backbone3=False, dim=[1024, 1024, 4096]).cuda()

    elif args.extractor == 5:

        apgem = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet-101-AP-GeM.pt",without_fc=False).cuda()
        apgem_lm18 = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet101-AP-GeM-LM18.pt",without_fc=False).cuda()

        ckpt = torch.load("./arxiv/Pitts_NetVLAD_trainO_epoch024.pth.tar") # cluster 64
        pitts_netvlad = ckpt['model']

        ext = ex.Ensemble3(apgem, apgem_lm18, pitts_netvlad, is_backbone1=False, is_backbone2=False, is_backbone3=False, dim=[1024, 1024, 4096]).cuda()
        
    elif args.extractor == 6:

        apgem = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet-101-AP-GeM.pt",without_fc=False).cuda()
        apgem_lm18 = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet101-AP-GeM-LM18.pt",without_fc=False).cuda()
        ext = ex.Ensemble2(apgem, apgem_lm18, is_backbone1=False, is_backbone2=False, dim=[1024, 1024]).cuda()
    
    elif args.extractor == 7:

        apgem = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet-101-AP-GeM.pt",without_fc=False).cuda()
        ckpt = torch.load("./arxiv/D2_NetVLAD_trainO_epoch024.pth.tar") # cluster 64
        d2_netvlad = ckpt['model']
        ext = ex.Ensemble2(apgem, d2_netvlad, is_backbone1=False, is_backbone2=False, dim=[1024, 4096]).cuda()
    
    elif args.extractor == 8:

        apgem = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet-101-AP-GeM.pt",without_fc=False).cuda()
        ckpt = torch.load("./arxiv/Pitts_NetVLAD_trainO_epoch024.pth.tar") # cluster 64
        pitts_netvlad = ckpt['model']
        ext = ex.Ensemble2(apgem, pitts_netvlad, is_backbone1=False, is_backbone2=False, dim=[1024, 4096]).cuda()

    elif args.extractor == 9:

        apgem_lm18 = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet101-AP-GeM-LM18.pt",without_fc=False).cuda()
        ckpt = torch.load("./arxiv/D2_NetVLAD_trainO_epoch024.pth.tar") # cluster 64
        d2_netvlad = ckpt['model']
        ext = ex.Ensemble2(apgem_lm18, d2_netvlad, is_backbone1=False, is_backbone2=False, dim=[1024, 4096]).cuda()
    
    elif args.extractor == 10:

        apgem_lm18 = ex.create_model("resnet101_rmac",pretrained="./arxiv/Resnet101-AP-GeM-LM18.pt",without_fc=False).cuda()
        ckpt = torch.load("./arxiv/Pitts_NetVLAD_trainO_epoch024.pth.tar") # cluster 64
        pitts_netvlad = ckpt['model']
        ext = ex.Ensemble2(apgem_lm18, pitts_netvlad, is_backbone1=False, is_backbone2=False, dim=[1024, 4096]).cuda()
    
    elif args.extractor == 11:

        ckpt = torch.load("./arxiv/D2_NetVLAD_trainO_epoch024.pth.tar") # cluster 64
        d2_netvlad = ckpt['model']
        ckpt = torch.load("./arxiv/Pitts_NetVLAD_trainO_epoch024.pth.tar") # cluster 64
        pitts_netvlad = ckpt['model']
        ext = ex.Ensemble2(d2_netvlad, pitts_netvlad, is_backbone1=False, is_backbone2=False, dim=[4096, 4096]).cuda()

    return ext

@u.timer
def criterion(args):
    if True:
        crt = ex.HardTripletLoss(margin=0.1, neg_num=args.neg_num)
    return crt

@u.timer
def handcraft_extractor(args):
    if args.handcraft == 0:
        ext = he.VLAD(ld = "sift")
    elif args.handcraft == 1:
        ext = he.VLAD(ld = "root_sift")
    return ext

@u.timer
def get_optimizer(args, model):
    if args.optimizer == 0:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    return optim

@u.timer
def get_scheduler(args, optim):
    if args.scheduler == 0:
        sche = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=30)
    elif args.scheduler == 1:
        sche = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    return sche

@u.timer
def train_handcraft(args, train_loader, valid_loader, index_loader, valid_dataset, index_dataset, save_root, writer):
    ext = handcraft_extractor(args)
    start_epoch = 0
    
    if args.ckpt_path is not None:
        ext.load(args.ckpt_path)
    else:

        batch_time = u.AverageMeter()
        data_time = u.AverageMeter()
        start = time.time()
        pbar = tqdm.tqdm(enumerate(train_loader), desc="Extract local descriptor!")

        if args.train is True:
            for batch_i, data in pbar:

                data_time.update(time.time() - start)
                start = time.time()

                ext.extract_ld(data)
                
                batch_time.update(time.time() - start)
                start = time.time()

                state_msg = (
                    'Data time: {:0.5f}; Batch time: {:0.5f};'.format(data_time.avg, batch_time.avg)
                )

                pbar.set_description(state_msg)
            
            ext.build_voca(k=args.cluster)
            ext.extract_vlad()
            filename = os.path.join(save_root, 'ckpt', 'checkpoint.pkl')
            ext.save(filename)

    
    if (args.valid is True) or (args.valid_sample is True):
        pbar = tqdm.tqdm(enumerate(valid_loader), desc="Extract query descriptor!")
        for batch_i, data in pbar:

            ext.extract_vlad_query(data)

        indexdb, validdb = ext.get_data()

        ldm = mt.LocDegThreshMetric(args, indexdb, validdb, index_dataset, valid_dataset, 0, os.path.join(save_root, "result"))
    return 

@u.timer
def train(args, train_loader, valid_loader, index_loader, valid_dataset, index_dataset, save_root, writer):
    ext = extractor(args)
    crt = criterion(args)
    
    optim = get_optimizer(args, ext)
    sche = get_scheduler(args, optim)

    start_epoch = 0

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path)
        ext = ckpt['model']
        optim = ckpt['optimizer']
        start_epoch = ckpt['epoch']+1
  
    for epoch in range(start_epoch,args.epochs):
        ext.train()
        sche.step()


        batch_time = u.AverageMeter()
        data_time = u.AverageMeter()
        losses = u.AverageMeter()
        
        start = time.time()

        pbar = tqdm.tqdm(enumerate(train_loader), desc="Epoch : %d"%epoch)


        size_all = len(train_loader)
        interv = math.floor(size_all/args.save_interval)
        sub_p=0

        if args.train is True:
            for batch_i, data in pbar:

                image = data['image'].cuda()
                label = data['label'].cuda()
                
                data_time.update(time.time() - start)
                start = time.time()

                output = ext(image)
                if output.dim()==1:
                    output = output.unsqueeze(0)

                loss = crt(output, label, args.tuple, args.batch)

                optim.zero_grad()
                loss.backward()
                optim.step()
            
                losses.update(loss.item())
                batch_time.update(time.time() - start)
                start = time.time()

                writer.add_scalars('train/loss', {'loss': losses.avg}, global_step=epoch*len(train_loader)+batch_i )

                state_msg = (
                    'Epoch: {:4d}; Loss: {:0.5f}; Data time: {:0.5f}; Batch time: {:0.5f};'.format(epoch, losses.avg, data_time.avg, batch_time.avg)
                )

                pbar.set_description(state_msg)

                if ((batch_i+1)%interv==0) or (size_all==batch_i+1):
                    
                    state = {'epoch': epoch,
                            'loss': losses.avg,
                            'model': ext,
                            'optimizer': optim}
                    filename = os.path.join(save_root, 'ckpt', 'checkpoint_subset{:03d}_epoch{:03d}.pth.tar'.format(sub_p, epoch))
                    torch.save(state, filename)
                    sub_p+=1



        if ((args.valid is True) or (args.valid_sample is True)) and ((epoch+1)%args.valid_interval==0):
            
            
            if (args.db_load is not None):
                with open(args.db_load, "rb") as a_file:
                    indexdb = pickle.load(a_file)
            else:
                # #index
                indexdb = make_inferDBandPredict(args, index_loader, ext, epoch, tp='index')

            #valid
            validdb = make_inferDBandPredict(args, valid_loader, ext, epoch, tp='valid')

            

            if args.db_load is None:
                if (args.extractor>=4):
                    indexdb['feat'], validdb['feat'] = ext.postprocessing(indexdb['feat'], validdb['feat'])

                if args.pca is True:
                    pca = pp.PCAwhitening(pca_dim=args.pca_dim, pca_whitening=True)
                    indexdb['feat'] = pca.fit_transform(indexdb['feat'])      
                    validdb['feat'] = pca.transform(validdb['feat']) 

            if (args.db_save is not None):
                if os.path.isfile(args.db_save) is True:
                    os.remove(args.db_save)
                a_file = open(args.db_save, "wb")
                pickle.dump(indexdb, a_file)
                a_file.close()


            ldm = mt.LocDegThreshMetric(args, indexdb, validdb, index_dataset, valid_dataset, epoch, os.path.join(save_root, "result"))

            if args.train is False:
                return

            if args.qualitative:
                return

            for key, value in ldm.items():
                writer.add_scalars('valid/top'+str(args.topk)+"_"+key, {key: value}, global_step=epoch)

    return 

@u.timer
def test(args, test_loader, index_loader, test_dataset, index_dataset, save_root):
    ext = extractor(args)

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path)
        ext = ckpt['model']
    
    if (args.db_load is not None):
        with open(args.db_load, "rb") as a_file:
            indexdb = pickle.load(a_file)
    else:
        # #index
        indexdb = make_inferDBandPredict(args, index_loader, ext, 0, tp='index')

    #valid
    testdb = make_inferDBandPredict(args, test_loader, ext, 0, tp='test')   
    
    if args.db_load is None:
        if (args.extractor>=4):
            indexdb['feat'], testdb['feat'] = ext.postprocessing(indexdb['feat'], testdb['feat'])

        if (args.pca is True):
            pca = pp.PCAwhitening(pca_dim=args.pca_dim, pca_whitening=True)
            indexdb['feat'] = pca.fit_transform(indexdb['feat'])      
            testdb['feat'] = pca.transform(testdb['feat'])     

    if (args.db_save is not None):
        if os.path.isfile(args.db_save) is True:
            os.remove(args.db_save)
        a_file = open(args.db_save, "wb")
        pickle.dump(indexdb, a_file)
        a_file.close()
    


    mt.LocDegThreshMetric(args, indexdb, testdb, index_dataset, test_dataset, 0, os.path.join(save_root, "result"))

    return 


def make_inferDBandPredict(args, loader, ext, epoch, tp='index'):
    ext.eval()

    pbar = tqdm.tqdm(enumerate(loader), desc="Make "+tp+" db; Epoch : %d"%epoch)

    datadb={}

    featdb = []
    labeldb = []
    posedb = []
    indexdb = []
    
    with torch.no_grad():
        for batch_i, data in pbar:

            image = data['image'].cuda()
            label = data['label'].cuda()
            pose = data['pose']
            index = data['index']
            
            output = ext(image)
            if output.dim()==1:
                output = output.unsqueeze(0)

            featdb.extend(output.detach().cpu().numpy())
            labeldb.extend(label.detach().cpu().numpy())
            indexdb.extend(index)
            posedb.extend(pose.numpy().tolist())

    datadb.update({'feat' : np.asarray(featdb)})
    datadb.update({'label' : np.asarray(labeldb)})
    datadb.update({'pose' : np.asarray(posedb)})
    datadb.update({'index' : np.asarray(indexdb)})

    del featdb, labeldb, posedb

    
    
    return datadb


@u.timer
def params_show(args):
    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz=KST).strftime('%Y%m%d-%H%M%S')

    params = {"time":now}
    params.update(vars(args))

    u.logger.info('{} {} {}'.format('-'*30, "Params List", '-'*30))
    
    for key, val in params.items():
        
        val = arg_name_converter(args, key, val)

        line = key + " = " + val

        before_size = len(key)
        after_size = len(val)

        u.logger.info('{} {} {}'.format('|'+' '*(33-before_size), line, ' '*(33-after_size)+'|'))
    
    u.logger.info('{}{}{}'.format('', '-'*73, ''))

    if args.save_folder is not None:
        save_root = '{}{}_{}'.format(args.save_root, 
                                        now, 
                                        args.save_folder)
    elif args.extractor is None:
        save_root = '{}{}_{}_{}'.format(args.save_root, 
                                    now, 
                                    arg_name_converter(args, "handcraft", args.handcraft), 
                                    arg_name_converter(args, "dataset", args.dataset))
    else:
        save_root = '{}{}_{}_{}'.format(args.save_root, 
                                    now, 
                                    arg_name_converter(args, "extractor", args.extractor), 
                                    arg_name_converter(args, "dataset", args.dataset))

    if not os.path.exists(save_root): os.makedirs(save_root)
    if not os.path.exists(os.path.join(save_root, "ckpt")): os.makedirs(os.path.join(save_root, "ckpt"))
    if not os.path.exists(os.path.join(save_root, "args")): os.makedirs(os.path.join(save_root, "args"))
    if not os.path.exists(os.path.join(save_root, "result")): os.makedirs(os.path.join(save_root, "result"))
    if not os.path.exists(os.path.join(save_root, "source")): os.makedirs(os.path.join(save_root, "source"))

      
    tar = tarfile.open( os.path.join(save_root, "source", 'sources.tar'), 'w' )
    tar.add( 'lib' )    
    tar.add(__file__)
    tar.close()

    return save_root



def main():
    args = get_args()

    random.seed(args.seed)
    # np.random.seed(args.seed)
    # because positive random selection
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if (args.extractor is not None) and (args.handcraft is not None):
        u.logger.error('Specify --extractor or --handcraft\n')
        raise ValueError('Specify --extractor or --handcraft')
    elif (args.extractor is None) and (args.handcraft is None):
        u.logger.error('Specify --extractor or --handcraft\n')
        raise ValueError('Specify --extractor or --handcraft')
    elif (args.pca is True) and (isinstance(args.pca_dim, int) is False):
        u.logger.error('Please set pca dimension\n')
        raise ValueError('Please set pca dimension')
    else:
        pass
    
    if (not args.train) and (not args.test) and (not args.valid) and (not args.valid_sample):
        u.logger.error('Specify --train or --show\n')
        raise ValueError('Specify --train or --show')
    else:
        save_root = params_show(args)

    save_args(args, os.path.join(save_root, "args"))

    run_tensorboard(save_root, args.port)

    _,valid_dataset,index_dataset,test_dataset,\
    train_loader,valid_loader,index_loader,test_loader = load_data(args)

    writer = SummaryWriter(os.path.join(save_root, 'tensorboardX'))

    if (args.train) or (args.valid) or (args.valid_sample):
        if args.extractor is not None:
            train(args, train_loader, valid_loader, index_loader, valid_dataset, index_dataset, save_root, writer)
        elif args.handcraft is not None:
            train_handcraft(args, train_loader, valid_loader, index_loader, valid_dataset, index_dataset, save_root, writer)

    if args.test:
        test(args, test_loader, index_loader, test_dataset, index_dataset, save_root)
    return True
if __name__ == '__main__':
    main()