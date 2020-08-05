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

import numpy as np

from tensorboardX import SummaryWriter

from torchvision.models import resnet18

from lib import datasets as db
from lib import extractor as ex
from lib import handcraft_extractor as he
from lib import metric as mt
from lib import postprocessing as pp
from lib import utils as u
import lib.extractor.RMAC as RMAC

from dirtorch.utils import common
from dirtorch.utils.common import tonumpy, matmul, pool
from sklearn.preprocessing import normalize

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='Random seed number')
    parser.add_argument('--port', type=int, default=6006, help='Tensorboard port')
    parser.add_argument('--image_size', type=int, nargs='+', default=(256,256), help='Resize image size')
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
    parser.add_argument('--qualitative', action='store_true', help='Qualitative mode')

    parser.add_argument('--pca', action='store_true', help='Use pca for dimension reduction')
    parser.add_argument('--pca_dim',  type=int, help='Reduced dimension using pca')
    parser.add_argument('--tuple', action='store_true', help='Use tuple (a,p,n) in training')
    parser.add_argument('--cen_crop',  action='store_true', help='Use center-crop')
    parser.add_argument('--topk',  type=int, default=1, help='Evaluation top k')
    parser.add_argument('--pose_filter', action='store_true', help='Filtering neighbors by using pose')
    parser.add_argument('--posek',  type=int, default=1, help='Evaluation filtering pose top k')
    parser.add_argument('--pose_ld_dense',  action='store_true', help='Dense status using pose local descriptor')
    parser.add_argument('--floor', type=str, default="b1",  help='Dense status using pose local descriptor')    
    parser.add_argument('--test_all',  action='store_true', help='Evaluation filtering pose top k')
    parser.add_argument('--Ensemble',  action='store_true', help='Ensemble')
    
    parser.add_argument('--pose_ld', type=int,  help='Select local descriptor to use in the pose filter : \n \
        0 - root SIFT \n \
        1 - SIFT \n \
        2 - D2 \n \
        ')
    
    parser.add_argument('--pose_matcher', type=int,  help='Select matcher to use in the pose filter : \n \
        0 - Brute-Force \n \
        1 - Flann-based \n \
        ')


    parser.add_argument('--dataset', type=int, default=0, help='Select dataset : \n \
        0 - NaverML_indoor \n \
        ')
    parser.add_argument('--optimizer', type=int, help='Select optimizer : \n \
        0 - Adam \n \
        1 - SGD \n \
        ')
    parser.add_argument('--scheduler', type=int, help='Select scheduler : \n \
        0 - CosineAnnealingLR \n \
        1 - StepLR \n \
        ')
    parser.add_argument('--extractor', type=int, help='Select feature extractor : \n \
        0 - D2+NetVLAD \n \
        1 - DELF \n \
        2 - res18+NetVLAD \n \
        3 - FishNet+NetVLAD \n \
        4 - RMaC+GeM \n \
        5 - RMaC+GeM18 \n \
        ')
    parser.add_argument('--handcraft', type=int, help='Select handcraft algorithm : \n \
        0 - SIFT+VLAD \n \
        1 - rootSIFT+VLAD \n \
        ')
    parser.add_argument('--searching', type=int, default=0, help='Select searching alg : \n \
        0 - kNearestNeighbors \n \
        ')
    parser.add_argument('--metric', type=int, default=0, help='Select metric : \n \
        0 - LocDegThreshMetric \n \
        1 - LocThreshMetric \n \
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
            return "root SIFT"
        elif argval == 1:
            return "SIFT"
        elif argval == 2:
            raise ValueError('Coming soon!')
            return "D2"
        else:
            if args.pose_filter is True:
                raise ValueError('Invalid arg pose_ld!')
            else:
                return "None"

    elif argname == "pose_matcher":
        if argval == 0:
            return "Brute-Force"
        elif argval == 1:
            return "Flann-based"
        else:
            if args.pose_filter is True:
                raise ValueError('Invalid arg pose_matcher!')
            else:
                return "None"

    elif argname == "dataset":
        if argval == 0:
            return "NaverML_indoor"
        else:
            raise ValueError('Invalid arg dataset!')

    elif argname == "optimizer":
        if argval == 0:
            return "Adam"
        elif argval == 1:
            return "SGD"
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
            return "D2+NetVLAD"
        elif argval == 2:
            return "res18+NetVLAD"
        elif argval == 3:
            return "FishNet+NetVLAD"
        elif argval == 4:
            return "RMaC+GeM"
        elif argval == 4:
            return "RMaC+GeMLM18" 
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
    
    elif argname == "metric":
        if argval == 0:
            return "LocDegThreshMetric"
        elif argval == 1:
            return "LocThreshMetric"
    
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
        tlist_train.append(db.Normalize())
        tlist_train.append(db.ToTensor())
    tform_train = tt.Compose(tlist_train)

    tlist_infer = []
    tlist_infer.append(db.Resize(args.image_size))
    if args.handcraft is not None:
        tlist_infer.append(db.Grayscale())
    if args.extractor is not None:
        tlist_infer.append(db.Normalize())
        tlist_infer.append(db.ToTensor())
    tform_infer = tt.Compose(tlist_infer)

    if args.dataset == 0:
        if (args.test is False):
            train_dataset = db.Load_NMLC_indoor("./NaverML_indoor/"+args.floor+"/train/", \
                                                "None", \
                                                transform = tform_train, \
                                                use_tuple=args.tuple)
            
            train_loader = DataLoader(train_dataset, \
                                    batch_size=args.batch, \
                                    shuffle=args.shuffle, \
                                    num_workers=args.workers, \
                                    collate_fn=train_dataset.collate_func)
        else:
            train_dataset = db.Load_NMLC_indoor("./NaverML_indoor/"+args.floor+"/train/", \
                                                "./NaverML_indoor/"+args.floor+"/train/csv/v2/train_val/train_"+args.floor+".csv", \
                                                transform = tform_train, \
                                                use_tuple=args.tuple)
            
            train_loader = DataLoader(train_dataset, \
                                    batch_size=args.batch, \
                                    shuffle=args.shuffle, \
                                    num_workers=args.workers, \
                                    collate_fn=train_dataset.collate_func)
            if args.test_all:
                valid_dataset = db.Load_NMLC_indoor("./NaverML_indoor/"+args.floor+"/test/", \
                                                    "./NaverML_indoor/"+args.floor+"/test/csv/test_"+args.floor+".csv", \
                                                    transform = tform_infer)
            else:
                valid_dataset = db.Load_NMLC_indoor("./NaverML_indoor/"+args.floor+"/train/", \
                                                    "./NaverML_indoor/"+args.floor+"/train/csv/v2/train_val/val_"+args.floor+".csv", \
                                                    transform = tform_infer)
            
            valid_loader = DataLoader(valid_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=valid_dataset.collate_func)
            if args.test_all:
                index_dataset = db.Load_NMLC_indoor("./NaverML_indoor/"+args.floor+"/train/", \
                                                    "./NaverML_indoor/"+args.floor+"/train/csv/v1/train_all/train_"+args.floor+".csv", \
                                                    transform = tform_infer)
            else:            
                index_dataset = db.Load_NMLC_indoor("./NaverML_indoor/"+args.floor+"/train/", \
                                                "./NaverML_indoor/"+args.floor+"/train/csv/v2/train_val/train_"+args.floor+".csv", \
                                                transform = tform_infer)
            
            index_loader = DataLoader(index_dataset, \
                                    batch_size=args.batch, \
                                    num_workers=args.workers, \
                                    collate_fn=index_dataset.collate_func)

    return train_dataset,valid_dataset,index_dataset,test_dataset, \
        train_loader,valid_loader,index_loader,test_loader
@u.timer
def extractor(args):
    if args.Ensemble:
        if args.ckpt_path:
            ext_3=torch.load(args.ckpt_path)
            ext3=ext_3["model"].cuda()
        ext = RMAC.create_model("resnet101_rmac",pretrained="./lib/extractor/RMAC/Resnet-101-AP-GeM.pt",without_fc=False) 
        ext = ext.cuda()
        ext2 = RMAC.create_model("resnet101_rmac",pretrained="./lib/extractor/RMAC/Resnet101-AP-GeM-LM18.pt",without_fc=False)
        ext2 = ext2.cuda()
        return ext,ext2,ext3
    else:

        if args.extractor == 0:
            d2 = ex.D2Net(model_file="./lib/extractor/D2Net/pretrained/d2_tf.pth").dense_feature_extraction
            netvlad = ex.NetVLAD(num_clusters=64, dim=512, alpha=1.0)
            ext = ex.EmbedNet(d2, netvlad).cuda()
            if args.ckpt_path:
                ext=torch.load(args.ckpt_path)
                ext=ext["model"].cuda() 
        elif args.extractor == 2:
            encoder = resnet18(pretrained=True)
            base_model = nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                encoder.maxpool,
                encoder.layer1,
                encoder.layer2,
                encoder.layer3,
                encoder.layer4)
            dim = list(base_model.parameters())[-1].shape[0]
            netvlad = ex.NetVLAD(num_clusters=64, dim=dim, alpha=1.0)
            ext = ex.EmbedNet(base_model, netvlad).cuda()
        elif args.extractor == 3:
            fishnet, dim = ex.fishnet150(layer=7)
            netvlad = ex.NetVLAD(num_clusters=64, dim=dim, alpha=1.0)
            ext = ex.EmbedNet(fishnet, netvlad).cuda()
        elif args.extractor==4:
            print("load ext")
            ext=RMAC.create_model("resnet101_rmac",pretrained="./lib/extractor/RMAC/Resnet-101-AP-GeM.pt",without_fc=False)
            ext = ext.cuda()
        elif args.extractor==5:
            print("load ext")
            ext=RMAC.create_model("resnet101_rmac",pretrained="./lib/extractor/RMAC/Resnet101-AP-GeM-LM18.pt",without_fc=False)
            ext = ext.cuda()
        return ext


@u.timer
def criterion(args):
    
    crt = ex.HardTripletLoss(margin=0.1)

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

    return sche

@u.timer
def train_handcraft(args, train_loader, valid_loader, index_loader, valid_dataset, index_dataset, save_root, writer):
    ext = handcraft_extractor(args)
    start_epoch = 0
    
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path)
        ext = ckpt['model']
    

    batch_time = u.AverageMeter()
    data_time = u.AverageMeter()
    start = time.time()
    pbar = tqdm.tqdm(enumerate(train_loader), desc="Extract local descriptor!")

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
    
    ext.build_voca()
    ext.extract_vlad()
    filename = os.path.join(save_root, 'ckpt', 'checkpoint.pkl')
    ext.save(filename)

    
    if args.valid is True:
        pbar = tqdm.tqdm(enumerate(valid_loader), desc="Extract query descriptor!")
        for batch_i, data in pbar:

            ext.extract_vlad_query(data)

        indexdb, validdb = ext.get_data()


        if args.dataset==2:
            if args.metric == 0:
                ldm = mt.LocDegThreshMetric(args, indexdb, validdb, index_dataset, valid_dataset, 0, os.path.join(save_root, "result"))
            elif args.metric == 1:
                ldm = mt.LocThreshMetric(args, indexdb, validdb, index_dataset, valid_dataset, 0, os.path.join(save_root, "result"))
            
            for key, value in ldm.items():
                print(key,value)
            
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
        start_epoch = ckpt['epoch']
    
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


        if (args.valid is True) and ((epoch+1)%args.valid_interval==0):

            # #DB
            indexdb,index_gem = make_inferDBandPredict(args, index_loader, ext, epoch, tp='index')
            
            #Query
            validdb,valid_gem = make_inferDBandPredict(args, valid_loader, ext, epoch, tp='valid')           
            
            pca = pp.PCAwhitening(pca_dim=4096, pca_whitening=True)
            name="only_VLAD"
            if not os.path.exists("pca_%s_%s.pkl"%(args.floor,name)):

                print("Start PCA fit")
                pca.fit(indexdb["feat"])
                print("Start PCA transform")
                pca.save(args.floor,name)
            else:
                print("Load PCA ")
                pca.load(args.floor,name)
                print("Start PCA transform")              
            whiten = {'whitenp': 0.25, 'whitenv': 4096, 'whitenm': 1.0}
            
            indexdb["feat"] = common.whiten_features(indexdb["feat"], pca.pca, **whiten)
            validdb["feat"] = common.whiten_features(validdb["feat"], pca.pca, **whiten)
            if index_gem.shape[0]!=0:

                indexdb["feat"]=normalize(indexdb["feat"],axis=1)
                validdb["feat"]=normalize(validdb["feat"],axis=1)
                
                indexdb["feat"]=np.hstack((indexdb["feat"],np.array(index_gem)))
                validdb["feat"]=np.hstack((validdb["feat"],np.array(valid_gem)))
                indexdb["feat"]=normalize(indexdb["feat"],axis=1)
                validdb["feat"]=normalize(validdb["feat"],axis=1)
            
            if args.dataset==0:
                if args.metric == 0:
                    ldm = mt.LocDegThreshMetric(args, indexdb, validdb, index_dataset, valid_dataset, epoch, os.path.join(save_root, "result"))
                elif args.metric == 1:
                    ldm = mt.LocThreshMetric(args, indexdb, validdb, index_dataset, valid_dataset, epoch, os.path.join(save_root, "result"))

                for key, value in ldm.items():
                    writer.add_scalars('valid/top'+str(args.topk)+"_"+key, {key: value}, global_step=epoch)
                    print(key,value)
    return 

@u.timer
def test(args, train_loader, valid_loader, index_loader, valid_dataset, index_dataset, save_root):
    ext=[]
    if args.Ensemble:
        ext=[]
        ext1,ext2,ext3 = extractor(args)
        ext.append(ext1)
        ext.append(ext2)
        ext.append(ext3)
    else:
        ext1 = extractor(args)
        ext.append(ext1)
    start_epoch = 0
    epoch=0
    if (args.valid is True) or (args.test is True):

        #DB
        indexdb,index_gem = make_inferDBandPredict(args, index_loader, ext, epoch, tp='index')
        
        #Query
        validdb,valid_gem = make_inferDBandPredict(args, valid_loader, ext, epoch, tp='valid')           
        
        pca = pp.PCAwhitening(pca_dim=4096, pca_whitening=True)
        name="only_VLAD"
        if not os.path.exists("pca_%s_%s.pkl"%(args.floor,name)):
            print("Start PCA fit")
            pca.fit(indexdb["feat"])
            print("Start PCA transform")
            pca.save(args.floor,name)
        else:
            print("Load PCA ")
            pca.load(args.floor,name)
            print("Start PCA transform")              

        whiten = {'whitenp': 0.25, 'whitenv': 4096, 'whitenm': 1.0}
        indexdb["feat"] = common.whiten_features(indexdb["feat"], pca.pca, **whiten)
        validdb["feat"] = common.whiten_features(validdb["feat"], pca.pca, **whiten)
        
        indexdb["feat"]=normalize(indexdb["feat"],axis=1)
        validdb["feat"]=normalize(validdb["feat"],axis=1)
        
        indexdb["feat"]=np.hstack((indexdb["feat"],np.array(index_gem)))
        validdb["feat"]=np.hstack((validdb["feat"],np.array(valid_gem)))
        
        indexdb["feat"]=normalize(indexdb["feat"],axis=1)
        validdb["feat"]=normalize(validdb["feat"],axis=1)
        
        if args.dataset==0:
            if args.metric == 0:
                mt.LocDegThreshMetric(args, indexdb, validdb, index_dataset, valid_dataset, epoch, os.path.join(save_root, "result"))
            elif args.metric == 1:
                mt.LocThreshMetric(args, indexdb, validdb, index_dataset, valid_dataset, epoch, os.path.join(save_root, "result"))

    return 


def make_inferDBandPredict(args, loader, exts, epoch, tp='index'):
    for ext in exts:
        ext.eval()

    pbar = tqdm.tqdm(enumerate(loader), desc="Make "+tp+" db; Epoch : %d"%epoch)

    datadb={}

    featdb = []
    featdb_gem = []
    
    labeldb = []
    posedb = []
    indexdb = []
    with torch.no_grad():
        for batch_i, data in pbar:


            image = data['image'].cuda()
            label = data['label'].cuda()
            pose = data['pose']
            index = data['index']

            if len(exts)==0:
                output = exts[0](image)
            else:    
                output = exts[0](image)
                output1 = exts[1](image)
                output2 = exts[2](image)                
                output_gem=torch.cat((output.unsqueeze(0),output1.unsqueeze(0)),dim=1)
                output=output2
                featdb_gem.extend(output_gem.detach().cpu().numpy().tolist())
            
            featdb.extend(output.detach().cpu().numpy().tolist())
            labeldb.extend(label.detach().cpu().numpy().tolist())
            indexdb.extend(index.tolist())
            if args.dataset==0:
                posedb.extend(pose.numpy().tolist())

    datadb.update({'feat' : featdb})
    datadb.update({'label' : labeldb})
    datadb.update({'pose' : posedb})
    datadb.update({'index' : indexdb})

    del featdb, labeldb, posedb
    
    return datadb ,featdb_gem


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
    if (args.extractor is not None) and (args.handcraft is not None):
        u.logger.error('Specify --extractor or --handcraft\n')
        raise ValueError('Specify --extractor or --handcraft')
    elif (args.extractor is None) and (args.handcraft is None):
        u.logger.error('Specify --extractor or --handcraft\n')
        raise ValueError('Specify --extractor or --handcraft')
    elif (args.topk == 1) and (args.pose_filter is True):
        u.logger.error('Pose filter support more than 1\n')
        raise ValueError('Pose filter support more than 1')
    elif (args.pca is True) and (isinstance(args.pca_dim, int) is False):
        u.logger.error('Please set pca dimension\n')
        raise ValueError('Please set pca dimension')
    elif (args.pose_filter is True):
        if (isinstance(args.pose_ld, int) is False) or (isinstance(args.pose_matcher, int) is False):
            u.logger.error('Please posek, pose_ld, pose_matcher\n')
            raise ValueError('Please posek, pose_ld, pose_matcher')
    else:
        pass
    

    save_root = params_show(args)

    save_args(args, os.path.join(save_root, "args"))
    if args.train:
        run_tensorboard(save_root, args.port)
        writer = SummaryWriter(os.path.join(save_root, 'tensorboardX'))

    _,valid_dataset,index_dataset,_,train_loader,valid_loader,index_loader,test_loader = load_data(args)

    
    if args.train:
        if args.extractor is not None:
            train(args, train_loader, valid_loader, index_loader, valid_dataset, index_dataset, save_root, writer)
        elif args.handcraft is not None:
            train_handcraft(args, train_loader, valid_loader, index_loader, valid_dataset, index_dataset, save_root, writer)
    else:
        test(args, train_loader, valid_loader, index_loader, valid_dataset, index_dataset, save_root)
        
if __name__ == '__main__':
    main()