import pandas as pd
import os
import numpy as np
import cv2
import torch
import torch.utils.data
import tqdm
import h5py


from scipy.spatial.transform import Rotation
from scipy.spatial.distance import euclidean
from pyquaternion import Quaternion

from lib.datasets.transforms import *
from lib import utils as u

pd.options.mode.chained_assignment = None

class Load_NMLC_indoor(torch.utils.data.Dataset):
    def __init__(self, root_path=None, ann_path=None, transform=None, use_tuple=False, sel_pos=0, neg_num=1):

        self.root_path = root_path
        self.ann = pd.read_csv(ann_path) if ann_path is not None else None
        self.transform = transform
        self.use_tuple = use_tuple
        self.sel_pos = sel_pos
        self.neg_num = neg_num
    def __len__(self):
        return len(self.ann)

    def __getcameraparams__(self, index):
        text_path = os.path.join(self.root_path, self.ann["date"][index], "camera_parameters.txt")
        f = open(text_path)
        while(1):
            line = f.readline()
            cam_id = self.ann["id"][index].split("_")[0]
            
            if cam_id==line.split(" ")[0]:
                line = line.strip("\n")
                lineblock = line.split(" ")
                K = np.asarray([[lineblock[3], 0, lineblock[5]], [0, lineblock[4], lineblock[6]], [0, 0, 1]]).astype("float")
                rdist = np.array([lineblock[7], lineblock[8], lineblock[11]]).astype("float")
                tdist = np.array([lineblock[9], lineblock[10]]).astype("float")
                break
        f.close

        return K, rdist, tdist

    def __loadimg__(self, idx, plabel = 0, tflag = True, name = False, gray = False, image=True):

        curr_label = np.concatenate((self.ann.iloc[idx,2:-1], np.array([plabel])), axis=0).astype('float64')
        #tx ty tz qw qx qy qz plabel
        curr_id = self.ann.iloc[idx].id
        curr_folder = str(self.ann.iloc[idx].date)
        
        if image is True:
            if gray is False:
                curr_img = cv2.imread(os.path.join(self.root_path, curr_folder,'images', curr_id+".jpg"))
            else:
                curr_img = cv2.imread(os.path.join(self.root_path, curr_folder,'images', curr_id+".jpg"), cv2.IMREAD_GRAYSCALE)

            h, w = curr_img.shape[0], curr_img.shape[1]
            K, rdist, tdist = self.__getcameraparams__(idx)
            if (w!=2*K[0,2]) | (h!=2*K[1,2]):
                distCoeffs = np.array([rdist[0], rdist[1], tdist[0], tdist[1], rdist[2]])
                curr_img = cv2.undistort(curr_img, K, distCoeffs)

            if gray is False:
                curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
                
            if name is True:
                floor = self.root_path.split("/")[2]
                name_path = os.path.join(floor, curr_folder, curr_id)

                data = {'image' : curr_img, \
                        'label' : curr_label, \
                        'index' : np.array([idx]), \
                        'name'  : name_path}
            else:            
                data = {'image' : curr_img, \
                        'label' : curr_label, \
                        'index' : np.array([idx])}
        else:
            if name is True:
                floor = self.root_path.split("/")[2]
                name_path = os.path.join(floor, curr_folder, curr_id)
                data = name_path

        if self.transform and (tflag is True):
            data = self.transform(data)


        return data

    def __selectneg__(self, idx, xbound=10, ybound=10):

        x_coor = self.ann.iloc[idx].tx
        y_coor = self.ann.iloc[idx].ty
        
        filt_x1 = np.where(self.ann['tx'] > x_coor+xbound)[0]
        filt_x2 = np.where(self.ann['tx'] < x_coor-xbound)[0]
        filt_x = np.union1d(filt_x1,filt_x2)

        filt_y1 = np.where(self.ann['ty'] > y_coor+ybound)[0]
        filt_y2 = np.where(self.ann['ty'] < y_coor-ybound)[0]
        filt_y = np.union1d(filt_y1,filt_y2)

        filt_xy = np.intersect1d(filt_x, filt_y)
        negind = np.random.choice(filt_xy, self.neg_num)
        
        return negind

    def __selectpos__(self, idx):
        if self.sel_pos==0:
            pos_list = self.ann.iloc[idx,-1:].tolist()
            pos_list = pos_list[0].split(", ")
            pos_list = list(map(int, pos_list))
            posind = np.random.choice(pos_list, 1).item()
        elif self.sel_pos==1:
            posind = idx-1 if self.ann.index[-1] == idx else idx+1
        return posind

    def __concatposneg__(self, anc, pos, neg_set):
        
        
        anc_img = anc['image'].unsqueeze(0)
        pos_img = pos['image'].unsqueeze(0)

        anc_lab = anc['label'].unsqueeze(0)
        pos_lab = pos['label'].unsqueeze(0)

        neg_img_set = torch.Tensor([])
        neg_lab_set = torch.Tensor([]).double()
        bundle_index = [anc['index'].item(), pos['index'].item()]
        for i in range(len(neg_set)):
            neg_img = neg_set[i]['image'].unsqueeze(0)
            neg_img_set = torch.cat((neg_img_set, neg_img), dim=0)
            neg_lab = neg_set[i]['label'].unsqueeze(0)
            neg_lab_set = torch.cat((neg_lab_set, neg_lab), dim=0)
            bundle_index.append(neg_set[i]['index'].item())

        bundle_lab = torch.cat((anc_lab, pos_lab, neg_lab_set), dim=0)
        bundle_img = torch.cat((anc_img, pos_img, neg_img_set), dim=0)
        bundle_index = np.asarray(bundle_index)
  
        data = {'image' : bundle_img, \
                'label' : bundle_lab, \
                'index' : bundle_index}

        
        return data

    def __getitem__(self, idx):
        
        anchor = self.__loadimg__(idx, plabel=1)
        
        if self.use_tuple is False:
            return anchor

        posind = self.__selectpos__(idx)
        pos = self.__loadimg__(posind, plabel=1)
        
        negind = self.__selectneg__(idx)
        neg_set = []
        for ni in negind:
            neg = self.__loadimg__(ni, plabel=0)
            neg_set.append(neg)

        apn_tuple = self.__concatposneg__(anchor, pos, neg_set)

        return apn_tuple

    def collate_func(self, batch):
    

        if type(batch[0]['image']) is np.ndarray:
            image = []
            label = []
            pose = []
            index = []
            for i, val in enumerate(batch):
                image.append(val['image'].tolist())
                label.append(val['label'][-1:].tolist())
                pose.append(val['label'][:-1].tolist())
                index.extend(val['index'])
            image = np.asarray(image).astype("uint8")
            label = np.asarray(label)
            pose = np.asarray(pose)
            index = np.asarray(index)
        else:
            image = torch.tensor([])
            label = torch.tensor([]).long()
            pose = torch.tensor([]).double()
            index = []
            for i, val in enumerate(batch):
                
                if val['image'].dim()!=4:
                    val['image'] = val['image'].unsqueeze(0)
                if val['label'].dim()==1:
                    val['label'] = val['label'].unsqueeze(0)
                
                image = torch.cat((image, val['image'].float()), dim=0)
                label = torch.cat((label, val['label'][:,-1:].long()), dim=0)
                pose = torch.cat((pose, val['label'][:,:-1]), dim=0)

                index.extend(val['index'])

            index = np.asarray(index)
    
        data = {'image' : image,
                'label' : label,
                'pose' : pose, 
                'index' : index}

        return data

    def __meterbetween__(self, xyz1, xyz2):
        return euclidean(xyz1, xyz2)

    def __degreebetween__(self, quart_wxyz1, quart_wxyz2):
            
        quart1 = Quaternion(quart_wxyz1)
        quart2 = Quaternion(quart_wxyz2)

        # if you want to see whether this code is correct or not
        # test under
        # A = Quaternion(axis=[1, 0, 0], angle=np.pi)
        # B = Quaternion(axis=[1, 0, 0], angle=np.pi/2)
        # (A.conjugate*B).degrees -> It may be 180-90=90
        return abs((quart1.conjugate*quart2).degrees)

    def __poseproc__(self, pose):
        
        pose_matrix = np.identity(4)
        pose_matrix[:3, 3] = pose[:3]

        r = Rotation.from_quat(pose[3:])
        mat = r.as_rotvec()
        pose_matrix[:3, :3] = mat
        
        return pose_matrix

    def make_video(self, root_path):
        import matplotlib.pyplot as plt
        
        if os.path.isdir("./jobs/frame/") is False:
            os.mkdir("./jobs/frame/")


        for place in ['b1', '1f']:
            if os.path.isdir(os.path.join("./jobs/frame",place)) is False:
                os.mkdir(os.path.join("./jobs/frame",place))

            for status  in ['train', 'test']:

                if os.path.isdir(os.path.join("./jobs/frame",place,status)) is False:
                    os.mkdir(os.path.join("./jobs/frame",place,status))

                status_path = os.path.join(root_path, place, status)
                subset = os.listdir(status_path)
                subset.sort()
                
                for sub in subset:
                    if sub=="csv":
                        continue
              
                    if os.path.isdir(os.path.join("./jobs/frame",place,status,sub)) is False:
                        os.mkdir(os.path.join("./jobs/frame",place,status,sub))

                    sub_path = os.path.join(status_path, sub, "images")
                    img_list = os.listdir(sub_path)
                    img_list.sort()

                    vidname = place + "_" + status + "_" + sub + ".avi"
                    out = cv2.VideoWriter(os.path.join("./jobs/frame",place,status,sub,vidname),cv2.VideoWriter_fourcc(*'DIVX'), 15, (640,512))

                    for i,img in enumerate(tqdm.tqdm(img_list, desc=place+status+sub)):

                        curr_img = cv2.imread(os.path.join(sub_path, img))
                        curr_img = cv2.resize(curr_img, (640,512))

                        out.write(curr_img)
                    out.release()             

