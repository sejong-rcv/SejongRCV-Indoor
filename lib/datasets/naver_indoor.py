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

pd.options.mode.chained_assignment = None

class Load_NMLC_indoor(torch.utils.data.Dataset):
    def __init__(self, root_path=None, ann_path=None, transform=None, use_tuple=False):

        self.root_path = root_path
        self.ann = pd.read_csv(ann_path) if ann_path is not None else None
        self.transform = transform
        self.use_tuple = use_tuple
        
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

    def __loadimg__(self, idx, plabel = 0, tflag = True):
        
        curr_label = np.concatenate((self.ann.iloc[idx,2:-1], np.array([plabel])), axis=0).astype('float64')
        #tx ty tz qw qx qy qz plabel
        curr_id = self.ann.iloc[idx].id
        curr_folder = str(self.ann.iloc[idx].date)
        #print(os.path.join(self.root_path, curr_folder,'images', curr_id+".jpg"))
        #import pdb;pdb.set_trace()
        curr_img = cv2.imread(os.path.join(self.root_path, curr_folder,'images', curr_id+".jpg"))

        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        
        data = {'image' : curr_img, \
                'label' : curr_label, \
                'index' : np.array([idx])}

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
        negind = np.random.choice(filt_xy, 1)
        
        return negind.item()

    def __selectpos__(self, idx):
        pos_list = self.ann.iloc[idx,-1:].tolist()
        pos_list = pos_list[0].split(", ")
        pos_list = list(map(int, pos_list))
        posind = np.random.choice(pos_list, 1)

        return posind.item()

    def __concatposneg__(self, anc, pos, neg):
        
        anc_img = anc['image'].unsqueeze(0)
        pos_img = pos['image'].unsqueeze(0)
        neg_img = neg['image'].unsqueeze(0)
        bundle_img = torch.cat((anc_img, pos_img, neg_img), dim=0)

        anc_lab = anc['label'].unsqueeze(0)
        pos_lab = pos['label'].unsqueeze(0)
        neg_lab = neg['label'].unsqueeze(0)
        bundle_lab = torch.cat((anc_lab, pos_lab, neg_lab), dim=0)
  
        bundle_index = np.array([anc['index'].item(), pos['index'].item(), neg['index'].item()])
   
        data = {'image' : bundle_img, \
                'label' : bundle_lab, \
                'index' : bundle_index}

        
        return data

    def __getitem__(self, idx):
        
        anchor = self.__loadimg__(idx, plabel=1)
        
        if self.use_tuple is False:
            return anchor
        #import pdb;pdb.set_trace()
        posind = self.__selectpos__(idx)
        pos = self.__loadimg__(posind, plabel=1)
        
        negind = self.__selectneg__(idx)
        neg = self.__loadimg__(negind, plabel=0)

        apn_tuple = self.__concatposneg__(anchor, pos, neg)

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

    def make_csv_v1(self, root_path, train=True, place="b1", lidar=False, make_valid=False):
        
        status = "train" if train is True else "test"
   
        if (place!='b1') and (place!='1f'):
            raise ValueError('Write correct place')
        
        dfdict = {}

        root_path = os.path.join(root_path, place, status)
        subset = os.listdir(root_path)
        subset.sort()

        id_list = []
        date_list = []
        tx_list = []
        ty_list = []
        tz_list = []
        qw_list = []
        qx_list = []
        qy_list = []
        qz_list = []
        

        min_ln = len(os.listdir(os.path.join(root_path, subset[0], 'images')))
        min_ind = 0
        for i, sub in enumerate(subset):
            if os.path.isdir(os.path.join(root_path, sub)) is False:
                continue
            if sub=='csv':
                continue
            
            if min_ln > len(os.listdir(os.path.join(root_path, sub, 'images'))):
                min_ln = len(os.listdir(os.path.join(root_path, sub, 'images')))
                min_ind = i


        for i,sub in enumerate(subset):
            
            if sub=='csv':
                continue

            if make_valid is True:
                if i!=min_ind:
                    continue
            else:
                if i==min_ind:
                    continue
            
            if os.path.isdir(os.path.join(root_path, sub)) is False:
                continue
            print(sub)
            h5_dict={}
            if train is True:
                f = h5py.File(os.path.join(root_path, sub, "jwon/groundtruth.hdf5"), 'r')
                keys = list(f.keys())
                for ki in range(0,len(keys),2):
                    if keys[ki].split("_")[0] != keys[ki+1].split("_")[0]:
                        import pdb; pdb.set_trace()

                    pose = f[keys[ki]][:]
                    stamp = f[keys[ki+1]][:]
                    
                    pose_stamp = np.concatenate((pose, stamp), axis=1)
                    h5_dict.update({keys[ki].split("_")[0]:pose_stamp})
                f.close()
            
           
            for cam_name in h5_dict.keys():
                if lidar is False:
                    if "lidar" in cam_name:
                        continue
                pose = h5_dict[cam_name][:,:-1]
                stamp = h5_dict[cam_name][:,-1].astype("int")

                stamp_list = list(map(str, stamp.squeeze()))
                pose_list = pose.tolist()
                for pose, stamp in zip(pose_list,stamp_list):
                    id_list.append("%s_%s"%(cam_name, stamp))
                    date_list.append(sub)
                    tx_list.append(pose[0])
                    ty_list.append(pose[1])
                    tz_list.append(pose[2])
                    qw_list.append(pose[3])
                    qx_list.append(pose[4])
                    qy_list.append(pose[5])
                    qz_list.append(pose[6])


        print(len(id_list))

        dfdict.update({'id':id_list})
        dfdict.update({'date':date_list})
        dfdict.update({'tx':tx_list})
        dfdict.update({'ty':ty_list})
        dfdict.update({'tz':tz_list})
        dfdict.update({'qw':qw_list})
        dfdict.update({'qx':qx_list})
        dfdict.update({'qy':qy_list})
        dfdict.update({'qz':qz_list})



        data_csv = pd.DataFrame(dfdict)
        if make_valid is False:
            data_csv.to_csv("./train_"+place+".csv", index=False)
        else:
            data_csv.to_csv("./valid_"+place+".csv", index=False)
        
        
        return

    def make_csv_v2(self, root_path, train=True, place="b1", lidar=False, make_valid=False, pdeg_thr=10, pdst_thr=0.5):
        
        status = "train" if train is True else "test"
   
        if (place!='b1') and (place!='1f'):
            raise ValueError('Write correct place')
        
        dfdict = {}

        root_path = os.path.join(root_path, place, status)
        subset = os.listdir(root_path)
        subset.sort()

        id_list = []
        date_list = []
        tx_list = []
        ty_list = []
        tz_list = []
        qw_list = []
        qx_list = []
        qy_list = []
        qz_list = []
        

        min_ln = len(os.listdir(os.path.join(root_path, subset[0], 'images')))
        min_ind = 0
        for i, sub in enumerate(subset):
            if os.path.isdir(os.path.join(root_path, sub)) is False:
                continue
            if sub=='csv':
                continue
            
            if min_ln > len(os.listdir(os.path.join(root_path, sub, 'images'))):
                min_ln = len(os.listdir(os.path.join(root_path, sub, 'images')))
                min_ind = i


        for i,sub in enumerate(subset):
            
            if sub=='csv':
                continue

            if make_valid is True:
                if i!=min_ind:
                    continue
            else:
                if i==min_ind:
                    continue
            
            if os.path.isdir(os.path.join(root_path, sub)) is False:
                continue
            
            h5_dict={}
            if train is True:
                f = h5py.File(os.path.join(root_path, sub, "jwon/groundtruth.hdf5"), 'r')
                keys = list(f.keys())
                for ki in range(0,len(keys),2):
                    if keys[ki].split("_")[0] != keys[ki+1].split("_")[0]:
                        import pdb; pdb.set_trace()

                    pose = f[keys[ki]][:]
                    stamp = f[keys[ki+1]][:]
                    
                    pose_stamp = np.concatenate((pose, stamp), axis=1)
                    h5_dict.update({keys[ki].split("_")[0]:pose_stamp})
                f.close()
            
           
            for cam_name in h5_dict.keys():
                if lidar is False:
                    if "lidar" in cam_name:
                        continue
                pose = h5_dict[cam_name][:,:-1]
                stamp = h5_dict[cam_name][:,-1].astype("int")

                stamp_list = list(map(str, stamp.squeeze()))
                pose_list = pose.tolist()
                for pose, stamp in zip(pose_list,stamp_list):
                    id_list.append("%s_%s"%(cam_name, stamp))
                    date_list.append(sub)
                    tx_list.append(pose[0])
                    ty_list.append(pose[1])
                    tz_list.append(pose[2])
                    qw_list.append(pose[3])
                    qx_list.append(pose[4])
                    qy_list.append(pose[5])
                    qz_list.append(pose[6])

        

        dfdict.update({'id':id_list})
        dfdict.update({'date':date_list})
        dfdict.update({'tx':tx_list})
        dfdict.update({'ty':ty_list})
        dfdict.update({'tz':tz_list})
        dfdict.update({'qw':qw_list})
        dfdict.update({'qx':qx_list})
        dfdict.update({'qy':qy_list})
        dfdict.update({'qz':qz_list})

        tx_list = np.expand_dims(np.asarray(tx_list), axis=1)
        ty_list = np.expand_dims(np.asarray(ty_list), axis=1)
        tz_list = np.expand_dims(np.asarray(tz_list), axis=1)
        qw_list = np.expand_dims(np.asarray(qw_list), axis=1)
        qx_list = np.expand_dims(np.asarray(qx_list), axis=1)
        qy_list = np.expand_dims(np.asarray(qy_list), axis=1)
        qz_list = np.expand_dims(np.asarray(qz_list), axis=1)

        pose_array = np.concatenate((tx_list,ty_list,tz_list,qw_list,qx_list,qy_list,qz_list), axis=1)
        
        dsc = "__"+place+"_valid" if make_valid is True else "__"+place+"_train"

        pos = []
        tol = 0
        viol = 0
        for i in tqdm.tqdm(range(pose_array.shape[0]), desc=dsc):
            xyz1 = pose_array[i,:3]
            qwxyz1 = pose_array[i,3:]
            subp = []
            cnt = 0
            for j in range(pose_array.shape[0]):
                if i==j:
                    continue
                xyz2 = pose_array[j,:3]
                qwxyz2 = pose_array[j,3:]
                dist = self.__meterbetween__(xyz1, xyz2)
                deg = self.__degreebetween__(qwxyz1, qwxyz2)

                if (pdeg_thr>=deg) and (pdst_thr>=dist):
                    subp.append(j)
                    cnt+=1
            if cnt==0:
                
                for j in range(pose_array.shape[0]):
                    if i==j:
                        continue
                    xyz2 = pose_array[j,:3]
                    qwxyz2 = pose_array[j,3:]
                    dist = self.__meterbetween__(xyz1, xyz2)
                    deg = self.__degreebetween__(qwxyz1, qwxyz2)

                    if (pdeg_thr+5>=deg) and (pdst_thr*2>=dist):
                        subp.append(j)
                        cnt+=1
                if cnt!=0:
                    tol+=1
            if cnt==0:
                subp.append(i)
                viol+=1
            pos.append(np.asarray(subp))
        
        data_csv = pd.DataFrame(dfdict)

        data_csv['pos_ind'] = "None"

        if len(pos) != data_csv.shape[0]:
            import pdb; pdb.set_trace()

        for i,pos_ind in enumerate(pos):
            str_pos = ", ".join(list(map(str,pos_ind.tolist())))
            data_csv['pos_ind'][i] = str_pos

        if make_valid is False:
            data_csv.to_csv("./train_"+place+".csv", index=False)
        else:
            data_csv.to_csv("./valid_"+place+".csv", index=False)
        
        oby = len(pos) - tol - viol
        return oby,tol, viol

    def analysis_dataset(self):
        alldata_b1 = db.__analysis_place__("./NaverML_indoor/", place="b1")
        alldata_1f = db.__analysis_place__("./NaverML_indoor/", place="1f")

        alldata = np.concatenate((alldata_b1, alldata_1f), axis=0)

        if os.path.isdir(os.path.join("analysis","frame_by_frame", "all", "b1+1f")) is False:
            os.mkdir(os.path.join("analysis","frame_by_frame", "all", "b1+1f"))
        if os.path.isdir(os.path.join("analysis","frame_by_frame", "all", "b1+1f", "distance")) is False:
            os.mkdir(os.path.join("analysis","frame_by_frame", "all", "b1+1f", "distance"))
        if os.path.isdir(os.path.join("analysis","frame_by_frame", "all", "b1+1f", "degree")) is False:
            os.mkdir(os.path.join("analysis","frame_by_frame", "all", "b1+1f", "degree"))

        self.__analysis_save__(alldata[:,0], os.path.join("analysis","frame_by_frame", "all", "b1+1f", "distance", "all.png"), "b1+1f", "distance")
        self.__analysis_save__(alldata[:,1], os.path.join("analysis","frame_by_frame", "all", "b1+1f", "degree", "all.png"), "b1+1f", "degree")

    def __analysis_place__(self, root_path, train=True, place="b1", lidar=False):
        
        status = "train" if train is True else "test"
   
        if (place!='b1') and (place!='1f'):
            raise ValueError('Write correct place')
        
        dfdict = {}

        root_path = os.path.join(root_path, place, status)
        subset = os.listdir(root_path)
        subset.sort()

        id_list = []
        date_list = []

        

        min_ln = len(os.listdir(os.path.join(root_path, subset[0], 'images')))
        min_ind = 0
        for i, sub in enumerate(subset):
            if os.path.isdir(os.path.join(root_path, sub)) is False:
                continue
            if sub=='csv':
                continue
            
            if min_ln > len(os.listdir(os.path.join(root_path, sub, 'images'))):
                min_ln = len(os.listdir(os.path.join(root_path, sub, 'images')))
                min_ind = i

        if os.path.isdir("analysis") is False:
            os.mkdir("analysis")
        if os.path.isdir(os.path.join("analysis","frame_by_frame")) is False:
            os.mkdir(os.path.join("analysis","frame_by_frame"))
        anlys_set = {}
        for i,sub in enumerate(subset):
            
            type_name="train"

            if sub=='csv':
                continue
            if i==min_ind:
                type_name="valid"
            if os.path.isdir(os.path.join(root_path, sub)) is False:
                continue
            
            if os.path.isdir(os.path.join("analysis","frame_by_frame", place)) is False:
                os.mkdir(os.path.join("analysis","frame_by_frame", place))
            if os.path.isdir(os.path.join("analysis","frame_by_frame", place, sub)) is False:
                os.mkdir(os.path.join("analysis","frame_by_frame", place, sub))
            if os.path.isdir(os.path.join("analysis","frame_by_frame", place, sub, "distance")) is False:
                os.mkdir(os.path.join("analysis","frame_by_frame", place, sub, "distance"))
            if os.path.isdir(os.path.join("analysis","frame_by_frame", place, sub, "degree")) is False:
                os.mkdir(os.path.join("analysis","frame_by_frame", place, sub, "degree"))



            h5_dict={}
            if train is True:
                f = h5py.File(os.path.join(root_path, sub, "jwon/groundtruth.hdf5"), 'r')
                keys = list(f.keys())
                for ki in range(0,len(keys),2):
                    if keys[ki].split("_")[0] != keys[ki+1].split("_")[0]:
                        import pdb; pdb.set_trace()

                    pose = f[keys[ki]][:]
                    stamp = f[keys[ki+1]][:]
                    
                    pose_stamp = np.concatenate((pose, stamp), axis=1)
                    h5_dict.update({keys[ki].split("_")[0]:pose_stamp})
                f.close()
            
            anlys_cam = {}
            for cam_name in h5_dict.keys():
                if lidar is False:
                    if "lidar" in cam_name:
                        continue
                pose = h5_dict[cam_name][:,:-1]
                stamp = h5_dict[cam_name][:,-1].astype("int")
                
                ind = np.argsort(stamp)
                stamp = stamp[ind]
                pose = pose[ind]

                dist_list = []
                deg_list = []
                for i in range(0, len(stamp)-1):
                    xyz1 = pose[i, :3]
                    xyz2 = pose[i+1, :3]

                    qwxyz1 = pose[i, 3:]
                    qwxyz2 = pose[i+1, 3:]

                    dist = self.__meterbetween__(xyz1, xyz2)
                    deg = self.__degreebetween__(qwxyz1, qwxyz2)

                    dist_list.append(dist)
                    deg_list.append(deg)
                
                dist_list = np.expand_dims(np.asarray(dist_list), axis=1)
                deg_list = np.expand_dims(np.asarray(deg_list), axis=1)
                dide_list = np.concatenate((dist_list, deg_list), axis=1)
                anlys_cam.update({cam_name:dide_list})

                self.__analysis_save__(dide_list[:,0], os.path.join("analysis","frame_by_frame", place, sub, "distance", type_name+"_"+cam_name+".png"), place, "distance")
                self.__analysis_save__(dide_list[:,1], os.path.join("analysis","frame_by_frame", place, sub, "degree", type_name+"_"+cam_name+".png"), place, "degree")


            anlys_set.update({sub : anlys_cam})

        # index 0: distance, 1: degree
        alldata = []
        for set_key, set_val in anlys_set.items():
            for key, val in set_val.items():
                alldata.extend(val.tolist())
        alldata = np.asarray(alldata)

        if os.path.isdir(os.path.join("analysis","frame_by_frame", "all")) is False:
            os.mkdir(os.path.join("analysis","frame_by_frame", "all"))
        if os.path.isdir(os.path.join("analysis","frame_by_frame", "all", place)) is False:
            os.mkdir(os.path.join("analysis","frame_by_frame", "all", place))
        if os.path.isdir(os.path.join("analysis","frame_by_frame", "all", place, "distance")) is False:
            os.mkdir(os.path.join("analysis","frame_by_frame", "all", place, "distance"))
        if os.path.isdir(os.path.join("analysis","frame_by_frame", "all", place, "degree")) is False:
            os.mkdir(os.path.join("analysis","frame_by_frame", "all", place, "degree"))

        self.__analysis_save__(alldata[:,0], os.path.join("analysis","frame_by_frame", "all", place, "distance", "all.png"), place, "distance")
        self.__analysis_save__(alldata[:,1], os.path.join("analysis","frame_by_frame", "all", place, "degree", "all.png"), place, "degree")
        
        return alldata

    def __analysis_save__(self, data, path, place, typ="degree"):
        import matplotlib.pyplot as plt 
        plt.figure(figsize=(10, 6))
        ys, xs, pathces = plt.hist(data, bins=10, rwidth=0.8)
 
        ysp = ys / ys.sum()

        for i in range(0, len(ys)):
            plt.text(x=xs[i]+0.08, y=ys[i]+0.15, 
                    s='{:0>.2f}%'.format(ysp[i]*100), 
                    fontproperties='serif', 
                    fontsize=10,
                    color='red')
        y_min, y_max = plt.ylim()

        plt.ylim(y_min, y_max+0.5)
        if typ=="degree":
            plt.xticks([(xs[i]+xs[i+1])/2 for i in range(0, len(xs)-1)], 
                ["{:d} ~ {:d}".format(int(xs[i]), int(xs[i+1])) for i in range(0, len(xs)-1)], fontsize=8)
            plt.xlabel("degree")
        else:
            plt.xticks([(xs[i]+xs[i+1])/2 for i in range(0, len(xs)-1)], 
                ["{:.1f} ~ {:.1f}".format(xs[i], xs[i+1]) for i in range(0, len(xs)-1)], fontsize=8)
            plt.xlabel("meter")
        plt.title(place)
        plt.savefig(path)
        plt.close()

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
