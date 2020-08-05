import cv2
import numpy as np
import pandas as pd
import pcl
import os
import time as tm
import tqdm
import torch
import pickle
import cupy as cp
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.transform import Rotation 
from scipy.spatial.distance import euclidean
from pyquaternion import Quaternion

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.transform import AffineTransform
import numpy as np
import pandas as pd
from SuperGlue.matching import Matching
from SuperGlue.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
from lib import metric as mt
import os
import pcl
import pickle
from scipy.spatial.distance import euclidean 
import cupy as cp
from lib import datasets as db
from lib import extractor as ex
from lib import metric as mt
from lib import handcraft_extractor as he
import torchvision.transforms as tt
import torch
from torch.utils.data import DataLoader
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--floor', type=str, default="b1", help='floor')
    parser.add_argument('--topk_name', type=str, default="test", help='floor')
    parser.add_argument('--save_name', type=str, default="test", help='floor')
    args = parser.parse_args()
    return args

def __homogeneousCoord__(pts,use_cupy=False):
    if use_cupy is False:
            ax = 1 if pts.shape[0]>pts.shape[1] else 0
            ones = np.ones((pts.shape[0],1)) if ax==1 else np.ones((1, pts.shape[0]))
            return np.concatenate((pts, ones), axis=ax)
    else:
            ax = 1 if pts.shape[0]>pts.shape[1] else 0
            ones = cp.ones((pts.shape[0],1)) if ax==1 else cp.ones((1, pts.shape[0]))
            return cp.concatenate((pts, ones), axis=ax)

def __hnormalized__( pts,use_cupy=False):
    if use_cupy is False:
            src = pts.T if pts.shape[0]<pts.shape[1] else pts
            dst = (src/src[:,2:])[:,:2]
            return dst
    else:
            src = pts.T if pts.shape[0]<pts.shape[1] else pts
            dst = (src/src[:,2:])[:,:2]
            return dst

def __world2image__( qwxyz, t, K, point3d,use_cupy=False):
    if use_cupy is False:
            I = np.eye(4)
            I[:3, 3] = t
            quat = Quaternion(qwxyz)
            Rm = quat.rotation_matrix
            Rm_inv = np.linalg.inv(Rm)
            I[:3, :3] = Rm
            Iinv = np.linalg.inv(I)
            hpts3d = __homogeneousCoord__(point3d).T
            point3d_local = np.matmul(Iinv, hpts3d)[0:3,:]
            image_pixel = __hnormalized__(np.matmul(K, point3d_local).T)
            
    else:
            t=t.astype("float")
            K=K.astype("float")
            I = cp.eye(4)
            I[:3, 3] = cp.asarray(t)
            quat = Quaternion(qwxyz)
            Rm = cp.asarray(quat.rotation_matrix)
            Rm_inv = cp.linalg.inv(Rm)
            Rm_inv = cp.asnumpy(Rm_inv)
            I[:3, :3] = cp.asarray(Rm)
            Iinv = cp.linalg.inv(I)
            hpts3d =  __homogeneousCoord__(cp.asarray(point3d), use_cupy).T
            point3d_local = cp.matmul(Iinv, hpts3d)[0:3,:]
            image_pixel =  __hnormalized__(cp.matmul(cp.asarray(K), point3d_local).T, use_cupy)
            image_pixel = cp.asnumpy(image_pixel)

    return image_pixel, Rm_inv

def __maskPlaneInlier__( pts, K, factor=0):
    if pts.shape[0]>pts.shape[1]:
        xmask = (pts[:,0]>=0) & (pts[:,0]<K[0,2]*2)
        ymask = (pts[:,1]>=0) & (pts[:,1]<K[1,2]*2)
    else:
        xmask = (pts[0,:]>=0) & (pts[0,:]<K[0,2]*2)
        ymask = (pts[1,:]>=0) & (pts[1,:]<K[1,2]*2)

    total_mask = xmask & ymask
    return total_mask

def __applyMask__( pts, mask, K=None):
    inlier = pts[mask]
    if K is not None:
        inlier[:,0] = inlier[:,0]+K[0,2]
        inlier[:,1] = inlier[:,1]+K[1,2]
    return inlier

def __surroundingMask__(inlier3d_map, ixyz, qwxyz, alpha=15):
    #import pdb;pdb.set_trace()
    
    qm = Quaternion(qwxyz).rotation_matrix
    qmi = np.linalg.inv(qm)

    rotated3d_map = np.matmul(qmi, (inlier3d_map-ixyz).T.astype("float")).T
    rotated3d_ixyz = np.matmul(qmi, (ixyz-ixyz).T.astype("float")).T

    img_mask_x = (rotated3d_map[:,0]<alpha) & (rotated3d_map[:,0]>-alpha)
    img_mask_z = (rotated3d_map[:,2]<alpha) & (rotated3d_map[:,2]>0)
    img_mask = img_mask_x&img_mask_z

    return img_mask

def __allocate3Dpoints__( kps_2d, surround2d_map, surround3d_map, res_thr=0.5):
    knn = mt.kNN_GPU(d=surround2d_map.shape[1],GPU=True, GPU_Number=torch.cuda.current_device())
    knn.train(np.ascontiguousarray(surround2d_map, dtype=np.float32))
    
    topk_list = knn.predict(np.ascontiguousarray(kps_2d, dtype=np.float32), 1)
    #try:
    sift_3d = surround3d_map[topk_list].squeeze(1)
    correspond_2d = surround2d_map[topk_list].squeeze(1)
    residual = np.linalg.norm((correspond_2d-kps_2d),axis=1)
    # print("Residual mean : {}".format(residual.mean()))
    residual_mask = residual<res_thr
    # print("Filtered residual mean : {}".format(residual[residual_mask].mean()))
    knn.delete()
    del knn
    return sift_3d, residual_mask

def __solvePnP__( pair_q, pair3d, K, dst=[0,0,0,0,0]):
    #import pdb;pdb.set_trace()
    _,solvR,solvt,inlierR = cv2.solvePnPRansac(np.expand_dims(pair3d, axis=2).astype("float64"), \
                                            np.expand_dims(pair_q, axis=2).astype("float64"), \
                                            K.astype("float64"), \
                                            np.expand_dims(np.array(dst), axis=1).astype("float64"), \
                                            iterationsCount=100000, \
                                            useExtrinsicGuess = True, \
                                            confidence = 0.999, \
                                            reprojectionError = 8, \
                                            flags = cv2.SOLVEPNP_AP3P) #0.8

    # The number of inliers is needed more than 20
    if inlierR.shape[0]<=0:
        return

    solvRR,_ = cv2.Rodrigues(solvR)
    solvRR_inv = np.linalg.inv(solvRR)
    solvtt = -np.matmul(solvRR_inv,solvt)

    rot = cv2.Rodrigues(solvRR_inv)[0].squeeze(1)
    query_qwxyz = Quaternion(matrix=solvRR_inv).elements
    query_xyz = solvtt.squeeze(1)

    return query_xyz, query_qwxyz, inlierR

def getcameraparams(info,qd=True):
    
    if qd:
        text_path=qrtext_path%info.date
    else:
        text_path=dbtext_path%info.date
    f = open(text_path)
    
    while(1):
        line = f.readline()
        cam_id = info.id.split("_")[0]

        if cam_id==line.split(" ")[0]:
            line = line.strip("\n")
            lineblock = line.split(" ")
            K = np.asarray([[lineblock[3], 0, lineblock[5]], [0, lineblock[4], lineblock[6]], [0, 0, 1]]).astype("float")
            rdist = np.array([lineblock[7], lineblock[8], lineblock[11]]).astype("float")
            tdist = np.array([lineblock[9], lineblock[10]]).astype("float")
            break
    f.close

    return K, rdist, tdist
def local_extract( image):
    feat = model.extract(image)
    kps = feat["keypoints"][:,:2]
    desc = feat['descriptors']

    return kps, desc
def local_match_ori( query_kps, query_desc, index_kps, index_desc, pts_cloud):

    knn = mt.kNN_GPU(d=index_desc.shape[1],GPU=True, GPU_Number=torch.cuda.current_device())
    knn.train(np.ascontiguousarray(index_desc, dtype=np.float32))
    qtoi = knn.predict(np.ascontiguousarray(query_desc, dtype=np.float32), 1)
    query_ind = np.expand_dims(np.arange(qtoi.shape[0]), axis=1)
    qind_qtoi = np.concatenate((query_ind, qtoi), axis=1)

    knn = mt.kNN_GPU(d=query_desc.shape[1],GPU=True, GPU_Number=torch.cuda.current_device())
    knn.train(np.ascontiguousarray(query_desc, dtype=np.float32))
    itoq = knn.predict(np.ascontiguousarray(index_desc, dtype=np.float32), 1)
    index_ind = np.expand_dims(np.arange(itoq.shape[0]), axis=1)
    itoq_iind = np.concatenate((itoq, index_ind), axis=1)

    total = np.concatenate((qind_qtoi, itoq_iind), axis=0)
    unique, match_cnt = np.unique(total, axis=0, return_counts=True)
    match_mask = np.where(match_cnt!=1)[0]
    matches = unique[match_mask]

    kps_left = query_kps[matches[:, 0],:]
    kps_right = index_kps[matches[:, 1],:]
    pts_3d = pts_cloud[matches[:, 1],:]
    np.random.seed(0)

    _, inliers = ransac(
        (kps_left, kps_right),
        AffineTransform, min_samples=3,
        residual_threshold=20, max_trials=1000
    )

    pts_query = kps_left[inliers]
    pts_index = kps_right[inliers]
    pts_3d = pts_3d[inliers]
    
    knn.delete()
    del knn

    if inliers.sum()<=40:
        return
    return pts_query, pts_index, pts_3d

def local_match(qrkey,dbkey,pts_3d):
    kps_left=qrkey
    kps_right=dbkey
    pts_query = kps_left
    pts_index = kps_right
    pts_3d = pts_3d
    return pts_query, pts_index, pts_3d     
if __name__=="__main__":
    ###################################################################################
    args = get_args()
    floor=args.floor
    print(floor)
    toplist=np.load("Ensemble_"+floor+"_result_Hard_GeM_pca_conf09_all.npy")

    qrcsv="NaverML_indoor/"+floor+"/test/csv/test_"+floor+".csv"
    dbcsv="NaverML_indoor/"+floor+"/train/csv/v1/train_all/train_"+floor+".csv"
    ###################################################################################
    qrcsv=pd.read_csv(qrcsv)
    dbcsv=pd.read_csv(dbcsv)
    
    roottrainimage="NaverML_indoor/"+floor+"/train/%s/images/%s.jpg"
    roottestimage="NaverML_indoor/"+floor+"/test/%s/images/%s.jpg"

    model=Matching().eval().cuda()

    if floor=="b1":
        cld_path = os.path.join("/raid3","naver_challenge2020_img2pcds","indoor")
    else:
        cld_path="/raid/datasets/naver_challenge2020_img2pcds/indoor"


    qrtext_path = os.path.join("./NaverML_indoor/"+floor+"/test/", "%s", "camera_parameters.txt")
    dbtext_path = os.path.join("./NaverML_indoor/"+floor+"/train/", "%s", "camera_parameters.txt")

    qdiff_xyz = []
    qdiff_qwxyz = []

    idiff_xyz = []
    idiff_qwxyz = []

    q_xyz = []
    q_qwxyz = []
    ###################################################################################    
    from tqdm import tqdm 
    cnt=0
    results=[]

    for qrind,dbinds in tqdm(enumerate(toplist)):
        qrresult=dict()
        qrresult["floor"]=floor
        qrinf=qrcsv.iloc[qrind]
        qrpath=roottestimage%(qrinf.date,qrinf.id)
        qrresult["name"]=qrinf.id+".jpg"

        ############################################image matching
        cnt+=1

        query_K, query_rdist, query_tdist = getcameraparams(qrinf)
        query_dist = np.asarray([query_rdist[0], query_rdist[1], query_tdist[0], query_tdist[1], query_rdist[2]])
        if qrresult["name"][0]=="A":
            query_dist_=None
            query_K_=None
        else:
            query_dist_=query_dist
            query_K_=query_K
        image0, inp0, scales0 =read_image(qrpath, "cuda", [1024,960], 0, False,query_dist_,query_K_)
        maxin=0
        maxindex=0
        for i in range(5):        
            index=dbinds[i]

            for idx,dbind in enumerate(range(index,index+1)):
                dbinf=dbcsv.iloc[dbind]
                dbpath=roottrainimage%(dbinf.date,dbinf.id)
                image1, inp1, scales1 =read_image(dbpath, "cuda", [1024,960], 0, False)
                with torch.no_grad():
                    pred=model({'image0': inp0, 'image1': inp1})

                pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']

                valid = (matches >-1)#*(conf>0.7)
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                query_kps=mkpts0
                index_kps=mkpts1

                index_K, index_rdist, index_tdist  = getcameraparams(dbinf,False)

                index_dist = np.asarray([index_rdist[0], index_rdist[1], index_tdist[0], index_tdist[1], index_rdist[2]])

                ############################################pose 
                index_xyz = np.asarray(dbinf[2:5])
                index_qwxyz = np.asarray(dbinf[5:])

                ############################################ pcd
                pcd_file = os.path.join(cld_path,floor, dbinf.date, "img2pcd", dbinf.id+".pkl")
                if os.path.isfile(pcd_file):
                    with open(pcd_file, "rb") as a_file:
                        surround3d_map = pickle.load(a_file)
                    if isinstance(surround3d_map, cp.ndarray):
                        surround3d_map = cp.asnumpy(surround3d_map)
                    surround2d_map, Rm = __world2image__(index_qwxyz, index_xyz, index_K, surround3d_map,use_cupy=True)
                else:
                    print("No PCD")

                query_kps=query_kps*scales0
                index_kps=index_kps*scales1
                try:
                    alloc3d, residual_mask = __allocate3Dpoints__(index_kps, surround2d_map, surround3d_map)
                except:
                    import pdb;pdb.set_trace()
                pts_query, pts_index, pts_3d = local_match(query_kps,  index_kps,  alloc3d)
                if idx==0:
                    pts_querys=pts_query
                    pts_3ds=pts_3d
                else:
                    pts_querys=np.vstack((pts_querys,pts_query))
                    pts_3ds=np.vstack((pts_3ds,pts_3d))
            try:
                query_xyz, query_qwxyz, inlierR = __solvePnP__(pts_querys, pts_3ds, query_K)
            except:
                query_xyz, query_qwxyz =index_xyz, index_qwxyz
                continue
            if inlierR.shape[0]>maxin:
                maxin=inlierR.shape[0]
                maxindex=index


        if True:
            if maxindex==0:
                maxindex=2
            elif maxindex==(len(dbcsv)-1):
                maxindex-=2
            for idx,dbind in enumerate(range(maxindex-1,maxindex+2)):
                dbinf=dbcsv.iloc[dbind]
                dbpath=roottrainimage%(dbinf.date,dbinf.id)
                image1, inp1, scales1 =read_image(dbpath, "cuda", [1024,960], 0, False)
                with torch.no_grad():
                    pred=model({'image0': inp0, 'image1': inp1})

                pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']

                valid = (matches >-1)*(conf>0.4)
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                query_kps=mkpts0
                index_kps=mkpts1

                index_K, index_rdist, index_tdist  = getcameraparams(dbinf,False)

                index_dist = np.asarray([index_rdist[0], index_rdist[1], index_tdist[0], index_tdist[1], index_rdist[2]])

                ############################################pose 
                index_xyz = np.asarray(dbinf[2:5])
                index_qwxyz = np.asarray(dbinf[5:])

                ############################################ pcd
                pcd_file = os.path.join(cld_path,floor, dbinf.date, "img2pcd", dbinf.id+".pkl")
                if os.path.isfile(pcd_file):
                    with open(pcd_file, "rb") as a_file:
                        surround3d_map = pickle.load(a_file)
                    if isinstance(surround3d_map, cp.ndarray):
                        surround3d_map = cp.asnumpy(surround3d_map)
                    surround2d_map, Rm = __world2image__(index_qwxyz, index_xyz, index_K, surround3d_map,use_cupy=True)
                else:
                    print("No PCD")

                query_kps=query_kps*scales0
                index_kps=index_kps*scales1
                try:
                    alloc3d, residual_mask = __allocate3Dpoints__(index_kps, surround2d_map, surround3d_map)
                except:
                    import pdb;pdb.set_trace()
                pts_query, pts_index, pts_3d = local_match(query_kps,  index_kps,  alloc3d)
                if idx==0:
                    pts_querys=pts_query
                    pts_3ds=pts_3d
                else:
                    pts_querys=np.vstack((pts_querys,pts_query))
                    pts_3ds=np.vstack((pts_3ds,pts_3d))
            try:
                query_xyz, query_qwxyz, inlierR = __solvePnP__(pts_querys, pts_3ds, query_K)
            except:
                query_xyz, query_qwxyz =index_xyz, index_qwxyz
                print("Solve PnP")
                
        qrresult["qw"]=query_qwxyz[0]
        qrresult["qx"]=query_qwxyz[1]
        qrresult["qy"]=query_qwxyz[2]
        qrresult["qz"]=query_qwxyz[3]
        qrresult["x"]=query_xyz[0]
        qrresult["y"]=query_xyz[1]
        qrresult["z"]=query_xyz[2]
        results.append(qrresult)
    with open(floor+'_%s.json'%args.save_name, 'w') as make_file:
        json.dump(results, make_file, indent="\t")
