import pickle
import numpy as np
import cv2
import os
import pcl
import cupy as cp
import pandas as pd
from pyquaternion import Quaternion
import json


if __name__ == "__main__":

    root_csv = ".csv"
    root_csv = pd.read_csv(root_csv)
    
    ind = 4000 

    covis = 1

    img_id = root_csv.iloc[ind,:].id
    img_date = root_csv.iloc[ind,:].date
    img_xyz = root_csv.iloc[ind,2:5]
    img_qwxyz = root_csv.iloc[ind,5:9]
    img_rt = Quaternion(img_qwxyz).rotation_matrix

    pcd_file = os.path.join("/naver_challenge2020_img2pcds/indoor/b1", img_date, "img2pcd", img_id+".pkl")
    with open(pcd_file, "rb") as a_file:
        surround3d_map = pickle.load(a_file)
    
    if isinstance(surround3d_map, cp.ndarray):
        surround3d_map = cp.asnumpy(surround3d_map)

        
    for i in range(1,covis+1):
        img_id = root_csv.iloc[ind-i,:].id
        img_date = root_csv.iloc[ind-i,:].date
        pcd_file = os.path.join("/naver_challenge2020_img2pcds/indoor/b1", img_date, "img2pcd", img_id+".pkl")
        with open(pcd_file, "rb") as a_file:
            locals()['map{}'.format(ind-i)] = pickle.load(a_file)
        
        if isinstance(locals()['map{}'.format(ind-i)], cp.ndarray):
            locals()['map{}'.format(ind-i)] = cp.asnumpy(locals()['map{}'.format(ind-i)])
        
        surround3d_map = np.concatenate((surround3d_map, locals()['map{}'.format(ind-i)]), axis=0)


        img_id = root_csv.iloc[ind+i,:].id
        img_date = root_csv.iloc[ind+i,:].date
        pcd_file = os.path.join("/naver_challenge2020_img2pcds/indoor/b1", img_date, "img2pcd", img_id+".pkl")
        with open(pcd_file, "rb") as a_file:
            locals()['map{}'.format(ind+i)] = pickle.load(a_file)

        if isinstance(locals()['map{}'.format(ind+i)], cp.ndarray):
            locals()['map{}'.format(ind+i)] = cp.asnumpy(locals()['map{}'.format(ind+i)])

        surround3d_map = np.concatenate((surround3d_map, locals()['map{}'.format(ind+i)]), axis=0)

    sampling_mask = np.arange(0, surround3d_map.shape[0]-1, covis*2+1)

    inlier_cloud = pcl.PointCloud()
    inlier_cloud.from_array(surround3d_map[sampling_mask])
    inlier_cloud.to_file("bf.pcd".encode('utf-8'))

    img_path = os.path.join("./NaverML_indoor/b1/train/", img_date, "images", img_id+".jpg")
    a = cv2.imread(img_path)
    cv2.imwrite("bf.jpg", a)

    print(img_xyz)
    print(img_rt)

    xyz = pd.DataFrame(img_xyz)
    xyz.to_csv("xyz.txt", index=False)

    rt = pd.DataFrame(img_rt)
    rt.to_csv("rt.txt", index=False)

    import pdb; pdb.set_trace()