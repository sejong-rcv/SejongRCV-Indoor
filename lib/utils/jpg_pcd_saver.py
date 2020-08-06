import pickle
import numpy as np
import cv2
import os
import pcl
import cupy as cp




if __name__ == "__main__":

    
    pcd_file = "filename.pcd"
    with open(pcd_file, "rb") as a_file:
        surround3d_map = pickle.load(a_file)
    
    if isinstance(surround3d_map, cp.ndarray):
        surround3d_map = cp.asnumpy(surround3d_map)

    inlier_cloud = pcl.PointCloud()
    inlier_cloud.from_array(surround3d_map)
    inlier_cloud.to_file("bf.pcd".encode('utf-8'))

    img_path = "filename.jpg"
    a = cv2.imread(img_path)
    cv2.imwrite("bf.jpg", a)

    import pdb; pdb.set_trace()