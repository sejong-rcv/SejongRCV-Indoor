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

from scipy.spatial.transform import Rotation 
from scipy.spatial.distance import euclidean
from pyquaternion import Quaternion

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.transform import AffineTransform


from lib import datasets as db
from lib import extractor as ex
from lib import metric as mt
from lib import handcraft_extractor as he
from lib import postprocessing as pp
from lib import utils as u


class PoseEstimation():
    def __init__(self, args, root_folder="./NaverML_indoor"):

        self.pose_ld = args.pose_ld
        self.pose_covisibility = args.pose_covisibility
        self.pose_pointcloud_load = args.pose_pointcloud_load
        self.dataset = args.dataset
        self.pose_timechecker = args.pose_timechecker
        self.pose_noniter = args.pose_noniter
        self.pose_cupy = args.pose_cupy

        if self.pose_ld == 0: # root SIFT
            self.local_descriptor = he.SIFT(root=True)
        elif self.pose_ld == 1: # SIFT
            self.local_descriptor = he.SIFT(root=False)
        elif self.pose_ld == 2: # D2 SS
            self.local_descriptor = ex.D2Net_local_extractor(args=args, 
                                                             model_file="./arxiv/d2_tf.pth",
                                                             use_relu=True,
                                                             max_edge=1600,
                                                             max_sum_edges=2800,
                                                             preprocessing='caffe',
                                                             multiscale=False)
        elif self.pose_ld == 3: # D2 MS
            self.local_descriptor = ex.D2Net_local_extractor(args=args, 
                                                             model_file="./arxiv/d2_tf.pth",
                                                             use_relu=True,
                                                             max_edge=1600,
                                                             max_sum_edges=2800,
                                                             preprocessing='caffe',
                                                             multiscale=True)
        elif self.pose_ld == 4: # SuperGlue
            superPointdict=dict()
            superPointdict["nms_radius"]=4
            superPointdict["keypoint_threshold"]= 0.005
            superPointdict["max_keypoints"]=1024
            self.local_descriptor = ex.SuperPoint(superPointdict).eval().cuda()

        if self.dataset==0:
                self.cld_path = "img2pcds/indoor_b1"
        elif self.dataset==1:
                self.cld_path = "img2pcds/indoor_1f"

        if self.pose_pointcloud_load is False:
            if self.dataset==0:
                self.cloud = pcl.load(os.path.join(root_folder, "b1", "train", "PointCloud_all", "map.pcd"))
                self.cld_arr = self.cloud.to_array()
                mask = (self.cld_arr[:,2]>-5.8) & (self.cld_arr[:,2]<0)
                self.cld_arr = self.cld_arr[mask]
                self.cloud = pcl.PointCloud()
                self.cloud.from_array(self.cld_arr)           
                resolution = 0.2
                self.octree = self.cloud.make_octreeSearch(resolution)
                self.octree.add_points_from_input_cloud()

            elif self.dataset==1:
                self.cloud = pcl.load(os.path.join(root_folder, "1f", "train", "PointCloud_all", "map.pcd"))
                self.cld_arr = self.cloud.to_array()
                mask = (self.cld_arr[:,2]>0)
                self.cld_arr = self.cld_arr[mask]
                self.cloud = pcl.PointCloud()
                self.cloud.from_array(self.cld_arr)  
                resolution = 0.2
                self.octree = self.cloud.make_octreeSearch(resolution)
                self.octree.add_points_from_input_cloud()        
            else:
                raise ValueError("check PoseEstimation params")


        self.grayscele_fn = db.Grayscale()

    def local_extract(self, image):
        
        img_zip = None
        score = None
        if (self.pose_ld==0) or (self.pose_ld==1):
            image = self.grayscele_fn(image)
            kps, desc = self.local_descriptor(image)
            kps = cv2.KeyPoint_convert(kps)

        elif (self.pose_ld==2) or (self.pose_ld==3):
            feat = self.local_descriptor.extract(image)
            kps = feat["keypoints"][:,:2]
            desc = feat['descriptors']
        
        elif self.pose_ld==4:
            rescale_size = [1024,960]
            im, inp, sc = ex.read_image(image, "cuda", rescale_size, 0, False)
            feat = self.local_descriptor({'image': inp})
            kps = feat['keypoints']
            desc = feat['descriptors']
            score = feat['scores']
            img_zip = {'img_tensor' : inp, \
                       'scale_factor' : sc}

        return kps, desc, score, img_zip
    
    def local_match(self, query_kps, query_desc, query_score, query_etc, \
                          index_kps, index_desc, index_score, index_etc, \
                          pts_cloud=None):
        
        conf=None
        if pts_cloud is None:
            pts_3d=None
            
        if (self.pose_ld==0) or (self.pose_ld==1):
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params,search_params)
            matches = matcher.knnMatch(query_desc, index_desc, k=2)

            pts_1 = []
            pts_2 = []
            pts_3d = []

            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    pts_1.append(query_kps[m.queryIdx])
                    pts_2.append(index_kps[m.trainIdx])
                    if pts_cloud is not None:
                        pts_3d.append(pts_cloud[m.trainIdx])
            
            pts_query = np.asarray(pts_1)
            pts_index = np.asarray(pts_2)
            if pts_cloud is not None:
                pts_3d = np.asarray(pts_3d)

            inq = np.unique(pts_query, axis=0, return_index=True)[1]
            ini = np.unique(pts_index, axis=0, return_index=True)[1]
            if pts_cloud is not None:
                in3 = np.unique(pts_3d, axis=0, return_index=True)[1]
            interin = np.intersect1d(np.intersect1d(ini, in3), in3)
            
            pts_query = pts_query[interin]
            pts_index = pts_index[interin]
            if pts_cloud is not None:
                pts_3d = pts_3d[interin]

            model, inliers = ransac(
                (pts_query, pts_index),
                AffineTransform, min_samples=3,
                residual_threshold=20, max_trials=1000
            )

        elif (self.pose_ld==2) or (self.pose_ld==3):
            
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
            if pts_cloud is not None:
                pts_3d = pts_cloud[matches[:, 1],:]
            np.random.seed(0)

            model, inliers = ransac(
                (kps_left, kps_right),
                AffineTransform, min_samples=3,
                residual_threshold=20, max_trials=1000
            )

            pts_query = kps_left[inliers]
            pts_index = kps_right[inliers]
            if pts_cloud is not None:
                pts_3d = pts_3d[inliers]
            
            knn.delete()
            del knn

        elif self.pose_ld==4:
            superGluedict=dict()
            superGluedict["weights"]="indoor"
            superGluedict["sinkhorn_iterations"]= 20
            superGluedict["match_threshold"]=0.02

            superglue_matcher = ex.SuperGlue(superGluedict).eval().cuda()
            data = {"image0" : query_etc['img_tensor'], \
                    "image1" : index_etc['img_tensor']}
            
            pred = {}
            pred0 = {"keypoints" : query_kps, \
                     "descriptors" : query_desc, \
                     "scores" : query_score}
            pred1 = {"keypoints" : index_kps, \
                     "descriptors" : index_desc, \
                     "scores" : index_score}
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

            data = {**data, **pred}
            for k in data:
                if isinstance(data[k], (list, tuple)):
                    data[k] = torch.stack(data[k])
            
            pred = {**pred, **superglue_matcher(data)}
            pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
            
            matches = pred['matches0']

            valid = matches >-1
            conf = pred['matching_scores0'][valid]

            pts_query = query_kps[0].cpu().detach().numpy()[valid]
            pts_index = index_kps[0].cpu().detach().numpy()[matches[valid]]
            if pts_cloud is not None:
                pts_3d = pts_cloud[matches[valid]]
            pts_query = pts_query * query_etc['scale_factor']
            pts_index = pts_index * index_etc['scale_factor']
           
            model, inliers = ransac(
                (pts_query, pts_index),
                AffineTransform, min_samples=3,
                residual_threshold=20, max_trials=1000
            )

            pts_query = pts_query[inliers]
            pts_index = inliers[inliers]
            if pts_cloud is not None:
                pts_3d = pts_3d[inliers]
            
        

        return pts_query, pts_index, pts_3d, conf


    def estimation(self, args, querydb, indexdb, query_dataset, index_dataset, topk_list, search_radius=15):

        qdiff_xyz = []
        qdiff_qwxyz = []

        idiff_xyz = []
        idiff_qwxyz = []

        q_xyz = []
        q_qwxyz = []

        g_xyz = []
        g_deg = []
        for qi, kList in enumerate(tqdm.tqdm(topk_list, desc="Estimating Pose!")):

            start = tm.time()
            query_ind = querydb['index'][qi]

            if (self.pose_ld==4):
                query_img = query_dataset.__loadimg__(query_ind, tflag=False, gray=True)['image']
            else:
                query_img = query_dataset.__loadimg__(query_ind, tflag=False)['image']
            query_K, query_rdist, query_tdist = query_dataset.__getcameraparams__(query_ind)
            query_dist = np.asarray([query_rdist[0], query_rdist[1], query_tdist[0], query_tdist[1], query_rdist[2]])

            if self.pose_timechecker: 
                print("Query load {}s".format(tm.time()-start))
                start = tm.time()
            query_kps, query_desc, query_score, query_etc = self.local_extract(query_img)
            
            if self.pose_timechecker:
                print("Query extract {}s".format(tm.time()-start))
                start = tm.time()


            

            for ni, neigh in enumerate(kList):
                
                pts_query_set = []
                pts_3d_set = []

                neighs = [neigh]
                
                if self.pose_covisibility:
                    neighs.append(neigh-self.pose_covisibility) if (neigh!=0) else True
                    neighs.append(neigh+self.pose_covisibility) if (neigh!=len(index_dataset)-1) else True

                for nni, nneigh in enumerate(neighs):

                    index_xyz = np.asarray(indexdb['pose'][nneigh][:3])
                    index_qwxyz = np.asarray(indexdb['pose'][nneigh][3:])

                    index_ind = indexdb['index'][nneigh]
                    if (self.pose_ld==4):
                        index_data = index_dataset.__loadimg__(index_ind, tflag=False, name=True, gray=True)
                    else:
                        index_data = index_dataset.__loadimg__(index_ind, tflag=False, name=True)
                    
                    index_img = index_data['image']
                    index_name = index_data['name']

                    index_K, index_rdist, index_tdist = index_dataset.__getcameraparams__(index_ind)
                    index_dist = np.asarray([index_rdist[0], index_rdist[1], index_tdist[0], index_tdist[1], index_rdist[2]])
                    
                    if self.pose_timechecker:
                        print("Index load {}s".format(tm.time()-start))
                        start = tm.time()
                    index_kps, index_desc, index_score, index_etc = self.local_extract(index_img)

                    if self.pose_timechecker:
                        print("Index extract {}s".format(tm.time()-start))
                        start = tm.time()
                    name_spliter = index_name.split("/")
                    pcd_file = os.path.join(self.cld_path,name_spliter[0], name_spliter[1], "img2pcd", name_spliter[2]+".pkl")
                    
                    if self.pose_pointcloud_load is True:
                        with open(pcd_file, "rb") as a_file:
                            surround3d_map = pickle.load(a_file)
                        
                        if self.pose_cupy is True:
                            if isinstance(surround3d_map, np.ndarray):
                                surround3d_map = cp.asarray(surround3d_map)
                        else:    
                            if isinstance(surround3d_map, cp.ndarray):
                                surround3d_map = cp.asnumpy(surround3d_map)

                        if self.pose_timechecker:
                            print("3D points load {}s".format(tm.time()-start))
                            start = tm.time()
                        surround2d_map, Rm = self.__world2image__(index_qwxyz, index_xyz, index_K, surround3d_map)
                        if isinstance(surround3d_map, cp.ndarray):
                            surround3d_map = cp.asnumpy(surround3d_map)
                        if self.pose_timechecker:
                            print("3D points processing {}s".format(tm.time()-start))
                            start = tm.time()
                        
                    else:
                        searchPoints = tuple(index_xyz)
                        [ind, sqdist] = self.octree.radius_search(searchPoints, search_radius) # (pts, radius, topk)
                        sub_cloud_array = self.cld_arr[ind]

                        image_pixel, Rm = self.__world2image__(index_qwxyz, index_xyz, index_K, sub_cloud_array)
                        maskInlier = self.__maskPlaneInlier__(image_pixel, index_K)
                        inlier3d_map = self.__applyMask__(sub_cloud_array, maskInlier)
                        inlier2d_map = self.__applyMask__(image_pixel, maskInlier)

                        img_mask = self.__surroundingMask__(inlier3d_map, index_xyz, index_qwxyz, alpha=search_radius)
                        surround3d_map = self.__applyMask__(inlier3d_map, img_mask)
                        surround2d_map = self.__applyMask__(inlier2d_map, img_mask)

                        if self.pose_timechecker:
                            print("3D points calcurate {}s".format(tm.time()-start))
                            start = tm.time()

                    if isinstance(index_kps, list):
                        if (self.pose_ld==4):
                            converted_kps = index_kps[0].cpu().detach().numpy() * index_etc['scale_factor']
                        else:
                            converted_kps = index_kps
                        alloc3d, residual_mask = self.__allocate3Dpoints__(converted_kps, surround2d_map, surround3d_map)
                    else:
                        alloc3d, residual_mask = self.__allocate3Dpoints__(index_kps, surround2d_map, surround3d_map)

                    
                    if self.pose_timechecker:
                        print("3D points allocate {}s".format(tm.time()-start))
                        start = tm.time()
                    

                    pts_query, pts_index, pts_3d, _ = self.local_match(query_kps, query_desc, query_score, query_etc, \
                                                                        index_kps, index_desc, index_score, index_etc, \
                                                                        alloc3d)
                    
                    if self.pose_timechecker:
                        print("Local matching {}s".format(tm.time()-start))
                        start = tm.time()

                    pts_query_set.extend(pts_query)
                    pts_3d_set.extend(pts_3d)
                pts_query_set = np.asarray(pts_query_set)
                pts_3d_set = np.asarray(pts_3d_set)

                try:
                    query_xyz, query_qwxyz, inlierR = self.__solvePnP__(pts_query_set, pts_3d_set, query_K)

                    if self.pose_timechecker:
                        print("PnP solve {}s".format(tm.time()-start))
                        start = tm.time()
                    if True:
                        break
                except:
                    if self.pose_noniter:
                        query_xyz, query_qwxyz = index_xyz, index_qwxyz
                        break

                    if self.pose_timechecker:
                        print("Next candidate {}/{}".format(ni,len(kList)))
                        start = tm.time()
                    continue
            
            try:
                q_xyz.append(query_xyz)
                q_qwxyz.append(query_qwxyz)
                del query_xyz, query_qwxyz
            except:
                q_xyz.append(indexdb['pose'][kList[0]][:3])
                q_qwxyz.append(indexdb['pose'][kList[0]][3:])

        q_xyz = np.asarray(q_xyz)
        q_qwxyz = np.asarray(q_qwxyz)



        return q_xyz, q_qwxyz

    def rerank(self, args, querydb, indexdb, query_dataset, index_dataset, topk_list, search_radius=15):

        qdiff_xyz = []
        qdiff_qwxyz = []

        idiff_xyz = []
        idiff_qwxyz = []

        q_xyz = []
        q_qwxyz = []

        g_xyz = []
        g_deg = []

        num_inlier_all = []
        topk_newlist = np.zeros_like(topk_list)
        for qi, kList in enumerate(tqdm.tqdm(topk_list, desc="Estimating Pose for rerank!")):
            
            start = tm.time()
            query_ind = querydb['index'][qi]

            if (self.pose_ld==4):
                query_img = query_dataset.__loadimg__(query_ind, tflag=False, gray=True)['image']
            else:
                query_img = query_dataset.__loadimg__(query_ind, tflag=False)['image']
            query_K, query_rdist, query_tdist = query_dataset.__getcameraparams__(query_ind)
            query_dist = np.asarray([query_rdist[0], query_rdist[1], query_tdist[0], query_tdist[1], query_rdist[2]])

            query_kps, query_desc, query_score, query_etc = self.local_extract(query_img)
        

            num_inlier_sub = []
            for ni, neigh in enumerate(kList):
                try:
                    index_xyz = np.asarray(indexdb['pose'][neigh][:3])
                    index_qwxyz = np.asarray(indexdb['pose'][neigh][3:])

                    i_xyz = np.asarray(indexdb['pose'][kList[0]][:3])
                    i_qwxyz = np.asarray(indexdb['pose'][kList[0]][3:])


                    index_ind = indexdb['index'][neigh]
                    if (self.pose_ld==4):
                        index_data = index_dataset.__loadimg__(index_ind, tflag=False, name=True, gray=True)
                    else:
                        index_data = index_dataset.__loadimg__(index_ind, tflag=False, name=True)
                    
                    index_img = index_data['image']
                    index_name = index_data['name']

                    index_K, index_rdist, index_tdist = index_dataset.__getcameraparams__(index_ind)
                    index_dist = np.asarray([index_rdist[0], index_rdist[1], index_tdist[0], index_tdist[1], index_rdist[2]])
                    index_kps, index_desc, index_score, index_etc = self.local_extract(index_img)

                    name_spliter = index_name.split("/")
                    pcd_file = os.path.join(self.cld_path,name_spliter[0], name_spliter[1], "img2pcd", name_spliter[2]+".pkl")
                    
                    if self.pose_pointcloud_load is True:
                        with open(pcd_file, "rb") as a_file:
                            surround3d_map = pickle.load(a_file)
                        if self.pose_cupy is True:
                            if isinstance(surround3d_map, np.ndarray):
                                surround3d_map = cp.asarray(surround3d_map)
                        else:    
                            if isinstance(surround3d_map, cp.ndarray):
                                surround3d_map = cp.asnumpy(surround3d_map)
        
                        surround2d_map, Rm = self.__world2image__(index_qwxyz, index_xyz, index_K, surround3d_map)
                        if isinstance(surround3d_map, cp.ndarray):
                            surround3d_map = cp.asnumpy(surround3d_map)
                    else:
                        searchPoints = tuple(index_xyz)
                        [ind, sqdist] = self.octree.radius_search(searchPoints, search_radius) # (pts, radius, topk)
                        sub_cloud_array = self.cld_arr[ind]

                        image_pixel, Rm = self.__world2image__(index_qwxyz, index_xyz, index_K, sub_cloud_array)
                        maskInlier = self.__maskPlaneInlier__(image_pixel, index_K)
                        inlier3d_map = self.__applyMask__(sub_cloud_array, maskInlier)
                        inlier2d_map = self.__applyMask__(image_pixel, maskInlier)

                        img_mask = self.__surroundingMask__(inlier3d_map, index_xyz, index_qwxyz, alpha=search_radius)
                        surround3d_map = self.__applyMask__(inlier3d_map, img_mask)
                        surround2d_map = self.__applyMask__(inlier2d_map, img_mask)

                    if isinstance(index_kps, list):
                        if (self.pose_ld==4):
                            converted_kps = index_kps[0].cpu().detach().numpy() * index_etc['scale_factor']
                        else:
                            converted_kps = index_kps
                        alloc3d, residual_mask = self.__allocate3Dpoints__(converted_kps, surround2d_map, surround3d_map)
                    else:
                        alloc3d, residual_mask = self.__allocate3Dpoints__(index_kps, surround2d_map, surround3d_map)

                    pts_query, pts_index, pts_3d, _ = self.local_match(query_kps, query_desc, query_score, query_etc, \
                                                                       index_kps, index_desc, index_score, index_etc, \
                                                                       alloc3d)

                    query_xyz, query_qwxyz, inlierR = self.__solvePnP__(pts_query, pts_3d, query_K)

                    num_inlier_sub.append(inlierR.shape[0])

                except:
                    num_inlier_sub.append(-1)

            num_inlier_sub = np.asarray(num_inlier_sub)
            new_seq = np.argsort(-num_inlier_sub)
            num_inlier_sub = num_inlier_sub[new_seq]
            topk_newlist[qi] = kList[new_seq]
            num_inlier_all.append(num_inlier_sub)
        
        num_inlier_all = np.asarray(num_inlier_all)

        return topk_newlist, num_inlier_all


    def __homogeneousCoord__(self, pts):
        if self.pose_cupy is False:
            ax = 1 if pts.shape[0]>pts.shape[1] else 0
            ones = np.ones((pts.shape[0],1)) if ax==1 else np.ones((1, pts.shape[0]))
            return np.concatenate((pts, ones), axis=ax)
        else:
            ax = 1 if pts.shape[0]>pts.shape[1] else 0
            ones = cp.ones((pts.shape[0],1)) if ax==1 else cp.ones((1, pts.shape[0]))
            return cp.concatenate((pts, ones), axis=ax)

    def __hnormalized__(self, pts):
        if self.pose_cupy is False:
            src = pts.T if pts.shape[0]<pts.shape[1] else pts
            dst = (src/src[:,2:])[:,:2]
            return dst
        else:
            src = pts.T if pts.shape[0]<pts.shape[1] else pts
            dst = (src/src[:,2:])[:,:2]
            return dst
    
    def __world2image__(self, qwxyz, t, K, point3d):
   
        if self.pose_cupy is False:
            I = np.eye(4)
            I[:3, 3] = t
            quat = Quaternion(qwxyz)
            Rm = quat.rotation_matrix
            Rm_inv = np.linalg.inv(Rm)
            I[:3, :3] = Rm
            Iinv = np.linalg.inv(I)
            hpts3d = self.__homogeneousCoord__(point3d).T
            point3d_local = np.matmul(Iinv, hpts3d)[0:3,:]
            image_pixel = self.__hnormalized__(np.matmul(K, point3d_local).T)
            
        else:
            I = cp.eye(4)
            I[:3, 3] = cp.asarray(t)
            quat = Quaternion(qwxyz)
            Rm = cp.asarray(quat.rotation_matrix)
            Rm_inv = cp.linalg.inv(Rm)
            Rm_inv = cp.asnumpy(Rm_inv)
            I[:3, :3] = cp.asarray(Rm)
            Iinv = cp.linalg.inv(I)
            hpts3d = self.__homogeneousCoord__(point3d, self.pose_cupy).T
            point3d_local = cp.matmul(Iinv, hpts3d)[0:3,:]
            image_pixel = self.__hnormalized__(cp.matmul(cp.asarray(K), point3d_local).T, self.pose_cupy)
            image_pixel = cp.asnumpy(image_pixel)

        return image_pixel, Rm_inv
    
    def __maskPlaneInlier__(self, pts, K, factor=0):
        if pts.shape[0]>pts.shape[1]:
            xmask = (pts[:,0]>=0) & (pts[:,0]<K[0,2]*2)
            ymask = (pts[:,1]>=0) & (pts[:,1]<K[1,2]*2)
        else:
            xmask = (pts[0,:]>=0) & (pts[0,:]<K[0,2]*2)
            ymask = (pts[1,:]>=0) & (pts[1,:]<K[1,2]*2)

        total_mask = xmask & ymask
        return total_mask

    def __applyMask__(self, pts, mask, K=None):
        inlier = pts[mask]
        if K is not None:
            inlier[:,0] = inlier[:,0]+K[0,2]
            inlier[:,1] = inlier[:,1]+K[1,2]
        return inlier

    def __surroundingMask__(self, inlier3d_map, ixyz, qwxyz, alpha=15):
    
        qm = Quaternion(qwxyz).rotation_matrix
        qmi = np.linalg.inv(qm)
        
        rotated3d_map = np.matmul(qmi, (inlier3d_map-ixyz).T).T
        rotated3d_ixyz = np.matmul(qmi, (ixyz-ixyz).T).T

        img_mask_x = (rotated3d_map[:,0]<alpha) & (rotated3d_map[:,0]>-alpha)
        img_mask_z = (rotated3d_map[:,2]<alpha) & (rotated3d_map[:,2]>0)
        img_mask = img_mask_x&img_mask_z
        
        return img_mask
    
    def __allocate3Dpoints__(self, kps_2d, surround2d_map, surround3d_map):
        knn = mt.kNN_GPU(d=surround2d_map.shape[1],GPU=True, GPU_Number=torch.cuda.current_device())
        knn.train(np.ascontiguousarray(surround2d_map, dtype=np.float32))
        topk_list = knn.predict(np.ascontiguousarray(kps_2d, dtype=np.float32), 1)
        cld_3d = surround3d_map[topk_list].squeeze(1)
        correspond_2d = surround2d_map[topk_list].squeeze(1)
        residual = np.linalg.norm((correspond_2d-kps_2d),axis=1)
        res_thr = residual.mean() + residual.std()
        residual_mask = residual<res_thr

        knn.delete()
        del knn
        return cld_3d, residual_mask

    def __solvePnP__(self, pair_q, pair3d, K, dst=[0,0,0,0,0]):
        _,solvR,solvt,inlierR = cv2.solvePnPRansac(np.expand_dims(pair3d, axis=2).astype("float64"), \
                                                np.expand_dims(pair_q, axis=2).astype("float64"), \
                                                K.astype("float64"), \
                                                np.expand_dims(np.array(dst), axis=1).astype("float64"), \
                                                iterationsCount=100000, \
                                                useExtrinsicGuess = True, \
                                                confidence = 0.999, \
                                                reprojectionError = 8, \
                                                flags = cv2.SOLVEPNP_AP3P) #0.8

        solvRR,_ = cv2.Rodrigues(solvR)
        solvRR_inv = np.linalg.inv(solvRR)
        solvtt = -np.matmul(solvRR_inv,solvt)
        
        rot = cv2.Rodrigues(solvRR_inv)[0].squeeze(1)
        query_qwxyz = Quaternion(matrix=solvRR_inv).elements
        query_xyz = solvtt.squeeze(1)

        return query_xyz, query_qwxyz, inlierR

