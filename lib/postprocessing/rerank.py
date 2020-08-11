import numpy as np
import torch
import tqdm
import cv2

from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.transform import AffineTransform

from lib import metric as mt
from lib import datasets as db
from lib import extractor as ex
from lib import handcraft_extractor as he
from lib import postprocessing as pp
from lib import utils as u

class Rerank():
    def __init__(self, args):
        self.pose_ld = args.pose_ld
        self.rerank = args.rerank
        self.lmr_score = args.lmr_score
        
        if self.pose_ld == 0: # SIFT
            self.local_descriptor = he.SIFT(root=False)
        elif self.pose_ld == 1: # rootSIFT
            self.local_descriptor = he.SIFT(root=True)
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
        
        self.grayscele_fn = db.Grayscale()
    
    def __call__(self, args, topk_list, querydb, indexdb, query_dataset, index_dataset):
        if self.rerank==0:
            topk_newlist = self.local_match_rerank(args, topk_list, querydb, indexdb, query_dataset, index_dataset, thr=self.lmr_score)
        elif self.rerank==1:
            topk_newlist = self.pnp_rerank(args, topk_list, querydb, indexdb, query_dataset, index_dataset)

        return topk_newlist
    
    def pnp_rerank(self, args, topk_list, querydb, indexdb, query_dataset, index_dataset):
        pose_estimation = pp.PoseEstimation(args)
        topk_newlist, _ = pose_estimation.rerank(args, querydb, indexdb, query_dataset, index_dataset, topk_list)
        return topk_newlist

    def local_match_rerank(self, args, topk_list, querydb, indexdb, query_dataset, index_dataset, thr=0):
        if isinstance(querydb['feat'], np.ndarray):
            query_feats = querydb['feat']
            index_feats = indexdb['feat']
        else:
            query_feats = np.asarray(querydb['feat'])
            index_feats = np.asarray(indexdb['feat'])

        posk_list_all = []
        num_inlier_all = []


        iter_desc = "Rerank!"
        
        for qi, index_i_list in enumerate(tqdm.tqdm(topk_list, desc=iter_desc)):
            
       
            query_ind = querydb['index'][qi]
            if self.pose_ld==4:
                query_img = query_dataset.__loadimg__(query_ind, tflag=False, gray=True)['image']
            else:
                query_img = query_dataset.__loadimg__(query_ind, tflag=False)['image']  
 
            query_kps, query_desc, query_score, query_etc = self.local_extract(query_img)
            
            cnt = 0
            posk_list_sub = []
            num_inlier_sub = []
            for ni, neigh in enumerate(index_i_list):
                index_ind = indexdb['index'][neigh]
                if self.pose_ld==4:
                    index_img = index_dataset.__loadimg__(index_ind, tflag=False, name=False, gray=True)['image']
                else:
                    index_img = index_dataset.__loadimg__(index_ind, tflag=False, name=False)['image']

                index_kps, index_desc, index_score, index_etc = self.local_extract(index_img)

                num_inlier, _, _, conf = self.local_match(query_kps, query_desc, query_score, query_etc, \
                                                    index_kps, index_desc, index_score, index_etc)

                if conf is None:
                    num_inlier_sub.append(num_inlier)
                else:
                    num_inlier_sub.append((conf>=thr).sum())
                posk_list_sub.append(neigh)

            num_inlier_sub = np.asarray(num_inlier_sub)
            posk_list_sub = np.asarray(posk_list_sub)

            new_seq = np.argsort(-num_inlier_sub)
            num_inlier_sub = num_inlier_sub[new_seq]
            posk_list_sub = posk_list_sub[new_seq]

            num_inlier_all.append(num_inlier_sub)
            posk_list_all.append(posk_list_sub)
        
        posk_list_all = np.asarray(posk_list_all)
        num_inlier_all = np.asarray(num_inlier_all)

        topk_newlist = posk_list_all
        return topk_newlist


    
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
            im, inp, sc = ex.read_image(image, "cuda", [1024,960], 0, False)
            feat = self.local_descriptor({'image': inp})
            kps = feat['keypoints']
            desc = feat['descriptors']
            score = feat['scores']
            img_zip = {'img_tensor' : inp, \
                       'scale_factor' : sc}
        

        return kps, desc, score, img_zip


    def local_match(self, query_kps, query_desc, query_score, query_etc, \
                          index_kps, index_desc, index_score, index_etc, ):
        
        conf=None
        if (self.pose_ld==0) or (self.pose_ld==1):
            if (query_kps.shape[0]>=2) & (index_kps.shape[0]>=2):
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params,search_params)
                matches = matcher.knnMatch(query_desc, index_desc, k=2)

                pts_1 = []
                pts_2 = []

                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
                        pts_1.append(query_kps[m.queryIdx])
                        pts_2.append(index_kps[m.trainIdx])
                
                pts_query = np.asarray(pts_1)
                pts_index = np.asarray(pts_2)

                num_inlier = pts_query.shape[0]
            else:
                pts_query = None
                pts_index = None
                num_inlier = 0
            

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

            pts_query = query_kps[matches[:, 0],:]
            pts_index = index_kps[matches[:, 1],:]

            num_inlier = pts_query.shape[0]
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
            num_inlier = pts_query.shape[0]
            pts_query = pts_query * query_etc['scale_factor']
            pts_index = pts_index * index_etc['scale_factor']

        

        return num_inlier, pts_query, pts_index, conf