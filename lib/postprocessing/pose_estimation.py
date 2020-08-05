import cv2
import numpy as np
import pandas as pd

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

from lib.handcraft_extractor.SIFT.SIFT import * 
from lib import datasets as db
from lib import extractor as ex

class SfM():
    def __init__(self, args, matcher="Brute-Force", ld="root_sift"):
 
        if args.pose_ld == 0: # root SIFT
            self.local_descriptor = SIFT(dense=args.pose_ld_dense, root=True)
        elif args.pose_ld == 1: # SIFT
            self.local_descriptor = SIFT(dense=args.pose_ld_dense, root=False)
        elif args.pose_ld == 2: # D2
            self.local_descriptor = ex.D2Net(model_file="./lib/extractor/D2Net/pretrained/d2_tf.pth", finetune=False)

        self.vSet_Views = {} #["ViewId", "points", "Orientation", "Location"])
        self.vSet_Connections = {} #["ViewId1", "ViewId2", "Matches", "RelativeOrientation", "RelatveLocation"]

        if args.pose_matcher == 0: # Brute-Force
            self.matcher = cv2.BFMatcher()
        elif args.pose_matcher == 1: # Flann-based
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else: 
            raise ValueError("check SfM params")
    
    def __initvset__(self, size):
        v = {"ViewId" : np.zeros((size)).astype("O"),
             "points" : np.zeros((size)).astype("O"),
             "Orientation" : np.zeros((size)).astype("O"),
             "Location" : np.zeros((size)).astype("O")}
        self.vSet_Views = pd.DataFrame(v)

        v = {"ViewId1" : np.zeros((size)).astype("O"),
             "ViewId2" : np.zeros((size)).astype("O"),
             "Matches" : np.zeros((size)).astype("O"),
             "RelativeOrientation" : np.zeros((size)).astype("O"),
             "RelatveLocation" : np.zeros((size)).astype("O")}
        self.vSet_Connections = pd.DataFrame(v)


    def __addvset__(self, i, pts=None, ori=None, loc=None):
        self.vSet_Views['ViewId'].iloc[i] = i
        self.vSet_Views['points'].iloc[i] = pts if pts is not None else None
        self.vSet_Views['Orientation'].iloc[i] = ori.astype("object") if ori is not None else None
        self.vSet_Views['Location'].iloc[i] = loc.astype("object") if loc is not None else None
        
    def __addConnection__(self, prev, curr, matches=None, relativeOri=None, relativeLoc=None):

        for ind, (v1,v2) in enumerate(zip(self.vSet_Connections["ViewId1"], self.vSet_Connections["ViewId2"])):
            if (v1==0) and (v2==0):
                break
        self.vSet_Connections['ViewId1'].iloc[ind] = prev
        self.vSet_Connections['ViewId2'].iloc[ind] = curr
        self.vSet_Connections['Matches'].iloc[ind] = matches.astype("object") if matches is not None else None
        self.vSet_Connections['RelativeOrientation'].iloc[ind] = relativeOri.astype("object") if relativeOri is not None else None
        self.vSet_Connections['RelatveLocation'].iloc[ind] = relativeLoc.astype("object") if relativeLoc is not None else None

    def pose_filter(self, args, img_dict, query_K):
        # img_dict : query, 0, 1, 2, 3, ...  (The smaller the index, the more similar)


        img_list = list(img_dict.keys())
        self.__initvset__(len(img_list))
        queryind = img_list.index("query")
        del img_list[queryind]

        topk = len(img_list)

        q_img = img_dict['query']

        if args.pose_ld == 2:
            pass
        else:
            resize_fn = db.Resize(args.image_size)
            grayscele_fn = db.Grayscale()
            q_img = resize_fn(q_img)
            q_img_gray = grayscele_fn(q_img)


            q_kps, q_desc = self.local_descriptor(q_img_gray)

        candidate_Enum = []
        for i, index_name in enumerate(img_list):
            cind_img = img_dict[index_name]
            if args.pose_ld == 2:
                imageset = {}
                imageset.update({"image1" : q_img})
                imageset.update({"image2" : cind_img})

                output = self.local_descriptor(imageset)
                output['dense_features1']


            cind_img_rs = resize_fn(cind_img)
            cind_img_gray = grayscele_fn(cind_img_rs)

            cind_kps, cind_desc = self.local_descriptor(cind_img_gray)

            try:
                matches = self.matcher.knnMatch(q_desc, cind_desc, k=2)
            except:
                matches = None
                good = []
                pts_q = []
                pts_cind = []

            if matches is None:
                pass
            else:
                good = []
                pts_q = []
                pts_cind = []
                for match_i,(m,n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
                        good.append(m)
                        pts_q.append(q_kps[m.queryIdx].pt)
                        pts_cind.append(cind_kps[m.trainIdx].pt)
            
            pts_q = np.asarray(pts_q)
            pts_cind = np.asarray(pts_cind)

            if np.asarray(good).shape[0] == 0:
                candidate_Enum.append(10000) # good matching not exist
                continue
            
            E, mask = cv2.findEssentialMat(pts_q, pts_cind, query_K, cv2.FM_RANSAC)


            if E is None:
                candidate_Enum.append(1000) # good matching is too insufficient
            else:
                candidate_Enum.append(E.shape[0])
 
        
        new_seq = np.argsort(np.array(candidate_Enum))

        return new_seq, np.array(candidate_Enum)[new_seq]



    def estimation(self, img_dict, query_K):
        img_list = list(img_dict.keys())
        self.__initvset__(len(img_list))
        queryind = img_list.index("query")
        del img_list[queryind]

        q_img = img_dict['query']
        q_img_gray = cv2.cvtColor(q_img, cv2.COLOR_RGB2GRAY)
        import pdb; pdb.set_trace()

        q_kps, q_desc = self.local_descriptor(q_img_gray)

        self.__addvset__(0, q_kps, np.eye(3), np.zeros((1,3)))

        for i, index_name in enumerate(img_list):
            cind_img = img_dict[index_name]
            cind_img_gray = cv2.cvtColor(cind_img, cv2.COLOR_RGB2GRAY)
            cind_kps, cind_desc = self.local_descriptor(cind_img_gray)

            matches = self.matcher.knnMatch(q_desc, cind_desc, k=2)

            good = []
            pts_q = []
            pts_cind = []
            for match_i,(m,n) in enumerate(matches):
                if m.distance < 0.3*n.distance:
                    good.append(m)
                    pts_q.append(q_kps[m.queryIdx].pt)
                    pts_cind.append(cind_kps[m.trainIdx].pt)

            pts_q = np.asarray(pts_q)
            pts_cind = np.asarray(pts_cind)

            E, mask = cv2.findEssentialMat(pts_q, pts_cind, query_K, cv2.FM_RANSAC)
            points, R, t, new_mask = cv2.recoverPose(E, pts_q, pts_cind, query_K)
 
            self.__addvset__(i+1, cind_kps)
            
            matches_inlier = np.concatenate((pts_q, pts_cind), axis=1)[mask.ravel()==1]
            self.__addConnection__(i, i+1, matches_inlier)

            prevOri = self.vSet_Views['Orientation'][i].astype("float64")
            prevLoc = self.vSet_Views['Location'][i].astype("float64")

            projMat_q = np.eye(4)[:3]
            projMat_cind = np.concatenate((R, t), axis=1)
            projPts_q = matches_inlier[:,:2]
            projPts_cind = matches_inlier[:,2:]
 
            projPts_4d = cv2.triangulatePoints(projMat_q, projMat_cind, projPts_q.T, projPts_cind.T)
            xyzPoints = (projPts_4d/projPts_4d[3])[:3].T
    
            import pdb; pdb.set_trace()
            

if __name__ == '__main__':
    
    img_dict = {}
    for i in range(3):
        locals()["img"+str(i)] = cv2.imread(str(i)+".png")
        locals()["img"+str(i)] = cv2.cvtColor(locals()["img"+str(i)], cv2.COLOR_BGR2RGB)
        if i==0:
            img_dict.update({"query":locals()["img"+str(i)]})
        else:
            img_dict.update({"img"+str(i):locals()["img"+str(i)]})

    sfm = SfM()
    sfm.estimation(img_dict)
    import pdb; pdb.set_trace()