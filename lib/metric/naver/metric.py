import os
import json
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import euclidean
from pyquaternion import Quaternion

from lib.metric.naver.searching_alg import kNN,kNN_GPU_
from lib import postprocessing as pp


def LocDegThreshMetric(args, indexdb, querydb, index_dataset, query_dataset, epoch, save_path, place="indoor"):
    
    # [m, degree]
    if place is "indoor":
        near = [0.25, 10.0]
        medium = [0.5, 10.0]
        far = [5.0, 10.0]
    elif place is "outdoor":
        near = [0.5, 2.0]
        medium = [1.0, 5.0]
        far = [5.0, 10.0]
    else:
        raise ValueError('Check place!')
    
    num_pred = len(querydb['feat'])
    n_num = 0
    m_num = 0
    f_num = 0
   
    D,topk_list = kNN_GPU_(indexdb['feat'], querydb['feat'], args.topk)
    if args.test:
        tesk="test"
    else:
        test="valid"
    
    np.save("Ensemble_%s_index_%03d_%s_%s.npy"%(args.floor,args.topk,args.sf,tesk),topk_list)
    
    if args.test:
        return 
    if args.pose_filter is True:
        topk_list, topk_Enum = pose_filter(args, topk_list, querydb, query_dataset, index_dataset)
    
    cor_n_dict = {}
    cor_m_dict = {}
    cor_f_dict = {}
    all_dict = {}
    for tpki, topk_val in enumerate(topk_list):

        n_true=0
        m_true=0
        f_true=0

        nt_list=[]
        mt_list=[]
        ft_list=[]
        all_list=[]
        
        for i,val in enumerate(topk_val):
            
            idx_pose = indexdb['pose'][val]
            pred_pose = querydb['pose'][tpki]

            idx_xyz = idx_pose[:3]
            pred_xyz = pred_pose[:3]

            dist_diff = euclidean(idx_xyz, pred_xyz)

            idx_qtn = Quaternion(idx_pose[3:])
            pred_qtn = Quaternion(pred_pose[3:])

            deg_diff = (idx_qtn.conjugate*pred_qtn).degrees
            deg_diff = abs(deg_diff)

            all_list.append([dist_diff, deg_diff, val])

            if (near[0]>=dist_diff) and (near[1]>=deg_diff):
                nt_list.append(val)
                n_true+=1
            if (medium[0]>=dist_diff) and (medium[1]>=deg_diff):
                mt_list.append(val)
                m_true+=1
            if (far[0]>=dist_diff) and (far[1]>=deg_diff):
                ft_list.append(val)
                f_true+=1
        
        cor_n_dict.update({tpki : nt_list})
        cor_m_dict.update({tpki : mt_list})
        cor_f_dict.update({tpki : ft_list})
        all_dict.update({tpki : all_list})

        if n_true!=0:
            n_num+=1
        if m_true!=0:
            m_num+=1
        if f_true!=0:
            f_num+=1

    near_prct = n_num/num_pred *100
    medium_prct = m_num/num_pred *100
    far_prct = f_num/num_pred *100

    
    printscript = '{} m / {} degree : {:.02f} '.format(near[0], near[1], near_prct),\
                '{} m / {} degree : {:.02f}  '.format(medium[0], medium[1], medium_prct),\
                '{} m / {} degree : {:.02f} '.format(far[0], far[1], far_prct)
    

    filename = 'result_{}_epoch{}_top{}.txt'.format(place, epoch, args.topk)
    with open(os.path.join(save_path, filename), 'w') as f:
        json.dump(printscript, f, indent=2)

    near = list(map(str,near))
    medium = list(map(str,medium))
    far = list(map(str,far))

    near = "_".join(near)
    medium = "_".join(medium)
    far = "_".join(far)

    if args.qualitative is True:
        qualitative(args, topk_list, cor_n_dict, cor_m_dict, cor_f_dict, all_dict, save_path, place, epoch)

    return {near : near_prct, medium : medium_prct, far : far_prct}


def LocThreshMetric(args, indexdb, querydb, index_dataset, query_dataset, epoch, save_path, place="indoor"):
    
    # [m, degree]
    if place is "indoor":
        near = [0.25]
        medium = [0.5]
        far = [5.0]
    elif place is "outdoor":
        near = [0.5]
        medium = [1.0]
        far = [5.0]
    else:
        raise ValueError('Check place!')
    
    num_pred = len(querydb['feat'])
    n_num = 0
    m_num = 0
    f_num = 0
    #import pdb;pdb.set_trace()
    topk_list = kNN_GPU_(indexdb['feat'], querydb['feat'], args.topk)

    topk_Enum = None
    if args.pose_filter is True:
        topk_list, topk_Enum = pose_filter(args, topk_list, querydb, query_dataset, index_dataset)


    cor_n_dict = {}
    cor_m_dict = {}
    cor_f_dict = {}
    all_dict = {}
    for tpki, topk_val in enumerate(topk_list):

        n_true=0
        m_true=0
        f_true=0

        nt_list=[]
        mt_list=[]
        ft_list=[]
        all_list=[]
        
        for i,val in enumerate(topk_val):
            
            idx_pose = indexdb['pose'][val]
            pred_pose = querydb['pose'][tpki]

            idx_xyz = idx_pose[:3]
            pred_xyz = pred_pose[:3]
            
            dist_diff = euclidean(idx_xyz, pred_xyz)

            idx_qtn = Quaternion(idx_pose[3:])
            pred_qtn = Quaternion(pred_pose[3:])

            deg_diff = (idx_qtn.conjugate*pred_qtn).degrees
            deg_diff = abs(deg_diff)

            all_list.append([dist_diff, deg_diff, val])

            if (near[0]>=dist_diff):
                nt_list.append(val)
                n_true+=1
            if (medium[0]>=dist_diff):
                mt_list.append(val)
                m_true+=1
            if (far[0]>=dist_diff):
                ft_list.append(val)
                f_true+=1
        
        cor_n_dict.update({tpki : nt_list})
        cor_m_dict.update({tpki : mt_list})
        cor_f_dict.update({tpki : ft_list})
        all_dict.update({tpki : all_list})

        if n_true!=0:
            n_num+=1
        if m_true!=0:
            m_num+=1
        if f_true!=0:
            f_num+=1

    near_prct = n_num/num_pred *100
    medium_prct = m_num/num_pred *100
    far_prct = f_num/num_pred *100

    
    printscript = '{} m : {:.02f} '.format(near[0], near_prct),\
                '{} m : {:.02f}  '.format(medium[0], medium_prct),\
                '{} m : {:.02f} '.format(far[0], far_prct)
    

    filename = 'result_{}_epoch{}_top{}.txt'.format(place, epoch, args.topk)
    with open(os.path.join(save_path, filename), 'w') as f:
        json.dump(printscript, f, indent=2)

    if args.qualitative is True:
        qualitative(args, topk_list, topk_Enum, cor_n_dict, cor_m_dict, cor_f_dict, all_dict, save_path, place, epoch)

    return {str(near[0]) : near_prct, str(medium[0]) : medium_prct, str(far[0]) : far_prct}


def pose_filter(args, topk_list, querydb, query_dataset, index_dataset):

    sfm = pp.SfM(args)

    topk_Enum = np.zeros_like(topk_list)
    for i, row in enumerate(tqdm.tqdm(topk_list, desc="Pose filtering!")):
        img_dict = {}
        K_dict = {}

        query_ind = querydb['index'][i]
        if args.pose_ld == 2:
            query_img = query_dataset.__loadimg__(query_ind, tflag=True)['image'].unsqueeze(0).cuda()
        else:
            query_img = query_dataset.__loadimg__(query_ind, tflag=False)['image']
        img_dict.update({"query" : query_img})

        query_K, _, _ = query_dataset.__getcameraparams__(query_ind)

        for ri, r in enumerate(row):
            if args.pose_ld == 2:
                index_img = index_dataset.__loadimg__(r, tflag=True)['image'].unsqueeze(0).cuda()
            else:
                index_img = index_dataset.__loadimg__(r, tflag=False)['image']

            img_dict.update({"img"+str(ri) : index_img})
            
        new_row, Enum = sfm.pose_filter(args, img_dict, query_K)

        topk_list[i] = topk_list[i][new_row]
        topk_Enum[i] = Enum
    
    topk_list = topk_list[:,:args.posek]
    topk_Enum = topk_Enum[:,:args.posek]

    return topk_list, topk_Enum


def qualitative(args, topk_list, topk_Enum, cor_n_dict, cor_m_dict, cor_f_dict, all_dict, save_path, place, epoch):

    from lib import datasets as db

    if os.path.isdir(os.path.join(save_path, "qualitative_epoch{}".format(epoch))) is False:
        os.mkdir(os.path.join(save_path, "qualitative_epoch{}".format(epoch)))

    valid_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                        "./NaverML_indoor/b1/train/csv/v2/train_val/val_b1.csv")
            
    index_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                        "./NaverML_indoor/b1/train/csv/v2/train_val/train_b1.csv")
    

    rows = 2
    cols = math.ceil((topk_list.shape[1]+1)/2)

    for query_i in tqdm.tqdm(range(topk_list.shape[0])):
        fig = plt.figure(figsize=(19.20,10.80))
        st = fig.suptitle("Green=near, Yellow=medium, Orange=far, Red=miss", fontsize="x-large")
        st.set_y(0.95)

        query_img = valid_dataset.__loadimg__(query_i)['image']
        ax = fig.add_subplot(rows, cols, 1)
        ax.imshow(query_img)
        ax.set_title("query", fontsize = 10)
        ax.axis("off")
        stat = "find"
        cnt=0
        for index_i, index in enumerate(topk_list[query_i]):
            
            if index in cor_n_dict[query_i]:
                color=[0,255,0]
                cnt+=1
            elif index in cor_m_dict[query_i]:
                color=[255,255,0]
                cnt+=1
            elif index in cor_f_dict[query_i]:
                color=[255,127,0]
                cnt+=1
            else:
                color=[255,0,0]

            index_img = index_dataset.__loadimg__(index)['image']

            top, bottom, left, right = [150]*4
            index_img = cv2.copyMakeBorder(index_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            ax = fig.add_subplot(rows, cols, index_i+2)
            ax.imshow(index_img)

            data_curr = np.asarray(all_dict[query_i])
            if topk_Enum is None:
                title = "dist:{:.2f}_deg:{:.2f}".format(data_curr[index_i,0], data_curr[index_i,1])
            else:
                title = "dist:{:.2f}_deg:{:.2f}_E:{:d}".format(data_curr[index_i,0], data_curr[index_i,1], topk_Enum[query_i][index_i])
            ax.set_title(title)
            ax.axis("off")

        if cnt==0:
            stat = "miss"

        filename = 'qualitative_{}_epoch{}_top{}_{}_{}.png'.format(place, epoch, args.topk, query_i, stat)
        plt.savefig(os.path.join(save_path, "qualitative_epoch{}".format(epoch), filename))
        plt.close()

