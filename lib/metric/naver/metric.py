import os
import json
import math
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pickle

from scipy.spatial.distance import euclidean
from pyquaternion import Quaternion

from lib import postprocessing as pp
from lib import metric as mt
from lib import utils as u

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


    if args.topk_load is not None:
        topk_path = os.path.join("./arxiv", args.topk_load)
        newk_list = np.load(topk_path)
        topk_list = newk_list[:topk_list.shape[0], :topk_list.shape[1]]

    else:
        if args.searching==0:
            topk_list = mt.kNN(indexdb['feat'], querydb['feat'], args.topk)
        elif args.searching==1:
            knn = mt.kNN_GPU(d=indexdb['feat'].shape[1], GPU=True, GPU_Number=torch.cuda.current_device())
            knn.train(np.ascontiguousarray(indexdb['feat'], dtype=np.float32))
            topk_list = knn.predict(np.ascontiguousarray(querydb['feat'], dtype=np.float32), args.topk)
            knn.delete()
            del knn
        elif args.searching==2:
            diffusion = pp.Diffusion_search(querydb['feat'], indexdb['feat'], trancation_size=1000)
            topk_list = diffusion.search()
            topk_list = topk_list[:, :args.topk]

        if args.qe is not None:
            query_expansion = pp.QueryExpansion(args)
            topk_list = query_expansion(args, topk_list, querydb, indexdb, query_dataset, index_dataset)

        if args.rerank is not None:
            rerank = pp.Rerank(args)
            topk_list = rerank(args, topk_list, querydb, indexdb, query_dataset, index_dataset)
        
                
        if args.topk_save is not None:
            topk_path = os.path.join("./arxiv", args.topk_save)
            np.save(topk_path, topk_list)

    

    q_xyz = None
    q_qwxyz = None
    if args.pose_estimation is True:
        pose_estimation = pp.PoseEstimation(args)
        q_xyz, q_qwxyz = pose_estimation.estimation(args, querydb, indexdb, query_dataset, index_dataset, topk_list)

    if args.test is True:

        if q_xyz is None:
            q_xyz = indexdb['pose'][topk_list[:,0]][:,:3]
            q_qwxyz = indexdb['pose'][topk_list[:,0]][:,3:]
        
        submitform(args, q_xyz, q_qwxyz, query_dataset, index_dataset, save_path)
        return


    cor_n_dict = {}
    cor_m_dict = {}
    cor_f_dict = {}
    all_dict = {}
    iall_dict = {}

    for tpki, topk_val in enumerate(topk_list):

        n_true=0
        m_true=0
        f_true=0

        nt_list=[]
        mt_list=[]
        ft_list=[]
        all_list=[]
        iall_list = []
        
        for i,val in enumerate(topk_val):
            
            pred_pose = indexdb['pose'][val]
            gt_pose = querydb['pose'][tpki]

            pred_xyz = pred_pose[:3]
            gt_xyz = gt_pose[:3]

            pred_qtn = Quaternion(pred_pose[3:])
            gt_qtn = Quaternion(gt_pose[3:])
            
            if args.pose_estimation is True:
                if i==0:
                    index_xyz = pred_pose[:3]
                    index_qtn = Quaternion(pred_pose[3:])
                    pred_xyz = q_xyz[tpki].tolist()
                    pred_qtn = Quaternion(q_qwxyz[tpki].tolist())
                else:
                    break

            dist_diff = euclidean(gt_xyz, pred_xyz)

            deg_diff = (gt_qtn.conjugate*pred_qtn).degrees
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

            if args.pose_estimation is True:
                idist_diff = euclidean(gt_xyz, index_xyz)
                ideg_diff = (gt_qtn.conjugate*index_qtn).degrees
                ideg_diff = abs(ideg_diff)
                iall_list.append([idist_diff, ideg_diff, val])
                break
        
        
        cor_n_dict.update({tpki : nt_list})
        cor_m_dict.update({tpki : mt_list})
        cor_f_dict.update({tpki : ft_list})
        all_dict.update({tpki : all_list})
        iall_dict.update({tpki : iall_list})

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
        qualitative(args, topk_list, cor_n_dict, cor_m_dict, cor_f_dict, all_dict, iall_dict, save_path, place, epoch)
    
    return {near : near_prct, medium : medium_prct, far : far_prct}


def submitform(args, xyz, qwxyz, query_dataset, index_dataset, save_path):

    if args.dataset==2:
        floor='b1'
    elif args.dataset==3:
        floor='1f'
    else:
        raise ValueError("Not support yet!")
    all_predict = []
    for i in range(len(query_dataset)):
        query_inform = query_dataset.__loadimg__(i, name=True ,image=False, tflag=False)
        img_name = query_inform.split('/')[2]+'.jpg'
        all_predict.append({'floor' : floor, \
                             'name' : img_name, \
                               'qw' : qwxyz[i][0], \
                               'qx' : qwxyz[i][1], \
                               'qy' : qwxyz[i][2], \
                               'qz' : qwxyz[i][3], \
                                'x' : xyz[i][0], \
                                'y' : xyz[i][1], \
                                'z' : xyz[i][2]})
    
    filename = 'result_{}_submit.json'.format(floor)
    with open(os.path.join(save_path, filename), 'w') as f:
        json.dump(all_predict, f, indent=1)



def qualitative(args, topk_list, cor_n_dict, cor_m_dict, cor_f_dict, all_dict, iall_dict, save_path, place, epoch):

    from lib import datasets as db

    if os.path.isdir(os.path.join(save_path, "qualitative_epoch{}".format(epoch))) is False:
        os.mkdir(os.path.join(save_path, "qualitative_epoch{}".format(epoch)))


    if (args.valid is True) and (args.valid_sample is False):
        valid_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                            "./NaverML_indoor/b1/train/csv/v2/train_val/val_b1.csv")
    elif (args.valid is False) and (args.valid_sample is True):
        valid_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                            "./NaverML_indoor/b1/train/csv/v2/train_val/val_b1_sample10.csv")
            
    index_dataset = db.Load_NMLC_indoor("./NaverML_indoor/b1/train/", \
                                        "./NaverML_indoor/b1/train/csv/v2/train_val/train_b1.csv")
    
    
    rows = 2
    cols = math.ceil((topk_list.shape[1]+1)/2)
    if args.pose_estimation is True:
        rows = 1
        cols = 2

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
            ax.axis("off")
            if args.pose_estimation is True:
                if index_i==0:
                    idata_curr = np.asarray(iall_dict[query_i])
                    title = "predict[{}] : {:.2f}(m)/{:.2f}(deg)  ||  index[{}] : {:.2f}(m)/{:.2f}(deg)".format(data_curr[index_i,2], data_curr[index_i,0], data_curr[index_i,1], idata_curr[index_i,2], idata_curr[index_i,0], idata_curr[index_i,1])
                    ax.set_title(title)
                    break
            else:
                title = "dist:{:.2f}_deg:{:.2f}".format(data_curr[index_i,0], data_curr[index_i,1])
                ax.set_title(title)
            

        if cnt==0:
            stat = "miss"

        filename = 'qualitative_{}_epoch{}_top{}_{}_{}.png'.format(place, epoch, args.topk, query_i, stat)
        plt.savefig(os.path.join(save_path, "qualitative_epoch{}".format(epoch), filename))
        plt.close()
    

    if args.pose_estimation is True:
        dist=[] # predict, index
        deg=[] # predict, index
        for i in range(len(all_dict)):
            pdata = all_dict[i][0]
            idata = iall_dict[i][0]
            dist.append([pdata[0], idata[0]])
            deg.append([pdata[1], idata[1]])

        dist = np.asarray(dist)
        deg = np.asarray(deg)

        positive_mask = (dist[:, 0] <= 5) & (deg[:, 0] <= 10)

        pos_dist = dist[positive_mask]
        pos_deg = deg[positive_mask]

        neg_dist = dist[~positive_mask]
        neg_deg = deg[~positive_mask]

        #--positive
        fig = plt.figure(figsize=(20, 20))
        scat_dist = plt.scatter(pos_dist[:,0], pos_dist[:,1], s=10)
        plt.xlabel('predict positive dist(m)', fontsize=20)
        plt.ylabel('index positive dist(m)', fontsize=20)
        plt.grid(True, axis='x', color='gray', linestyle='--')
        plt.grid(True, axis='y', color='gray', linestyle='--')

        plt.xticks(np.arange(0, 6, 1), 
           [ "{}".format(x) for x in np.arange(0, 6, 1)], 
           fontsize=10
          )
        plt.yticks(np.arange(0, np.ceil(pos_dist[:,1].max()), 1), 
           [ "{}".format(x) for x in np.arange(0, np.ceil(pos_dist[:,1].max()), 1)], 
           fontsize=10
          )

        near_mask = (dist[:, 0] <= 0.25) & (deg[:, 0] <= 10)
        medi_mask = (dist[:, 0] <= 0.5) & (deg[:, 0] <= 10)
        far_mask = (dist[:, 0] <= 5) & (deg[:, 0] <= 10)
        
        near_mask = np.where(near_mask==True)[0]
        medi_mask = np.where(medi_mask==True)[0]
        far_mask = np.where(far_mask==True)[0]

        thre_data = "0.25/10 index {} \n 0.5/10 index {} \n 5/10 index {} \n".format(near_mask, medi_mask, far_mask)
        f = open(os.path.join(save_path, "qualitative_epoch{}".format(epoch), "Analysis positive dist predict-index.txt"), 'w')
        f.write(thre_data)
        f.close()

        plt.title(thre_data)
        filename = "Analysis positive dist predict-index.png"
        plt.savefig(os.path.join(save_path, "qualitative_epoch{}".format(epoch), filename))
        plt.close()


        #--
        fig = plt.figure(figsize=(20, 20))
        scat_deg = plt.scatter(pos_deg[:,0], pos_deg[:,1], s=10)
        plt.xlabel('predict positive deg(m)', fontsize=20)
        plt.ylabel('index positive deg(m)', fontsize=20)
        plt.grid(True, axis='x', color='gray', linestyle='--')
        plt.grid(True, axis='y', color='gray', linestyle='--')

        plt.xticks(np.arange(0, 11, 1), 
           [ "{}".format(x) for x in np.arange(0, 11, 1)], 
           fontsize=10
          )
        plt.yticks(np.arange(0, np.ceil(pos_deg[:,1].max()), 1), 
           [ "{}".format(x) for x in np.arange(0, np.ceil(pos_deg[:,1].max()), 1)], 
           fontsize=10
          )

        near_mask = (dist[:, 0] <= 0.25) & (deg[:, 0] <= 10)
        medi_mask = (dist[:, 0] <= 0.5) & (deg[:, 0] <= 10)
        far_mask = (dist[:, 0] <= 5) & (deg[:, 0] <= 10)
        
        near_mask = np.where(near_mask==True)[0]
        medi_mask = np.where(medi_mask==True)[0]
        far_mask = np.where(far_mask==True)[0]


        thre_data = "0.25/10 index {} \n 0.5/10 index {} \n 5/10 index {} \n".format(near_mask, medi_mask, far_mask)
        f = open(os.path.join(save_path, "qualitative_epoch{}".format(epoch), "Analysis positive deg predict-index.txt"), 'w')
        f.write(thre_data)
        f.close()

        plt.title(thre_data)
        filename = "Analysis positive deg predict-index.png"
        plt.savefig(os.path.join(save_path, "qualitative_epoch{}".format(epoch), filename))
        plt.close()


        #--negative
        fig = plt.figure(figsize=(20, 20))
        scat_dist = plt.scatter(neg_dist[:,0], neg_dist[:,1], s=10)
        plt.xlabel('predict negative dist(m)', fontsize=20)
        plt.ylabel('index negative dist(m)', fontsize=20)
        plt.grid(True, axis='x', color='gray', linestyle='--')
        plt.grid(True, axis='y', color='gray', linestyle='--')

        plt.xticks(np.arange(5, np.ceil(neg_dist[:,0].max()), 1), 
           [ "{}".format(x) for x in np.arange(5, np.ceil(neg_dist[:,0].max()), 1)], 
           fontsize=10
          )
        plt.yticks(np.arange(0, np.ceil(neg_dist[:,1].max()), 1), 
           [ "{}".format(x) for x in np.arange(0, np.ceil(neg_dist[:,1].max()), 1)], 
           fontsize=10
          )

        neg_mask = np.where(positive_mask==False)[0]

        plt.title("Negative index {}".format(neg_mask))
        filename = "Analysis negative dist predict-index.png"
        plt.savefig(os.path.join(save_path, "qualitative_epoch{}".format(epoch), filename))
        plt.close()

        #--
        fig = plt.figure(figsize=(20, 20))
        scat_deg = plt.scatter(neg_deg[:,0], neg_deg[:,1], s=10)
        plt.xlabel('predict negative deg(m)', fontsize=20)
        plt.ylabel('index negative deg(m)', fontsize=20)
        plt.grid(True, axis='x', color='gray', linestyle='--')
        plt.grid(True, axis='y', color='gray', linestyle='--')

        plt.xticks(np.arange(10, np.ceil(neg_deg[:,0].max()), 1), 
           [ "{}".format(x) for x in np.arange(10, np.ceil(neg_deg[:,0].max()), 1)], 
           fontsize=10
          )
        plt.yticks(np.arange(0, np.ceil(neg_deg[:,1].max()), 1), 
           [ "{}".format(x) for x in np.arange(0, np.ceil(neg_deg[:,1].max()), 1)], 
           fontsize=10
          )

        neg_mask = np.where(positive_mask==False)[0]

        plt.title("Negative index {}".format(neg_mask))
        filename = "Analysis negative deg predict-index.png"
        plt.savefig(os.path.join(save_path, "qualitative_epoch{}".format(epoch), filename))
        plt.close()

        #--
        fig = plt.figure(figsize=(20, 20))
        scat_dist = plt.scatter(dist[:,0], dist[:,1], s=10)
        plt.xlabel('predict dist(m)', fontsize=20)
        plt.ylabel('index dist(m)', fontsize=20)
        plt.grid(True, axis='x', color='gray', linestyle='--')
        plt.grid(True, axis='y', color='gray', linestyle='--')

        plt.xticks(np.arange(0, np.ceil(dist[:,0].max()), 1), 
           [ "{}".format(x) for x in np.arange(0, np.ceil(dist[:,1].max()), 1)], 
           fontsize=10
          )
        plt.yticks(np.arange(0, np.ceil(dist[:,1].max()), 1), 
           [ "{}".format(x) for x in np.arange(0, np.ceil(dist[:,1].max()), 1)], 
           fontsize=10
          )

        filename = "Analysis dist predict-index.png"
        plt.savefig(os.path.join(save_path, "qualitative_epoch{}".format(epoch), filename))
        plt.close()

        #--
        fig = plt.figure(figsize=(20, 20))
        scat_deg = plt.scatter(deg[:,0], deg[:,1], s=10)
        plt.xlabel('predict deg', fontsize=20)
        plt.ylabel('index deg', fontsize=20)
        plt.grid(True, axis='x', color='gray', linestyle='--')
        plt.grid(True, axis='y', color='gray', linestyle='--')

        plt.xticks(np.arange(0, np.ceil(deg[:,0].max()), 10), 
           [ "{}".format(x) for x in np.arange(0, np.ceil(deg[:,1].max()), 10)], 
           fontsize=10
          )
        plt.yticks(np.arange(0, np.ceil(deg[:,1].max()), 10), 
           [ "{}".format(x) for x in np.arange(0, np.ceil(deg[:,1].max()), 10)], 
           fontsize=10
          )

        filename = "Analysis deg predict-index.png"
        plt.savefig(os.path.join(save_path, "qualitative_epoch{}".format(epoch), filename))
        plt.close()



