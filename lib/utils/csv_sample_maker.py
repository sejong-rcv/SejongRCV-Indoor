import numpy as np
import pandas as pd
import torch
from lib import metric as mt
import time as tm
from pyquaternion import Quaternion
import tqdm
def csv_sample_maker(path1, path2, save_path, step=10, dist_thre=0.5, deg_thre=10):
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    datan1 = np.asarray(data1)
    datan2 = np.asarray(data2)

    dataset = np.concatenate((datan1, datan2))
    

    a = tm.time()
    all_xyz = dataset[:,2:5].astype("float64")

    knn = mt.kNN_GPU(d=all_xyz.shape[1], GPU=True, GPU_Number=torch.cuda.current_device())
    knn.train(np.ascontiguousarray(all_xyz, dtype=np.float32))
    topk_list = knn.predict(np.ascontiguousarray(all_xyz, dtype=np.float32), 1024)
    knn.delete()
    del knn
    print(tm.time()-a)

    all_qwxyz = dataset[:,5:9].astype("float64")

    for i, row in enumerate(tqdm.tqdm(topk_list)):

        pos = []
        for j, col in enumerate(row):
            txyz = all_xyz[i]
            sxyz = all_xyz[col]
            if np.linalg.norm(txyz-sxyz)<=dist_thre:
                tqwxyz = Quaternion(all_qwxyz[i])
                sqwxyz = Quaternion(all_qwxyz[col])
                rel_deg = abs((tqwxyz.conjugate*sqwxyz).degrees)
                if rel_deg<=deg_thre:
                    pos.append(col)
        pos.sort()
        pos_line = ', '.join(list(map(str, pos)))

        dataset[i, -1] = pos_line
    
    data_pd = pd.DataFrame({data1.columns[0] : dataset[:,0], \
                        data1.columns[1] : dataset[:,1], \
                        data1.columns[2] : dataset[:,2], \
                        data1.columns[3] : dataset[:,3], \
                        data1.columns[4] : dataset[:,4], \
                        data1.columns[5] : dataset[:,5], \
                        data1.columns[6] : dataset[:,6], \
                        data1.columns[7] : dataset[:,7], \
                        data1.columns[8] : dataset[:,8], \
                        data1.columns[9] : dataset[:,9]})

    data_pd.to_csv(save_path, index=False)



if __name__ == "__main__":
    csv_sample_maker(path1, path2, save_path)
