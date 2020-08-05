import os
import numpy as np
import h5py


def make_csv(root, place, status="train", use_valid=True):
    
    if (status!="train") and (use_valid is True):
        raise ValueError("check arg in")

    timestamp = os.listdir(os.path.join(root, place, status))
    timestamp.remove("csv")
    timestamp.sort()

    valid_ind=None
    if use_valid is True:
        valid_ind = check_minsize(os.path.join(root, place, status), timestamp)
        valid_time = np.asarray(timestamp)[valid_ind:valid_ind+1]
        # valid_imglist = make_imglist(os.path.join(root, place, status), valid_time)
    
    train_ind = np.setdiff1d(np.arange(len(timestamp)), valid_ind)
    train_time = np.asarray(timestamp)[train_ind]
    train_imglist = make_imglist(os.path.join(root, place, status), train_time)
    

    import pdb; pdb.set_trace()

def make_dataframe(root, img_list):
    pass

def make_imglist(root, timestamp):

    img_list = []
    for time in timestamp:
        curr_list = os.listdir(os.path.join(root, time, "images"))
        curr_list.sort()
        img_list.extend(curr_list)

    img_list=np.asarray(img_list)
    img_list.sort()

    for time in timestamp:
        f = h5py.File(os.path.join(root, time, "jwon/groundtruth.hdf5"), 'r')
        keys = list(f.keys())

        all_datalist = []
        for ki in range(0,len(keys),2):
            if keys[ki].split("_")[0] != keys[ki+1].split("_")[0]:
                import pdb; pdb.set_trace()
            else:
                camera_name = keys[ki].split("_")[0]
            
            if "lidar" in camera_name:
                continue

            cam_list = cam_ind(img_list, camera_name)
            try:
                if len(cam_list) != cam_list[-1]-cam_list[0]+1:
                    raise ValueError("sort!")
            except:
                import pdb; pdb.set_trace()
            curr_imglist = img_list[cam_list]
            curr_imglist.sort()
            curr_imglist = np.expand_dims(curr_imglist, axis=1)

            pose = f[keys[ki]][:]
            stamp = f[keys[ki+1]][:]
            
            sort_ind = np.squeeze(np.argsort(stamp, axis=0), axis=1)
            pose = pose[sort_ind]
            stamp = stamp[sort_ind]
            try:
                curr_datalist = np.concatenate((curr_imglist, pose, stamp), axis=1)
            except:
                import pdb; pdb.set_trace()
            all_datalist.extend(curr_datalist)

        f.close()
    return np.asarray(all_datalist)


def cam_ind(img_list, camera_name):

    ind=[]
    for i,im_name in enumerate(img_list):
        im_cam = im_name.split("_")[0]
        if im_cam==camera_name:
            ind.append(i)
    
    return ind

def check_minsize(root, folder_list):

    min_size=float('inf')
    min_ind=0

    for i,folder in enumerate(folder_list):
        path = os.path.join(root, folder, "images")
        curr_size = len(os.listdir(path))
        if min_size>curr_size:
            min_size=curr_size
            min_ind=i
    
    return min_ind




if __name__ == '__main__':
    make_csv(root="../../../NaverML_indoor", place="b1", status="train")