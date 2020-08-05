import pandas as pd
import numpy as np
import os
from tqdm import tqdm


if __name__ == '__main__':


    root_imgdir = "/raid3/datasets/google_v1/train/image/"
    root_csvdir = "/raid3/datasets/google_v1/train/csv/subset/"

    img_subset_list = os.listdir(root_imgdir)
    csv_subset_list = os.listdir(root_csvdir)

    img_subset_list.sort()
    csv_subset_list.sort()

    start = pd.read_csv(os.path.join(root_csvdir, csv_subset_list[0]))
    
    start['folder'] = str(img_subset_list[0])
    
    for i, (img, csv) in enumerate(tqdm(zip(img_subset_list, csv_subset_list))):
        if i==0:
            continue
        curr_csv = pd.read_csv(os.path.join(root_csvdir, csv))
        curr_csv['folder'] = str(img)

        start = pd.concat([start, curr_csv], ignore_index=True)
    
    import pdb; pdb.set_trace()
    # start.to_csv("./train.csv", index=False)
        