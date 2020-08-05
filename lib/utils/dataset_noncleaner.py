import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def cleaner(img_path, ann_path, clean_path):
    ann = pd.read_csv(ann_path)

    omit_index = []
    for i in tqdm(range(len(ann))):

        exist = os.path.isfile(os.path.join(img_path, ann.iloc[i].id + '.jpg'))
        if exist is False:
            omit_index.append(i)
            ann.iloc[i].id = np.nan

    ann.dropna(axis=0, inplace=True)

    print("Omit index num : " + str(len(omit_index)))
    print("Omit index list : "  + str(omit_index))
    
    ann.to_csv(clean_path, index=False)
    return 0

if __name__ == '__main__':

    # cleaner("./datasets/google_v1/train/",
    #         "./datasets/google_v1/csv/origin/train.csv",
    #         "./datasets/google_v1/csv/clean/train.csv")

    # cleaner("./datasets/google_v1/test/",
    #         "./datasets/google_v1/csv/origin/test.csv",
    #         "./datasets/google_v1/csv/clean/test.csv")

    # cleaner("./datasets/google_v1/index/",
    #         "./datasets/google_v1/csv/origin/index.csv",
    #         "./datasets/google_v1/csv/clean/index.csv")

    cleaner("./datasets/google_v1/index/",
            "./datasets/google_v1/csv/origin/index.csv",
            "./datasets/google_v1/csv/clean/index.csv")

