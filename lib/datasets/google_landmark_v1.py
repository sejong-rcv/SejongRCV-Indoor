import pandas as pd
import os
import numpy as np
import cv2
import torch
import torch.utils.data
import tqdm

from lib.datasets.transforms import *

pd.options.mode.chained_assignment = None

class Load_GLD1(torch.utils.data.Dataset):

    def __init__(self, image_path, ann_path, transform=None, use_tuple=True, posneg_ratio=5):
        
        self.ann_path = ann_path
        self.ann = pd.read_csv(self.ann_path)
        self.img_path = image_path
        self.transform = transform
        self.use_tuple = use_tuple
        self.posneg_ratio = posneg_ratio


    def __len__(self):
        return len(self.ann)

    def __loadimg__(self, idx):

        curr_id = self.ann.iloc[idx].id
        curr_url = self.ann.iloc[idx].url
        curr_label = int(self.ann.iloc[idx].landmark_id)
        curr_folder = str(self.ann.iloc[idx].folder)
        curr_folder = '{:07d}'.format(int(curr_folder))

        curr_img = cv2.imread(os.path.join(self.img_path, curr_folder, curr_id + '.jpg'))
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)

        data = {'image' : curr_img, \
                'label' : curr_label, \
                'url'   : curr_url}
        

        if self.transform:
            data = self.transform(data)


        return data

    def __concatposneg__(self, anc, pos, neg):
        anc_img = anc['image'].unsqueeze(0)
        pos_img = pos['image'].unsqueeze(0)
        bundle_img = torch.cat((anc_img, pos_img), dim=0)

        bundle_url = []
        bundle_url.append(anc['url'])
        bundle_url.append(pos['url'])

        anc_lab = anc['label'].unsqueeze(0)
        pos_lab = pos['label'].unsqueeze(0)
        bundle_lab = torch.cat((anc_lab, pos_lab), dim=0)

        for sub in neg:
            neg_img = sub['image'].unsqueeze(0)
            bundle_img = torch.cat((bundle_img, neg_img), dim=0)

            bundle_url.append(sub['url'])

            neg_lab = sub['label'].unsqueeze(0)
            bundle_lab = torch.cat((bundle_lab, neg_lab), dim=0)

        data = {'image' : bundle_img, \
                'label' : bundle_lab, \
                'url'   : bundle_url}

        
        return data

    def __getitem__(self, idx):

        anchor = self.__loadimg__(idx)

        if self.use_tuple is False:
            return anchor

        anc_cls = anchor['label'].item()
        
        
        pos_ind = np.where(self.ann.iloc[:,2]==anc_cls)[0]
        pos_pick = np.random.choice(pos_ind, 1)

        pos = self.__loadimg__(pos_pick[0])

        neg_ind = np.setdiff1d(np.array((range(self.ann.index.max()+1))), pos_ind)
        neg_pick = np.random.choice(neg_ind, 1*self.posneg_ratio)
        
        
        neg = []
        for i, part in enumerate(neg_pick):
            neg.append(self.__loadimg__(part))
        
        apn_tuple = self.__concatposneg__(anchor, pos, neg)

        return apn_tuple

    def collate_func(self, batch):
        
        image = torch.tensor([])
        label = torch.tensor([]).long()
        url = []

        for i, val in enumerate(batch):

            image = torch.cat((image, val['image']), dim=0)
            label = torch.cat((label, val['label']), dim=0)
            url.append(val['url'])
        
        data = {'image' : image,
                'label' : label,
                'url'   : url}

        return data


