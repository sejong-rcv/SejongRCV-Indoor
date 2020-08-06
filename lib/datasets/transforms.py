import pandas as pd
import os
import numpy as np
import cv2
import torch
import torch.utils.data

class Resize(object): 

    def __init__(self, output_size, flag='same'):

        assert isinstance(output_size, (int, tuple, list))
        self.output_size = output_size
        self.flag = flag

    def __call__(self, data):   

        if isinstance(data, dict):
            image = data['image']
        elif isinstance(data, np.ndarray):
            image = data
        else:
            raise ValueError("Type error!")
        
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if self.flag is 'valid':
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
                    
            elif self.flag is 'same':
                new_h, new_w = self.output_size, self.output_size

            else:
                raise ValueError('Check Resize flag input!')

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        img = cv2.resize(image, (new_w, new_h))

        if isinstance(data, dict):
            data['image'] = img
        elif isinstance(data, np.ndarray):
            data = img
        else:
            raise ValueError("Type error!")

        return data


class Normalize(object):

    def __init__(self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        assert isinstance(mean, tuple)
        assert isinstance(std, tuple)
        self.mean = mean
        self.std = std
 
    def __call__(self, data):

        image = data['image']

        
        image = image/255

        image[:,:,0] = (image[:,:,0] - self.mean[0])/self.std[0]
        image[:,:,1] = (image[:,:,1] - self.mean[1])/self.std[1]
        image[:,:,2] = (image[:,:,2] - self.mean[2])/self.std[2]

        image = image.astype(np.float32)
        
        
        data['image'] = image


        return data


class ToTensor(object):

    def __call__(self, data):

        
        image, label = data['image'], data['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.transpose(image, (2, 0, 1))

        data['image'] = torch.from_numpy(image)
        data['label'] = torch.from_numpy(np.array(label))
    
        return data


class CenterCrop(object):

    def __init__(self, crop_ratio=0.15):
        self.crop_ratio = crop_ratio
 
    def __call__(self, data):

        image = data['image']

        H,W,C = image.shape
        
        mg_H = int(H*self.crop_ratio)
        mg_W = int(W*self.crop_ratio)

        image = image[mg_H:H-mg_H, mg_W:W-mg_W, :]
        
        data['image'] = image

        return data

class Grayscale(object):

    def __call__(self, data):

        if isinstance(data, dict):
            image = data['image']
        elif isinstance(data, np.ndarray):
            image = data
        else:
            raise ValueError("Type error!")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if isinstance(data, dict):
            data['image'] = image
        elif isinstance(data, np.ndarray):
            data = image
        else:
            raise ValueError("Type error!")


        return data