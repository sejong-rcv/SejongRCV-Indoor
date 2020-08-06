import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from lib import postprocessing as pp
from lib import utils as u

class Ensemble3(nn.Module):
    def __init__(self, network1, network2, network3, is_backbone1=False, is_backbone2=False, is_backbone3=False):
        super(Ensemble3, self).__init__()
        self.network1 = network1
        self.network2 = network2
        self.network3 = network3

        self.is_backbone1 = is_backbone1
        self.is_backbone2 = is_backbone2
        self.is_backbone3 = is_backbone3

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))

        self.size1=None
        self.size2=None
        self.size3=None

    
    def forward(self, x):

        out1 = self.network1(x)
        out2 = self.network2(x)
        out3 = self.network3(x)

        out1 = out1.unsqueeze(0) if out1.dim()==1 else out1
        out2 = out2.unsqueeze(0) if out2.dim()==1 else out2
        out3 = out3.unsqueeze(0) if out3.dim()==1 else out3

        if self.is_backbone1:
            out1 = self.maxpool(out1)
            out1 = out1.view(out1.size(0), -1)
            out1 = F.normalize(out1, p=2, dim=1)
        if self.is_backbone2:
            out2 = self.maxpool(out2)
            out2 = out2.view(out2.size(0), -1)
            out2 = F.normalize(out2, p=2, dim=1)
        if self.is_backbone3:
            out3 = self.maxpool(out3)
            out3 = out3.view(out3.size(0), -1)
            out3 = F.normalize(out3, p=2, dim=1)
        
        self.size1, self.size2, self.size3 = out1.shape[1], out2.shape[1], out3.shape[1]

        out_concat = torch.cat((out1, out2, out3), dim=1)
        
        return out_concat

    def postprocessing(self, index_feat, test_feat, dim=[4096, 4096, 4096]):

        index_feat1 = index_feat[:, :self.size1]
        index_feat2 = index_feat[:, self.size1:self.size1+self.size2]
        index_feat3 = index_feat[:, self.size1+self.size2:]

        test_feat1 = test_feat[:, :self.size1]
        test_feat2 = test_feat[:, self.size1:self.size1+self.size2]
        test_feat3 = test_feat[:, self.size1+self.size2:]

        if self.is_backbone1 is False:
            if self.size1>dim[0]:
                pca = pp.PCAwhitening(pca_dim=dim[0], pca_whitening=True)
                index_feat1 = pca.fit_transform(index_feat1)      
                test_feat1 = pca.transform(test_feat1)
        if self.is_backbone2 is False:
            if self.size2>dim[1]:
                pca = pp.PCAwhitening(pca_dim=dim[1], pca_whitening=True)
                index_feat2 = pca.fit_transform(index_feat2)      
                test_feat2 = pca.transform(test_feat2)
        if self.is_backbone3 is False:
            if self.size3>dim[2]:
                pca = pp.PCAwhitening(pca_dim=dim[2], pca_whitening=True)
                index_feat3 = pca.fit_transform(index_feat3)      
                test_feat3 = pca.transform(test_feat3) 

        index_feat = np.concatenate((index_feat1, index_feat2, index_feat3), axis=1)
        test_feat = np.concatenate((test_feat1, test_feat2, test_feat3), axis=1)

        index_feat = F.normalize(torch.from_numpy(index_feat), p=2, dim=1).numpy()
        test_feat = F.normalize(torch.from_numpy(test_feat), p=2, dim=1).numpy()
        
        return index_feat, test_feat