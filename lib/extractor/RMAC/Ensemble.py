import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
from torchvision.models import resnet18,resnet152 
import os
import numpy as np

from . import *

class Ensemble(nn.Module):
    def __init__(self,model_name="ResNet101",dim=512):
        super(Ensemble, self).__init__()

       
        self.SEResNext101 = se_resnext101()
        
        for param in self.SEResNext101.parameters():
            #import pdb;pdb.set_trace()
            param.requires_grad = False
        print("Freeze ResNet")
    def forward(self, image):
        #output1=self.ResNet101(image)
        #selfimport pdb;pdb.set_trace()
        #output2=self.resnet(image)
        output2= self.SEResNext101(image)
        
        #output=torch.cat((output1,output2),dim=1)
        output=output2;
        #import pdb;pdb.set_trace()
        return output;
        