import torch

from .superpoint import SuperPoint
from .superglue import SuperGlue


class SP(torch.nn.Module):
    def __init__(self, config={}):
        super().__init__()
        superPointdict=dict()
        superPointdict["nms_radius"]=4
        superPointdict["keypoint_threshold"]= 0.005
        superPointdict["max_keypoints"]=1024
        
        self.superpoint = SuperPoint(superPointdict)


    def forward(self, data):

        pred = {}
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred



