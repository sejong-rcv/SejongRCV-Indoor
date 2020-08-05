import cv2
import numpy as np

class SIFT(object):
    def __init__(self, dense=False, root=False):
        self.alg = self.sift if root is False else self.root_sift
        self.dense = dense
        
    
    def __call__(self, image):
        (kps, descs) = self.alg(image)
        return (kps, descs)
    
    def sift(self, image, eps=1e-7, step_size=8):

        sift = cv2.xfeatures2d.SIFT_create()
        
        if self.dense is True:
            kps = [cv2.KeyPoint(x, y, step_size) 
                for y in range(0, image.shape[0], step_size) 
                for x in range(0, image.shape[1], step_size)]
            (kps, descs) = sift.compute(image, kps)
        else:
            (kps, descs) = sift.detectAndCompute(image, None)
        
        
        return (kps, descs)

    def root_sift(self, image, eps=1e-7):

        (kps, descs) = self.sift(image, eps)
        if len(kps) == 0:
            return ([], None)
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        return (kps, descs)