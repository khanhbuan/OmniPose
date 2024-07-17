import cv2
import torch
import numbers
import collections
import numpy as np

def normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def to_tensor(pic):
    img = torch.from_numpy(pic.transpose(2, 0, 1))
    return img.float()

def resize(img, kpt, center, ratio):
    if not (isinstance(ratio, numbers.Number) or (isinstance(ratio, collections.Iterable) and len(ratio) == 2)):
        raise TypeError('Got inappropriate ratio arg: {}'.format(ratio))
    
    h, w, _ = img.shape
    if w < 64:
        img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - w, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        w = 64
    
    if isinstance(ratio, numbers.Number):
        num = len(kpt)
        for i in range(num):
            kpt[i][0] *= ratio
            kpt[i][1] *= ratio
        center[0] *= ratio
        center[1] *= ratio
        return cv2.resize(img, (0, 0), fx=ratio, fy=ratio), kpt, center
    else:

        num = len(kpt)
        for i in range(num):
            kpt[i][0] *= ratio[1]
            kpt[i][1] *= ratio[0]
        center[0] *= ratio[1]
        center[1] *= ratio[0]

    return np.ascontiguousarray(cv2.resize(img,(368, 368),interpolation=cv2.INTER_CUBIC)), kpt, center

class Resized(object):
    def __init__(self, size):
        assert (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    @staticmethod
    def get_params(img, output_size):
        height, width, _ = img.shape
        return (output_size[0] * 1.0 / height, output_size[1] * 1.0 / width)

    def __call__(self, img, kpt, center):
        ratio = self.get_params(img, self.size)
        return resize(img, kpt, center, ratio)
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, kpt, center):
        for t in self.transforms:
            img, kpt, center = t(img, kpt, center)

        return img, kpt, center