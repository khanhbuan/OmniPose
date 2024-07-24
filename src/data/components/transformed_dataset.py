import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

class transformed_dataset(Dataset):
    def __init__(self, dataset, stride, sigma, transform=None):
        self.dataset = dataset
        self.stride = stride
        self.sigma = sigma
        self.transform = transform
    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        img_path, kpt = self.dataset.__getitem__(idx)
        
        img = np.array(cv2.imread(img_path), dtype=np.float32)

        if self.transform is not None:
            transformed = self.transform(image=img, keypoints=kpt)
            img, kpt = transformed["image"], transformed["keypoints"]
        
        height, width, _ = img.shape
        
        heatmap = np.zeros((int(height/self.stride), int(width/self.stride), len(kpt)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 256 to 64
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = gaussian_kernel(size_h = int(height/self.stride), size_w = int(width/self.stride), 
                                       center_x = x, center_y = y, sigma = self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i] = heat_map

        img = torch.permute(torch.from_numpy(img), (2, 0, 1))
        heatmap = torch.permute(torch.from_numpy(heatmap), (2, 0, 1))

        return img, heatmap
