import numpy as np
import cv2
from torch.utils.data import Dataset
from src.data.components import Transform

def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

class transformed_dataset(Dataset):
    def __init__(self, dataset, stride, sigma, mode):
        self.dataset = dataset
        self.stride = stride
        self.sigma = sigma
        self.mode = mode

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        img_path, kpt, center = self.dataset.__getitem__(idx)
        
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        
        if self.mode == "train":
            transform = Transform.Compose([
                Transform.RandomHorizontalFlip(p=0.5),
                Transform.Resized(256)
            ])
        
        else:
            transform = Transform.Resized(256)

        img, kpt, center = transform(img, kpt, center)

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

        img = Transform.normalize(Transform.to_tensor(img), 
                                  [128.0, 128.0, 128.0],
                                  [256.0, 256.0, 256.0])
        
        heatmap = Transform.to_tensor(heatmap)

        return img, heatmap
    