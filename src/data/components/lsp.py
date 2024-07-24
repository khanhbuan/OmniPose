import os
import scipy.io
from torch.utils.data import Dataset

def read_data_file(root_dir):
    arr = os.listdir(os.path.join(root_dir, 'images'))
    arr.sort()
    arr = ["./data/lsp/images/" + dir for dir in arr]
    return arr

def read_mat_file(root_dir):
    mat = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
    kpts = mat.transpose([2, 0, 1])
    return kpts

class LSP_Data(Dataset):
    def __init__(self, root_dir):
        self.img_list = read_data_file(root_dir)
        self.kpt_list = read_mat_file(root_dir)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        kpt = self.kpt_list[idx]

        return img_path, kpt

    def __len__(self):
        return len(self.img_list)