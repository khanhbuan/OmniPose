import torch
import cv2
import numpy as np
import hydra
from typing import Optional
from lightning import LightningDataModule
from omegaconf import DictConfig
# from src.models.modelmodule import UniPoseModule
# from src.models.components.unipose import unipose

def get_kpts(maps, img_h = 256.0, img_w = 256.0):

    # maps (1, 14, 46, 46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts

def draw_paints(im, kpts):
    """
    0 = Right Ankle
    1 = Right Knee
    2 = Right Hip
    3 = Left Hip
    4 = Left Knee
    5 = Left Ankle
    6 = Right Wrist
    7 = Right Elbow
    8 = Right Shoulder
    9 = Left Shoulder
    10 = Left Elbow
    11 = Left Wrist
    12 = Neck
    13 = Head Top
    """
           #       RED           GREEN          BLACK          CYAN           YELLOW          PINK
    colors = [[000,000,255], [000,255,000], [000,000,000], [255,255,000], [000,255,255], [255,000,255], \
              [000,255,000], [255,000,000], [255,255,000], [255,000,255], [128,000,000], [128,128,128], [000,000,255], [181,61,253]]
           #     GREEN           BLUE           CYAN           PINK            NAVY           GRAY           RED          MAGENTA 
    
    for idx, k in enumerate(kpts):
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=3, thickness=-1, color=colors[idx])
    cv2.imwrite('output.png', im)

@hydra.main(version_base="1.3", config_path="configs/data", config_name="lsp")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup()
    img, heat = datamodule.data_train.__getitem__(1)
    print(img.shape, heat.shape)

if __name__ == "__main__":
    main()