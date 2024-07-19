import torch
import cv2
import numpy as np
from src.models.modelmodule import UniPoseModule
from src.models.components.unipose import unipose

def get_kpts(maps, img_h = 368.0, img_w = 368.0):

    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6[1:]:
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

if __name__ == "__main__":
    ckpt_path="./logs/train/runs/2024-07-18_10-58-30/checkpoints/epoch_060.ckpt"
    model = UniPoseModule.load_from_checkpoint(net=unipose(), checkpoint_path=ckpt_path)
    model.eval()

    # img_path = "./data/lsp/images/im00001.jpg"
    img_path = "./test.png"
    img = cv2.resize(cv2.imread(img_path), (368, 368))
    render_img = img
    
    img = torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1))

    mean = [128.0, 128.0, 128.0]
    std = [256.0, 256.0, 256.0]
    for t, m, s in zip(img, mean, std):
        t.sub_(m).div_(s)
    
    img = img[None,:,:,:]
    
    input = img.to("cuda")
    heat = model(input) 
    
    kpts = get_kpts(heat)
    draw_paints(render_img, kpts)