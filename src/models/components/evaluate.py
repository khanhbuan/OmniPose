import torch
import numpy as np
from torchmetrics import Metric

def calc_dists(preds, target, normalize):
    preds  =  preds.astype(np.float32)
    target = target.astype(np.float32)
    dists  = np.zeros((preds.shape[1], preds.shape[0]))

    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 0 and target[n, c, 1] > 0:
                normed_preds =  preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1

    return dists

def dist_acc(dists, threshold = 0.5):
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], threshold).sum() * 1.0 / num_dist_cal
    else:
        return -1
    
def get_max_preds(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds= np.tile(idx, (1,1,2)).astype(np.float32)

    preds[:,:,0] = (preds[:,:,0]) % width
    preds[:,:,1] = np.floor((preds[:,:,1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1,1,2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def accuracy(output, target, thr_PCK, hm_type='gaussian'):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == "gaussian":
        pred, _   = get_max_preds(output)
        target, _ = get_max_preds(target)

        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h,w]) / 10

    dists = calc_dists(pred, target, norm)

    acc = 0
    cnt = 0
    visible = np.zeros((len(idx)))

    for i in range(len(idx)):
        acc = dist_acc(dists[idx[i]])
        if acc >= 0:
            cnt += 1
            visible[i] = 1
        else:
            acc = 0

    PCK = np.zeros((len(idx)))
    avg_PCK = 0

    pelvis = [(target[0,2,0] + target[0,3,0])/2, (target[0,2,1] + target[0,3,1])/2]
    torso  = np.linalg.norm(target[0,12,:] - pelvis)

    for i in range(len(idx)):
        PCK[i] = dist_acc(dists[idx[i]], thr_PCK*torso)
        if PCK[i] < 0:
            PCK[i] = 0

    return PCK, visible

class PCK(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("PCK", default=torch.zeros(14), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(14), dist_reduce_fx="sum")
    
    def update(self, pred, target, idx):
        acc_PCK, visible = accuracy(pred, target, 0.2)
        
        for j in range(0, 14):
                if visible[j] == 1:
                    self.PCK[j] = (self.PCK[j]*self.count[j] + acc_PCK[j])  / (self.count[j] + 1)
                    self.count[j] += 1

    def compute(self):
        return self.PCK.sum()/14

    def reset(self):
        self.PCK = torch.zeros(14)
        self.count = torch.zeros(14)