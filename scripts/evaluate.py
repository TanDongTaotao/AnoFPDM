# copied from https://github.com/SoloChe/AnoDDPM/blob/master/evaluation.py

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import torch.nn.functional as F
import torchvision
from torch.nn.modules.utils import _pair, _quadruple
from skimage.measure import label, regionprops
import scipy.spatial.distance as sp


def evaluate(
    real_mask: torch.Tensor,
    recon_mask: torch.Tensor,
    source: torch.Tensor,
    ano_map: torch.Tensor,
    label: torch.Tensor = None, 
    cc_filter=True,
):
    """
    real_mask: [b, 1, h, w]; recon_mask: [b, 1, h, w];
    ano_map: [b, 1, h, w]; source: [b, n_mod, h, w]
    label: [b] - 0 for normal, 1 for anomalous, image-level label
    if label is not none, only evaluate on anomalous samples, for unhealthy-only settings
    Rerurn a dict of average metrics (float) for each batch
    """
    if label is not None:
        real_mask = real_mask[torch.where(label == 1)[0], ...]
        recon_mask = recon_mask[torch.where(label == 1)[0], ...]
        source = source[torch.where(label == 1)[0], ...]
        ano_map = ano_map[torch.where(label == 1)[0], ...]

    if cc_filter:
        recon_mask = connected_components_3d(recon_mask, thr=40) # post-processed by connected components

    dice_batch = dice_coeff(real_mask, recon_mask)
    iou_batch = IoU(real_mask, recon_mask)
    precision_batch = precision(real_mask, recon_mask)
    recall_batch = recall(real_mask, recon_mask)
    fpr, tpr, _ = ROC_AUC(source, real_mask, ano_map)
    AUC_score_batch = auc(fpr, tpr)
    AUPRC_score_batch = AUPRC_score(source, real_mask, ano_map)

    return {
        "dice": dice_batch,
        "iou": iou_batch,
        "precision": precision_batch,
        "recall": recall_batch,
        "AUC": AUC_score_batch,
        "AUPRC": AUPRC_score_batch,
        "recon_mask": recon_mask,
    }


# for anomalous dataset - metric of crossover
def dice_coeff(real_mask: torch.Tensor, recon_mask: torch.Tensor, smooth=0.000001):
    intersection = torch.logical_and(recon_mask, real_mask).sum(dim=[1, 2, 3])
    union = torch.sum(recon_mask, dim=[1, 2, 3]) + torch.sum(real_mask, dim=[1, 2, 3])
    dice = torch.mean((2.0 * intersection + smooth) / (union + smooth))
    return dice.item()


def IoU(real_mask, recon_mask, smooth=0.000001):
    intersection = torch.logical_and(real_mask, recon_mask).sum(dim=(1, 2, 3))
    union = torch.logical_or(real_mask, recon_mask).sum(dim=(1, 2, 3))
    iou = torch.mean((intersection + smooth) / (union + smooth))
    return iou.item()


def precision(real_mask, recon_mask):
    TP = (real_mask == 1) & (recon_mask == 1)
    FP = (real_mask == 1) & (recon_mask == 0)
    pr = torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)
    return pr.item()


def recall(real_mask, recon_mask):
    TP = (real_mask == 1) & (recon_mask == 1)
    FN = (real_mask == 0) & (recon_mask == 1)
    re = torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)
    return re.item()


def ROC_AUC(source, real_mask, ano_map):
    # note that source is rescaled to [-1, 1]
    # find the region with mean intensity > -0.95 for auc
    # you can also comment the following line to calculate prauc for the whole image
    foreground = source.mean(dim=1, keepdim=True).reshape(-1) > -0.95
    real_mask = real_mask.reshape(-1)[foreground]
    ano_map = ano_map.reshape(-1)[foreground]
    return roc_curve(
        real_mask.detach().cpu().numpy().flatten(),
        ano_map.detach().cpu().numpy().flatten(),
    )


def AUPRC_score(source, real_mask, ano_map):
    # note that source is rescaled to [-1, 1]
    foreground = source.mean(dim=1, keepdim=True).reshape(-1) > -0.95
    real_mask = real_mask.reshape(-1)[foreground]
    ano_map = ano_map.reshape(-1)[foreground]
    return average_precision_score(
        real_mask.detach().cpu().numpy().flatten(),
        ano_map.detach().cpu().numpy().flatten(),
    )


def AUC_score(fpr, tpr):
    return auc(fpr, tpr)


# image-level classification metrics
def get_stats(Y, PRED_Y):
    y_curr = Y[-1]
    Y = torch.cat(Y, dim=0)
    PRED_Y = torch.cat(PRED_Y, dim=0)
    acc = torch.sum(Y == PRED_Y).float() / Y.shape[0]
    re = recall(Y, PRED_Y)
    pr = precision(Y, PRED_Y)
    num_ano = torch.where(Y == 1)[0].shape[0]
    num_ano_curr = torch.where(y_curr == 1)[0].shape[0]
    return {"acc": acc.item(), "recall": re, "precision": pr, "num_ano_total": num_ano, "num_ano_curr": num_ano_curr}


def median_pool(ano_map, kernel_size=5, stride=1, padding=2):
    # ano_map: [b, 1, h, w]; source: [b, n_mod, h, w]
    k = _pair(kernel_size)
    stride = _pair(stride)
    padding = _quadruple(padding)
    x = F.pad(ano_map, padding, mode="reflect")
    x = x.unfold(2, k[0], stride[0]).unfold(3, k[1], stride[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x

def connected_components_3d(volume, thr=20):
    device = volume.device
    is_batch = True
    is_torch = torch.is_tensor(volume)
    if is_torch:
        volume = volume.cpu().numpy()
    if volume.ndim == 3:
        volume = np.expand_dims(volume, axis=0)
        is_batch = False
    # shape [b, d, h, w], treat every sample in batch independently
    pbar = range(len(volume))
    for i in pbar:
        cc_volume = label(volume[i], connectivity=3)
        props = regionprops(cc_volume)

        nonzero_props = []
        for prop in props:
            if prop["filled_area"] <= thr:
                volume[i, cc_volume == prop["label"]] = 0
            else:
                nonzero_props.append(prop)

    if not is_batch:
        volume = volume.squeeze(0)
    if is_torch:
        volume = torch.from_numpy(volume).to(device)
    return volume

class logging_metrics:
    def __init__(self, logger):
        self.DICE = []
        self.DICE_ANO = []
        self.IOU = []
        self.IOU_ANO = []
        self.RECALL = []
        self.RECALL_ANO = []
        self.PRECISION = []
        self.PRECISION_ANO = []
        self.AUC = []
        self.AUC_ANO = []
        self.AUPRC = []
        self.AUPRC_ANO = []
        self.logger = logger
     
    def logging(self, eval_metrics,  eval_metrics_ano, cls_metrics, k):
        
        
        self.DICE.append(eval_metrics["dice"])
        self.DICE_ANO.append(eval_metrics_ano["dice"]*cls_metrics["num_ano_curr"])

        self.IOU.append(eval_metrics["iou"])
        self.IOU_ANO.append(eval_metrics_ano["iou"]*cls_metrics["num_ano_curr"])

        self.RECALL.append(eval_metrics["recall"])
        self.RECALL_ANO.append(eval_metrics_ano["recall"]*cls_metrics["num_ano_curr"])

        self.PRECISION.append(eval_metrics["precision"])
        self.PRECISION_ANO.append(eval_metrics_ano["precision"]*cls_metrics["num_ano_curr"])

        self.AUC.append(eval_metrics["AUC"])
        self.AUC_ANO.append(eval_metrics_ano["AUC"]*cls_metrics["num_ano_curr"])

        self.AUPRC.append(eval_metrics["AUPRC"])
        self.AUPRC_ANO.append(eval_metrics_ano["AUPRC"]*cls_metrics["num_ano_curr"])

        self.logger.log(
            f"-------------------------------------at batch {k}-----------------------------------------"
        )
        self.logger.log(f"mean dice: {eval_metrics['dice']:0.3f}")
        self.logger.log(f"mean iou: {eval_metrics['iou']:0.3f}")
        self.logger.log(f"mean precision: {eval_metrics['precision']:0.3f}")
        self.logger.log(f"mean recall: {eval_metrics['recall']:0.3f}")
        self.logger.log(f"mean auc: {eval_metrics['AUC']:0.3f}")
        self.logger.log(f"mean pr auc: {eval_metrics['AUPRC']:0.3f}")
        self.logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        self.logger.log(f"mean dice ano: {eval_metrics_ano['dice']:0.3f}")
        self.logger.log(f"mean iou ano: {eval_metrics_ano['iou']:0.3f}")
        self.logger.log(f"mean precision ano: {eval_metrics_ano['precision']:0.3f}")
        self.logger.log(f"mean recall ano: {eval_metrics_ano['recall']:0.3f}")
        self.logger.log(f"mean auc ano: {eval_metrics_ano['AUC']:0.3f}")
        self.logger.log(f"mean pr auc ano: {eval_metrics_ano['AUPRC']:0.3f}")
        self.logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        self.logger.log(f"running dice: {np.mean(self.DICE):0.3f}")  # keep 3 decimals
        self.logger.log(f"running iou: {np.mean(self.IOU):0.3f}")
        self.logger.log(f"running precision: {np.mean(self.PRECISION):0.3f}")
        self.logger.log(f"running recall: {np.mean(self.RECALL):0.3f}")
        self.logger.log(f"running auc: {np.mean(self.AUC):0.3f}")
        self.logger.log(f"running pr auc: {np.mean(self.AUPRC):0.3f}")
        self.logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        self.logger.log(f"running dice ano: {np.sum(self.DICE_ANO)/cls_metrics['num_ano_total']:0.3f}")
        self.logger.log(f"running iou ano: {np.sum(self.IOU_ANO)/cls_metrics['num_ano_total']:0.3f}")
        self.logger.log(f"running precision ano: {np.sum(self.PRECISION_ANO)/cls_metrics['num_ano_total']:0.3f}")
        self.logger.log(f"running recall ano: {np.sum(self.RECALL_ANO)/cls_metrics['num_ano_total']:0.3f}")
        self.logger.log(f"running auc ano: {np.sum(self.AUC_ANO)/cls_metrics['num_ano_total']:0.3f}")
        self.logger.log(f"running auprc ano: {np.sum(self.AUPRC_ANO)/cls_metrics['num_ano_total']:0.3f}")
        self.logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        self.logger.log(f"running cls acc: {cls_metrics['acc']:0.3f}")
        self.logger.log(f"running cls recall: {cls_metrics['recall']:0.3f}")
        self.logger.log(f"running cls precision: {cls_metrics['precision']:0.3f}")
        self.logger.log(f"running cls num_ano: {cls_metrics['num_ano_total']}")
        self.logger.log(
            "-------------------------------------------------------------------------------------------"
        )
        