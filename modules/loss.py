import torch
import torch.nn as nn

from deepvac.syszux_loss import BalanceCrossEntropyLoss, DiceLoss, MaskL1Loss

class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, deepvac_config=None):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss
        :param beta: threshold_map loss
        :param ohem_ratio
        :param reduction
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(deepvac_config)
        self.dice_loss = DiceLoss(deepvac_config)
        self.l1_loss = MaskL1Loss(deepvac_config)

    def __call__(self, pred, batch):
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, batch[0], batch[1])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch[2], batch[3])
        metrics = dict(loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps)
        if pred.size()[1] > 2:
            binary_maps = pred[:, 2, :, :]
            loss_binary_maps = self.dice_loss(binary_maps, batch[0], batch[1])
            metrics['loss_binary_maps'] = loss_binary_maps
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics
