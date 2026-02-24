import torch
import torch.nn.functional as F


def kl_normal(post_mean, post_std, prior_mean, prior_std):
    var_ratio = (post_std / prior_std) ** 2
    t1 = ((post_mean - prior_mean) / prior_std) ** 2
    return 0.5 * (var_ratio + t1 - 1 - torch.log(var_ratio + 1e-8))


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def bce_logits_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)
