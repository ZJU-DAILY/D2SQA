import torch
import torch.nn.functional as F


def compute_jsd_sigmoid(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl_pm = p*torch.log((p+eps)/(m+eps)) + (1-p)*torch.log((1-p+eps)/(1-m+eps))
    kl_qm = q*torch.log((q+eps)/(m+eps)) + (1-q)*torch.log((1-q+eps)/(1-m+eps))
    jsd   = 0.5 * (kl_pm + kl_qm)        # (B, C)
    return jsd.mean()



def dynamic_loss(alpha, kl_loss, task_loss):

    total_loss = alpha * kl_loss + (1 - alpha) * task_loss
    return total_loss

