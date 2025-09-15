import torch
import torch.nn.functional as F


def pairwise_rankloss(scores: torch.Tensor, mask: torch.Tensor,
                      tau: float = 0.7,
                      weights=(1.0, 0.7, 0.5, 0.3)) -> torch.Tensor:
    """
    Adjacent pairwise ranking loss for ordered slots.
    scores: [B, T, L]
    mask:   [B, T, L] (bool) True for valid slot
    Returns: scalar tensor
    """
    B, T, L = scores.shape
    loss, cnt = 0.0, 0
    s = scores.clone()
    s = torch.where(mask, s, torch.full_like(s, float('-inf')))
    for k in range(L - 1):
        s_top = s[..., k]
        s_bot = s[..., k + 1]
        valid = torch.isfinite(s_top) & torch.isfinite(s_bot)
        if valid.any():
            gap = (s_top[valid] - s_bot[valid]) / tau
            loss = loss + weights[k] * F.softplus(-gap).mean()
            cnt += 1
    if cnt == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype, requires_grad=True)
    return loss / cnt


def listwise_rankloss(scores: torch.Tensor, mask: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """
    Listwise soft target distribution over valid slots per (b, t).
    scores: [B, T, L]
    mask:   [B, T, L]
    """
    B, T, L = scores.shape
    loss, cnt = 0.0, 0
    for b in range(B):
        for t in range(T):
            keep = mask[b, t]
            if keep.sum() >= 2:
                s_kept = scores[b, t, keep] / tau
                Lk = s_kept.size(0)
                target = torch.arange(Lk, 0, -1, device=scores.device, dtype=scores.dtype)
                q = F.softmax(target, dim=0)
                p = F.log_softmax(s_kept, dim=0)
                loss = loss + -(q * p).sum()
                cnt += 1
    if cnt == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype, requires_grad=True)
    return loss / cnt


