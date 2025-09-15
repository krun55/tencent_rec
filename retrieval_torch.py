import torch
import torch.nn.functional as F


@torch.no_grad()
def topk_cosine_torch(query: torch.Tensor,
                      items: torch.Tensor,
                      k: int = 100,
                      chunk: int = 4096,
                      device: str = 'cuda',
                      use_bf16: bool = True):
    """
    Pure PyTorch exact retrieval by cosine similarity.
    Args:
        query: [Nq, D] float tensor (cpu or cuda)
        items: [Ni, D] float tensor (cpu or cuda)
        k: top-k
        chunk: chunk size for queries
        device: compute device
        use_bf16: use bfloat16 autocast for GEMM
    Returns:
        (indices, scores): both [Nq, k] on cpu
    """
    items = F.normalize(items, dim=-1).to(device, non_blocking=True)
    out_idx, out_val = [], []
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    Nq = query.size(0)
    for s in range(0, Nq, chunk):
        q = F.normalize(query[s:s + chunk].to(device, non_blocking=True), dim=-1)
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            sims = q @ items.T
        val, idx = sims.topk(k, dim=1)
        out_idx.append(idx.cpu())
        out_val.append(val.cpu())
    return torch.vstack(out_idx), torch.vstack(out_val)


