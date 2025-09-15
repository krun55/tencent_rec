import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset, UserSeqDatasetPacked, ItemDatasetPacked, collate_fn_packed
try:
    from model import BaselineModel
except Exception:
    # Fallback: import model.py by absolute file path in this directory
    import importlib.util, os, sys
    _mpath = os.path.join(os.path.dirname(__file__), 'model.py')
    _spec = importlib.util.spec_from_file_location('model', _mpath)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_mod)  # type: ignore
    BaselineModel = getattr(_mod, 'BaselineModel')
from rankloss import pairwise_rankloss, listwise_rankloss
from seed_utils import fix_random_seeds, make_worker_init_fn, make_generator


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.004, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--clip_norm', default=1.0, type=float)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--prefetch_factor', default=4, type=int)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--tf32', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度累积步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='梯度裁剪阈值')
    # Memory-saving options
    parser.add_argument('--grad_checkpointing', default=1, type=int, choices=[0, 1], help='启用块级梯度检查点以节省显存')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--deterministic', action='store_true')

    # HSTU 架构参数
    parser.add_argument('--encoder_type', default='hstu', type=str, choices=['transformer', 'hstu'])
    parser.add_argument('--dqk', default=128, type=int, help='每头 QK 维度')
    parser.add_argument('--dv', default=128, type=int, help='每头 V/U 维度')
    parser.add_argument('--use_rel_bias', default=1, type=int, choices=[0, 1], help='是否使用相对位置/时间偏置')
    parser.add_argument('--rel_pos_buckets', default=32, type=int, help='相对位置分桶数')
    parser.add_argument('--rel_time_buckets', default=32, type=int, help='相对时间分桶数（0表示不使用）')
    parser.add_argument('--hstu_activation', default='softmax', type=str, choices=['silu', 'softmax'], help='HSTU 激活函数')
    parser.add_argument('--norm_type', default='rmsnorm', type=str, choices=['layernorm', 'rmsnorm'], help='归一化类型')
    
    # ID Embedding switches
    parser.add_argument('--use_user_id_emb', default=1, type=int, choices=[0, 1], help='是否使用用户ID embedding')
    parser.add_argument('--use_item_id_emb', default=1, type=int, choices=[0, 1], help='是否使用物品ID embedding')
     # Dynamic feature importance (SENet)
    parser.add_argument('--use_senet', action='store_true', default=True, help='启用SENet动态特征重要性')
    parser.add_argument('--no_use_senet', dest='use_senet', action='store_false', help='关闭SENet动态特征重要性')
    parser.add_argument('--senet_reduction', default=4, type=int, help='SENet通道缩减比')
   
    # RoPE 相关参数
    parser.add_argument('--use_rope', default=1, type=int, choices=[0, 1], help='是否在注意力中使用RoPE')
    parser.add_argument('--rope_theta', default=10000.0, type=float, help='RoPE 基数 theta')
    parser.add_argument('--rope_dim', default=0, type=int, help='RoPE 作用维度，0表示等于dqk')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # Training data mode
    parser.add_argument('--train_item_only', action='store_true',
                        help='训练时不使用序列特征，anchor 使用 item_feat_dict 特征（无用户/时间分量）')

    # Time feature switches and params
    parser.add_argument('--use_timefeat', action='store_true',
                        help='Enable time features (hour/weekday/week/time-diff)')
    parser.add_argument('--time_diff_log_base', type=float, default=2.0,
                        help='Log base for diff bucketing (>1); used if bucketization enabled')
    parser.add_argument('--time_diff_max_bucket', type=int, default=63,
                        help='Max bucket id for time diff (0..max_bucket)')

    # Ranking loss switches and params
    parser.add_argument('--use_rankloss', default=0, type=int, choices=[0, 1], help='是否使用排序损失')
    parser.add_argument('--rankloss_mode', default='pairwise', choices=['pairwise', 'listwise'])
    parser.add_argument('--rankloss_tau', type=float, default=0.5)
    parser.add_argument('--rankloss_weights', type=str, default='1.0,0.4,0.25,0.1')
    parser.add_argument('--lambda_rank', type=float, default=0.6)
    # popularity / action definitions
    parser.add_argument('--head_ratio', type=float, default=0.3)
    parser.add_argument('--pop_beta', type=float, default=15.0)
    parser.add_argument('--click_actions', type=str, default='1')
    parser.add_argument('--expo_actions', type=str, default='0')
    
    # New interface: apply 50% dropout to ID and item embeddings on the first epoch only
    parser.add_argument('--first_epoch_id_item_dropout', action='store_true',default=True,
                        help='If set, applies 50% dropout to user-id and item-id embeddings during the first epoch of this run only')

    # CTR features switches/paths
    parser.add_argument('--use_ctr', action='store_true', help='Enable user/item global CTR continuous features')
    parser.add_argument('--use_userctr', action='store_true', help='Enable user CTR feature')
    parser.add_argument('--use_itemctr', action='store_true', help='Enable item CTR feature')
    parser.add_argument('--ctr_dir', type=str, default=None, help='Directory containing user_ctr.json/item_ctr.json/ctr_meta.json')

    # InfoNCE weighting/temperature (action-aware)
    parser.add_argument('--nce_weight_scheme', type=str, choices=['none', 'fixed', 'cb', 'focal'], default='fixed',
                        help='样本加权方案：none=不加权; fixed=固定权重; cb=按先验类平衡; focal=配合focal_gamma')
    parser.add_argument('--w_click', type=float, default=3.0, help='点击样本权重')
    parser.add_argument('--w_expo', type=float, default=0.5, help='曝光样本权重')
    parser.add_argument('--weight_cap', type=float, default=6.0, help='类平衡比值上限')
    parser.add_argument('--two_tau', action='store_true', help='按动作类型使用两套温度，仅训练时生效')
    parser.add_argument('--tau_click', type=float, default=None, help='点击样本温度（若未设则回退到 --temperature）')
    parser.add_argument('--tau_expo', type=float, default=None, help='曝光样本温度（若未设则回退到 --temperature）')
    parser.add_argument('--focal_gamma', type=float, default=0.0, help='Focal 指数；0 表示关闭')
    args = parser.parse_args()

    return args


def compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask, temperature, writer, global_step, max_negs: int = 10240,
                         sample_weights: torch.Tensor = None, sample_tau: torch.Tensor = None, focal_gamma: float = 0.0):
    """
    InfoNCE with in-batch negatives (only neg_embs), cosine similarity, masked padding, labels=0.
    Args:
        seq_embs: [B, T, D]
        pos_embs: [B, T, D]
        neg_embs: [B, T, D]
        loss_mask: [B, T] with 1 at valid item positions
        temperature: float
        sample_weights: optional [B, T] per-sample weights
        sample_tau: optional [B, T] per-sample temperatures
        focal_gamma: float, focal reweighting exponent
    Returns:
        loss: scalar tensor
    """
    device = seq_embs.device
    dtype = seq_embs.dtype
    valid_mask = loss_mask.bool()
    num_valid = valid_mask.sum()
    if num_valid == 0:
        zero_loss = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
        return (
            zero_loss,
            torch.tensor(0.0, device=device),  # acc1
            torch.tensor(0.0, device=device),  # acc10
            0,
            0,
        )

    # Compute similarity and CE in FP32 to enhance contrast under mixed precision
    with torch.cuda.amp.autocast(enabled=False):
        anchors_f = F.normalize(seq_embs[valid_mask].float(), dim=-1)  # [N, D]
        positives_f = F.normalize(pos_embs[valid_mask].float(), dim=-1)  # [N, D]
        neg_pool_f = F.normalize(neg_embs[valid_mask].float(), dim=-1)  # [K, D]

        # 固定负样本池上限，随机子采样，降低K波动
        if neg_pool_f.size(0) > max_negs:
            idx = torch.randperm(neg_pool_f.size(0), device=device)[:max_negs]
            neg_pool_f = neg_pool_f[idx]

        N = anchors_f.size(0)
        K_eff = neg_pool_f.size(0)

        # Positive logits and similarities
        pos_sim = (anchors_f * positives_f).sum(dim=-1)  # [N]
        if sample_tau is not None:
            tau_inv = (1.0 / sample_tau[valid_mask].float()).to(dtype=anchors_f.dtype, device=anchors_f.device)  # [N]
        else:
            tau_inv = torch.full_like(pos_sim, 1.0 / float(temperature))  # [N]
        pos_logit = pos_sim * tau_inv                                   # [N]

        # Streaming log-sum-exp over negatives to avoid allocating [N, K]
        # Initialize with the positive term
        m = pos_logit.clone()                # row-wise max over processed logits
        s = torch.ones_like(pos_logit)       # row-wise sumexp normalized by m (exp(pos - m) = 1)
        max_neg = torch.full_like(pos_logit, float('-inf'))  # for acc@1
        greater_than_pos_cnt = torch.zeros_like(pos_logit, dtype=torch.long)  # for acc@10

        # For logging neg mean similarity without building full matrix
        neg_sum_sim_total = 0.0
        neg_num_total = 0

        chunk = 1024 if K_eff >= 1024 else max(1, K_eff)
        if K_eff > 0:
            for st in range(0, K_eff, chunk):
                ed = min(K_eff, st + chunk)
                neg_chunk = neg_pool_f[st:ed]                     # [k_c, D]
                # Similarity (no temperature) for logging
                neg_sims_chunk = anchors_f @ neg_chunk.T          # [N, k_c]
                neg_sum_sim_total += float(neg_sims_chunk.sum().item())
                neg_num_total += (N * (ed - st))
                # Logits for loss/metrics
                if sample_tau is not None:
                    neg_logits_chunk = neg_sims_chunk * tau_inv.unsqueeze(1)  # [N, k_c]
                else:
                    neg_logits_chunk = neg_sims_chunk / float(temperature)    # [N, k_c]

                # Update acc@1 helper (row-wise max over negatives)
                c_m, _ = neg_logits_chunk.max(dim=1)              # [N]
                max_neg = torch.maximum(max_neg, c_m)

                # Update acc@10 helper: count negatives strictly greater than pos
                greater_than_pos_cnt += (neg_logits_chunk > pos_logit.unsqueeze(1)).sum(dim=1)

                # Streaming log-sum-exp merge: combine (m, s) with current chunk
                # s_c = sum(exp(neg - c_m)) per row
                s_c = torch.exp(neg_logits_chunk - c_m.unsqueeze(1)).sum(dim=1)  # [N]
                new_m = torch.maximum(m, c_m)
                s = s * torch.exp(m - new_m) + s_c * torch.exp(c_m - new_m)
                m = new_m

        # Final log-sum-exp over [pos] U [all negs]
        log_denom = m + torch.log(s)
        log_prob_pos = pos_logit - log_denom
        # Weights
        if sample_weights is not None:
            w = sample_weights[valid_mask].float()
            Z = torch.clamp(w.sum(), min=1e-12)
        else:
            w = torch.ones_like(log_prob_pos)
            Z = float(N)

        # Optional focal reweighting
        if focal_gamma and float(focal_gamma) > 0.0:
            pt = torch.clamp(log_prob_pos.exp(), min=1e-12, max=1.0)
            w = w * torch.pow((1.0 - pt), float(focal_gamma))

        loss = -(w * log_prob_pos).sum() / Z

        # Logging
        writer.add_scalar("Model/nce_pos_logits", float(pos_sim.mean().item()), global_step)
        if K_eff > 0 and neg_num_total > 0:
            writer.add_scalar("Model/nce_neg_logits", float(neg_sum_sim_total / float(neg_num_total)), global_step)
        writer.add_scalar("Model/neg_pool_size", float(K_eff), global_step)
        writer.add_scalar("Data/valid_tokens", float(N), global_step)
        writer.add_scalar("Loss/nce_weighted", float(loss.item()), global_step)

    # Metrics: acc@1 via pos versus max_negative; acc@10 via strictly greater count
    acc1 = (pos_logit >= max_neg).float().mean() if K_eff > 0 else torch.ones((), device=device, dtype=pos_logit.dtype)
    acc10 = (greater_than_pos_cnt < 10).float().mean() if K_eff > 0 else torch.ones((), device=device, dtype=pos_logit.dtype)

    return loss, acc1, acc10, int(N), int(K_eff)


def lr_for_step(step: int, total_steps: int, initial_lr: float, min_lr: float = 1e-6, warmup_steps: int = 0) -> float:
    """
    Warmup + Cosine (串联) learning rate scheduler based on optimizer update step.
    Args:
        step: current global step (0-indexed)
        total_steps: total number of optimizer update steps
        initial_lr: initial learning rate
        min_lr: minimum learning rate
    Returns:
        current learning rate
    """
    if total_steps <= 0:
        return initial_lr
    # Linear warmup phase
    if warmup_steps > 0 and step < warmup_steps:
        return initial_lr * float(step + 1) / float(max(1, warmup_steps))
    # Cosine on the remaining steps
    rem_total = max(1, total_steps - max(0, warmup_steps))
    t = min(max(0, step - max(0, warmup_steps)), rem_total)
    cos_decay = 0.5 * (1 + np.cos(np.pi * t / rem_total))
    return min_lr + (initial_lr - min_lr) * cos_decay

if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    # Early and rank-aware seeding (DDP-safe if RANK is set)
    rank = int(os.environ.get('RANK', 0))
    fix_random_seeds(args.seed + rank, deterministic=args.deterministic)
    worker_init_fn = make_worker_init_fn(args.seed)
    dl_generator = make_generator(args.seed, rank)

    # Read CTR prior if available (for class-balanced weighting)
    global_ctr = None
    try:
        if args.ctr_dir is not None:
            meta_path = Path(args.ctr_dir) / 'ctr_meta.json'
            if meta_path.is_file():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    _meta = json.load(f)
                    global_ctr = float(_meta.get('global_ctr', 0.1))
    except Exception:
        global_ctr = None
    base_dataset = UserSeqDatasetPacked(data_path, args)
    # Deterministic split via global seed (already set). Optionally could pass generator=dl_generator
    train_dataset, valid_dataset = torch.utils.data.random_split(base_dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        collate_fn=collate_fn_packed,
        generator=dl_generator,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        collate_fn=collate_fn_packed,
        generator=dl_generator,
        worker_init_fn=worker_init_fn,
    )
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_statistics, feat_types = base_dataset.feat_statistics, base_dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    if getattr(args, 'train_item_only', False):
        print("[TRAIN] item-only anchor mode enabled: using item_feat_dict features only (no user/time)")

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # math/precision knobs
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Optimizer with no_weight_decay for biases/norm/1D params
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or ('bias' in n) or ('norm' in n.lower()):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {'params': decay, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    try:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.98), fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    # Pre-compute total optimizer update steps for global-step LR scheduling
    updates_per_epoch = max(1, len(train_loader) // max(1, args.gradient_accumulation_steps))
    total_update_steps = updates_per_epoch * args.num_epochs
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        # ID/Item embedding dropout退火（首个epoch 0.5 -> 最后0.0 线性）
        if args.first_epoch_id_item_dropout:
            p0, p1 = 0.5, 0.0
            denom = max(1, (args.num_epochs - epoch_start_idx))
            ratio = float(max(0, epoch - epoch_start_idx)) / float(denom)
            model.set_id_item_dropout(p0 + (p1 - p0) * ratio)
        else:
            model.set_id_item_dropout(0.0)
        # Ensure DistributedSampler (if any) shuffles deterministically per-epoch
        try:
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
        except Exception:
            pass
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Packed forward; model handles device transfer
            seq_embs, pos_embs, neg_embs, loss_mask = model.forward_packed(batch)
            # Build per-sample weights/temperatures for action-aware NCE
            sample_weights = None
            sample_tau = None
            try:
                if 'next_action_type' in batch:
                    act = batch['next_action_type'].to(seq_embs.device)
                    if args.nce_weight_scheme != 'none':
                        if args.nce_weight_scheme == 'fixed':
                            wc, we = float(args.w_click), float(args.w_expo)
                        elif args.nce_weight_scheme == 'cb':
                            pi = global_ctr if global_ctr is not None else 0.1
                            ratio = (1.0 - pi) / max(pi, 1e-12)
                            ratio = min(ratio, float(args.weight_cap))
                            wc, we = ratio, 1.0
                        else:  # focal uses base fixed weights
                            wc, we = float(args.w_click), float(args.w_expo)
                        sample_weights = torch.where(
                            act == 1,
                            torch.full_like(act, wc, dtype=torch.float32),
                            torch.full_like(act, we, dtype=torch.float32),
                        )
                        sample_weights = sample_weights * loss_mask
                    if args.two_tau:
                        tc = float(args.tau_click) if args.tau_click is not None else float(args.temperature)
                        te = float(args.tau_expo) if args.tau_expo is not None else float(args.temperature)
                        sample_tau = torch.where(
                            act == 1,
                            torch.full_like(act, tc, dtype=torch.float32),
                            torch.full_like(act, te, dtype=torch.float32),
                        )
            except Exception:
                sample_weights = None
                sample_tau = None
            
            # 梯度累积：只在累计开始清零梯度
            if step % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
                
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, acc1, acc10, num_valid, neg_pool_size = compute_infonce_loss(
                    seq_embs, pos_embs, neg_embs, loss_mask, args.temperature, writer, global_step,
                    sample_weights=sample_weights, sample_tau=sample_tau, focal_gamma=args.focal_gamma
                )
                total_loss = loss

                # Rank loss (optional)
                if args.use_rankloss and 'rank_ids' in batch:
                    scores_mask = model.score_rank_from_packed(batch, seq_embs)
                    if scores_mask is not None:
                        scores, rmask = scores_mask
                        weights = tuple(float(x) for x in args.rankloss_weights.split(',') if x != '')
                        if args.rankloss_mode == 'pairwise':
                            L_rank = pairwise_rankloss(scores, rmask, tau=args.rankloss_tau, weights=weights)
                        else:
                            L_rank = listwise_rankloss(scores, rmask, tau=args.rankloss_tau)
                        writer.add_scalar("Loss/rank", L_rank.item(), global_step)
                        total_loss = total_loss + args.lambda_rank * L_rank

            # 梯度累积：缩放损失
            total_loss = total_loss / args.gradient_accumulation_steps

            # 动态LR（按更新步）
            current_lr = lr_for_step(
                global_step, total_update_steps, args.lr, min_lr=1e-6, warmup_steps=args.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            # metrics & logging
            B, T = batch['seq_ids'].shape[0], batch['seq_ids'].shape[1]
            expected_bxt = B * T
            is_pool_eq_bxt = int(neg_pool_size == expected_bxt)
            log_json = json.dumps(
                {
                    'global_step': global_step,
                    'loss': float(loss.item()),
                    'acc@1': float(acc1.item()),
                    'acc@10': float(acc10.item()),
                    'epoch': epoch,
                    'lr': current_lr,
                    'num_valid': int(num_valid),
                    'negatives_in_pool': int(neg_pool_size),
                    'expected_BxT': int(expected_bxt),
                    'is_pool_eq_BxT': is_pool_eq_bxt,
                    'time': time.time(),
                    'seed': int(args.seed),
                    'deterministic': bool(args.deterministic),
                }
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', total_loss.item(), global_step)
            writer.add_scalar('LR/train', current_lr, global_step)
            writer.add_scalar('Acc1/train', acc1.item(), global_step)
            writer.add_scalar('Acc10/train', acc10.item(), global_step)

            # 梯度累积：只在累积步数完成时更新
            for param in model.item_emb.parameters():
                total_loss = total_loss + args.l2_emb * torch.norm(param)

            # 数值检查（loss是否为有限数）
            if not torch.isfinite(total_loss):
                print(f"[WARN] non-finite loss at step={global_step}, skip update")
                optimizer.zero_grad(set_to_none=True)
                continue

            # 反传
            total_loss.backward()
            # 记录总loss（包含正则项）
            writer.add_scalar('Loss/train_total', total_loss.item(), global_step)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                writer.add_scalar('Grad/grad_norm', float(grad_norm), global_step)
                global_step += 1

        # 处理epoch尾包：若最后不足累计步，仍需执行一次更新
        remainder = len(train_loader) % max(1, args.gradient_accumulation_steps)
        if remainder != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            # 设置当前学习率再更新
            current_lr = lr_for_step(
                global_step, total_update_steps, args.lr, min_lr=1e-6, warmup_steps=args.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            if torch.isfinite(grad_norm):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar('Grad/grad_norm', float(grad_norm), global_step)
            global_step += 1

        model.eval()
        valid_loss_sum = 0
        valid_acc10_sum = 0
        valid_acc1_sum = 0
        valid_num_batches = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            with torch.no_grad():
                seq_embs, pos_embs, neg_embs, loss_mask = model.forward_packed(batch)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss, acc1, acc10, num_valid, neg_pool_size = compute_infonce_loss(
                        seq_embs, pos_embs, neg_embs, loss_mask, args.temperature, writer, global_step
                    )
                    total_loss = loss
                    if args.use_rankloss and 'rank_ids' in batch:
                        scores_mask = model.score_rank_from_packed(batch, seq_embs)
                        if scores_mask is not None:
                            scores, rmask = scores_mask
                            weights = tuple(float(x) for x in args.rankloss_weights.split(',') if x != '')
                            if args.rankloss_mode == 'pairwise':
                                L_rank = pairwise_rankloss(scores, rmask, tau=args.rankloss_tau, weights=weights)
                            else:
                                L_rank = listwise_rankloss(scores, rmask, tau=args.rankloss_tau)
                            total_loss = total_loss + args.lambda_rank * L_rank
            valid_loss_sum += total_loss.item()
            valid_acc1_sum += acc1.item()
            valid_acc10_sum += acc10.item()
            valid_num_batches += 1
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        if valid_num_batches > 0:
            writer.add_scalar('Acc1/valid', valid_acc1_sum / valid_num_batches, global_step)
            writer.add_scalar('Acc10/valid', valid_acc10_sum / valid_num_batches, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
