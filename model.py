from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Avoid importing dataset at module import time to prevent potential circular deps in some envs.
# Local save helper identical to dataset.save_emb
def _save_emb_local(emb, save_path):
    import struct
    import numpy as np
    from pathlib import Path
    num_points = emb.shape[0]
    num_dimensions = emb.shape[1]
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        if isinstance(emb, np.ndarray):
            emb.tofile(f)
        else:
            # tensor -> numpy
            emb.detach().cpu().numpy().tofile(f)


class RMSNorm(torch.nn.Module):
    """
    RMSNorm: Root Mean Square Layer Normalization
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # x: [*, dim]
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class SENetGate(torch.nn.Module):
    """
    Squeeze-and-Excitation style gating over a list of feature blocks.
    - Treat each feature block (e.g., item id emb, sparse emb, mm emb, etc.) as one "channel".
    - Squeeze: global average over the last dimension of each block -> [B, T, 1]
    - Excite: two-layer MLP across blocks -> per-block weights in [0, 1]
    - Apply: multiply each block by its scalar weight (per step), preserving shapes

    Args:
        num_blocks: number of feature blocks to gate
        reduction: reduction ratio for bottleneck hidden size
    """
    def __init__(self, num_blocks: int, reduction: int = 4):
        super().__init__()
        self.num_blocks = int(num_blocks)
        hidden = max(1, self.num_blocks // max(1, int(reduction)))
        self.fc1 = torch.nn.Linear(self.num_blocks, hidden)
        self.fc2 = torch.nn.Linear(hidden, self.num_blocks)
        self.act = torch.nn.SiLU()

    def forward(self, feature_list):
        if feature_list is None or len(feature_list) == 0:
            return feature_list
        # Expect list of tensors with shapes [B, T, D_i]
        # Squeeze across feature dim -> [B, T, 1] per block, then concat to [B, T, M]
        pooled = []
        for x in feature_list:
            # Some continual features may have D_i == 1; mean still valid
            pooled.append(x.mean(dim=-1, keepdim=True))
        S = torch.cat(pooled, dim=-1)  # [B, T, M]
        B, T, M = S.shape
        # Excitation across blocks per time step
        W = self.fc2(self.act(self.fc1(S.view(B * T, M))))  # [B*T, M]
        W = torch.sigmoid(W).view(B, T, M)  # [B, T, M]
        # Apply weights
        gated = []
        for i, x in enumerate(feature_list):
            gated.append(x * W[..., i].unsqueeze(-1))
        return gated


class HSTUBlock(torch.nn.Module):
    """
    HSTU Block: Hierarchical State Tracking Unit (Stable Version)
    Replaces MHA + FFN with a unified block containing pointwise projection, 
    aggregation attention, pointwise transformation and gating.
    
    This version prioritizes numerical stability over performance.
    """
    def __init__(self, hidden_units, num_heads, dqk=64, dv=64, dropout=0.0,
                 use_rel_bias=False, rel_pos_buckets=32, rel_time_buckets=0,
                 norm_first=True, attention_mode='softmax', gate_threshold=0.1,
                 use_layernorm=False, gate_residual=True,
                 use_rope=False, rope_theta=10000.0, rope_dim=0):
        super().__init__()
        H, h = hidden_units, num_heads
        self.h, self.dqk, self.dv = h, dqk, dv
        self.norm_first = norm_first
        self.attention_mode = attention_mode
        self.gate_threshold = gate_threshold
        self.gate_residual = gate_residual
        # Rotary Positional Embedding
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        # 0 -> use dqk; otherwise clamp to dqk and even number
        self.rope_dim = int(rope_dim)
        
        # 使用最稳定的配置：只使用softmax注意力
        self.attention_mode = 'softmax'
        
        # 简化的线性层
        self.q_proj = torch.nn.Linear(H, h * dqk, bias=False)
        self.k_proj = torch.nn.Linear(H, h * dqk, bias=False)
        self.v_proj = torch.nn.Linear(H, h * dv, bias=False)
        self.u_proj = torch.nn.Linear(H, h * dv, bias=False)
        
        # 输出投影层
        self.out = torch.nn.Linear(h * dv, H, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        # Position-wise FFN (adds expressiveness similar to Transformer FFN)
        ffn_hidden = 4 * H
        self.ffn1 = torch.nn.Linear(H, ffn_hidden, bias=True)
        self.ffn2 = torch.nn.Linear(ffn_hidden, H, bias=True)
        self.ffn_dropout = torch.nn.Dropout(dropout)
        
        # 归一化层
        if use_layernorm:
            self.in_norm = torch.nn.LayerNorm(H, eps=1e-6)
        else:
            self.in_norm = RMSNorm(H)
        
        # 门控权重参数
        if gate_residual:
            self.gate_weight = torch.nn.Parameter(torch.ones(1) * 0.1)
            self.residual_weight = torch.nn.Parameter(torch.ones(1) * 0.9)
        
        # 初始化权重
        for module in [self.q_proj, self.k_proj, self.v_proj, self.u_proj, self.out, self.ffn1, self.ffn2]:
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        # 相对偏置相关
        self.use_rel_bias = use_rel_bias
        if use_rel_bias:
            self.rel_pos_buckets = rel_pos_buckets
            self.rel_time_buckets = rel_time_buckets
            self.rel_pos_bias = torch.nn.Embedding(rel_pos_buckets, 1)
            torch.nn.init.normal_(self.rel_pos_bias.weight, mean=0.0, std=0.01)
            if rel_time_buckets > 0:
                self.rel_time_bias = torch.nn.Embedding(rel_time_buckets, 1)
                torch.nn.init.normal_(self.rel_time_bias.weight, mean=0.0, std=0.01)

    @staticmethod
    def _rotate_every_two(x):
        """Rotate pairs (x0,x1)->(-x1,x0) along last dim."""
        x_shape = x.shape
        x = x.view(*x_shape[:-1], -1, 2)
        x0 = x[..., 0]
        x1 = x[..., 1]
        x_rot = torch.stack((-x1, x0), dim=-1)
        return x_rot.view(*x_shape)

    def _apply_rope(self, q_or_k, T, rope_dim):
        if rope_dim <= 0:
            return q_or_k
        device = q_or_k.device
        dtype = q_or_k.dtype
        half = rope_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, rope_dim, 2, device=device, dtype=dtype) / rope_dim))
        t = torch.arange(T, device=device, dtype=dtype)
        freqs = torch.einsum('t,d->td', t, inv_freq)  # [T, half]
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)  # [1,1,T,rope_dim]
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)  # [1,1,T,rope_dim]
        x1 = q_or_k[..., :rope_dim]
        x2 = q_or_k[..., rope_dim:]
        x1_rot = x1 * cos + self._rotate_every_two(x1) * sin
        return torch.cat([x1_rot, x2], dim=-1)

    def _build_rab(self, pos_ids, time_ids, T):
        """Build relative position/time bias"""
        if not self.use_rel_bias:
            return None
            
        if pos_ids is None and time_ids is None:
            return None
            
        if pos_ids is not None:
            device = pos_ids.device
        elif time_ids is not None:
            device = time_ids.device
        else:
            return None
        
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        
        rab = torch.zeros(1, 1, T, T, device=device)
        
        if pos_ids is not None:
            # 确保pos_ids的长度与T匹配
            if pos_ids.shape[-1] != T:
                pos_ids = pos_ids[..., :T]  # 截取前T个元素
            rel_pos = j - i
            rel_pos = torch.clamp(rel_pos, -self.rel_pos_buckets//2, self.rel_pos_buckets//2-1)
            rel_pos = rel_pos + self.rel_pos_buckets//2
            pos_bias = self.rel_pos_bias(rel_pos).squeeze(-1)
            rab = rab + pos_bias.unsqueeze(0).unsqueeze(0)
        
        if time_ids is not None and self.rel_time_buckets > 0:
            # 确保time_ids的长度与T匹配
            if time_ids.shape[-1] != T:
                time_ids = time_ids[..., :T]  # 截取前T个元素
            # 处理batch维度：time_ids应该是[B, T]，我们需要[B, T, T]的time_diff
            if time_ids.dim() == 2:  # [B, T]
                time_diff = time_ids.unsqueeze(1) - time_ids.unsqueeze(2)  # [B, T, T]
            else:  # [T]
                time_diff = time_ids.unsqueeze(0) - time_ids.unsqueeze(1)  # [T, T]
                time_diff = time_diff.unsqueeze(0)  # [1, T, T]
            # 数值稳定与dtype安全：在浮点上执行 log1p(|Δt|)
            time_diff_f = torch.abs(time_diff).to(dtype=torch.float32)
            time_diff_f = torch.log1p(time_diff_f)
            time_diff_f = torch.clamp(time_diff_f, 0.0, float(self.rel_time_buckets - 1))
            time_diff = time_diff_f.to(dtype=torch.long)
            time_bias = self.rel_time_bias(time_diff).squeeze(-1)  # [B, T, T] 或 [1, T, T]
            # 确保time_bias的形状与rab匹配 [1, 1, T, T]
            if time_bias.dim() == 3:
                time_bias = time_bias.unsqueeze(0)  # [1, B, T, T] 或 [1, 1, T, T]
            rab = rab + time_bias
        
        return rab

    def _stable_attention(self, Q, K, V, mask=None, rab=None):
        """Stable attention mechanism using only softmax"""
        B, h, T, d = Q.shape
        
        # 计算注意力分数
        scale = d ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        if rab is not None:
            # rab的形状应该是[B, h, T, T]来匹配scores
            if rab.shape[0] == 1:  # [1, B, T, T] -> [B, h, T, T]
                rab = rab.squeeze(0)  # [B, T, T]
                rab = rab.unsqueeze(1)  # [B, 1, T, T]
                rab = rab.expand(-1, scores.shape[1], -1, -1)  # [B, h, T, T]
            scores = scores + rab
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask.logical_not(), float('-inf'))
        
        # 数值稳定性：clamp scores
        scores = torch.clamp(scores, min=-50, max=50)
        
        # 使用softmax
        A = F.softmax(scores, dim=-1)
        
        # 最终clamping
        A = torch.clamp(A, min=1e-8, max=1.0)
        
        return torch.matmul(A, V)

    def forward(self, x, attn_mask=None, pos_ids=None, time_ids=None):
        # 输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: input contains NaN/Inf")
            return x
        
        # Pre-norm 或 Post-norm
        if self.norm_first:
            z = self.in_norm(x)
        else:
            z = x
        
        B, T, H = z.shape
        
        # 分别计算Q, K, V, U
        Q = self.q_proj(z).view(B, T, self.h, self.dqk).transpose(1, 2)
        K = self.k_proj(z).view(B, T, self.h, self.dqk).transpose(1, 2)
        V = self.v_proj(z).view(B, T, self.h, self.dv).transpose(1, 2)
        U = self.u_proj(z).view(B, T, self.h, self.dv).transpose(1, 2)
        
        # 数值检查（确保实际生效）
        if torch.isnan(Q).any() or torch.isinf(Q).any():
            print(f"Warning: Q contains NaN/Inf, clamping")
            Q = torch.clamp(Q, min=-100, max=100)
        if torch.isnan(K).any() or torch.isinf(K).any():
            print(f"Warning: K contains NaN/Inf, clamping")
            K = torch.clamp(K, min=-100, max=100)
        if torch.isnan(V).any() or torch.isinf(V).any():
            print(f"Warning: V contains NaN/Inf, clamping")
            V = torch.clamp(V, min=-100, max=100)
        if torch.isnan(U).any() or torch.isinf(U).any():
            print(f"Warning: U contains NaN/Inf, clamping")
            U = torch.clamp(U, min=-100, max=100)
        
        # 应用 RoPE 到 Q/K（若开启）
        if self.use_rope:
            rope_dim = self.rope_dim if self.rope_dim > 0 else self.dqk
            rope_dim = min(rope_dim, self.dqk)
            rope_dim = rope_dim - (rope_dim % 2)  # ensure even
            if rope_dim > 0:
                Q = self._apply_rope(Q, T, rope_dim)
                K = self._apply_rope(K, T, rope_dim)

        # 构建相对偏置
        rab = self._build_rab(pos_ids, time_ids, T)
        
        # 注意力计算
        pooled = self._stable_attention(Q, K, V, attn_mask, rab)
        
        # 门控机制
        pooled = pooled.transpose(1, 2).contiguous().view(B, T, -1)
        U_reshaped = U.transpose(1, 2).contiguous().view(B, T, -1)
        
        # 稳定的门控（不再强制下限，保留 sigmoid 动态范围）
        gate_values = torch.sigmoid(U_reshaped)
        
        gated = gate_values * pooled
        
        # 输出投影
        y = self.out(gated)
        y = self.dropout(y)
        
        # 残差连接
        if self.gate_residual:
            # Learnable scalars (sigmoid to [0,1]) for a softer mix
            gate_w = torch.sigmoid(self.gate_weight)
            residual_w = torch.sigmoid(self.residual_weight)
            output = residual_w * x + gate_w * y
        else:
            output = x + y

        # Position-wise FFN + residual
        ffn_inter = self.ffn1(output)
        ffn_inter = F.gelu(ffn_inter)
        ffn_inter = self.ffn_dropout(ffn_inter)
        ffn_out = self.ffn2(ffn_inter)
        ffn_out = self.dropout(ffn_out)
        output = output + ffn_out

        # 最终归一化（post-norm 情况）
        if self.norm_first:
            return output
        else:
            return self.in_norm(output)


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.grad_checkpointing = bool(getattr(args, 'grad_checkpointing', 0))
        # Embedding usage switches (1 on by default)
        self.use_user_id_emb = bool(getattr(args, 'use_user_id_emb', 1))
        self.use_item_id_emb = bool(getattr(args, 'use_item_id_emb', 1))
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        # Training mode: item-only anchors (no sequence/user/time fusion)
        self.train_item_only = bool(getattr(args, 'train_item_only', False))

        # 根据编码器类型选择不同的架构
        self.encoder_type = getattr(args, 'encoder_type', 'hstu')  # 默认使用 HSTU
        
        if self.encoder_type == 'hstu':
            # HSTU 架构：统一的 blocks 替换 MHA + FFN
            self.blocks = torch.nn.ModuleList()
            for _ in range(args.num_blocks):
                new_block = HSTUBlock(
                    hidden_units=args.hidden_units,
                    num_heads=args.num_heads,
                    dqk=getattr(args, 'dqk', 64),
                    dv=getattr(args, 'dv', 64),
                    dropout=args.dropout_rate,
                    use_rel_bias=getattr(args, 'use_rel_bias', False),
                    rel_pos_buckets=getattr(args, 'rel_pos_buckets', 32),
                    rel_time_buckets=getattr(args, 'rel_time_buckets', 0),
                    norm_first=args.norm_first,
                    attention_mode=getattr(args, 'attention_mode', 'softmax'),
                    gate_threshold=getattr(args, 'gate_threshold', 0.1),
                    use_layernorm=getattr(args, 'use_layernorm', False),
                    gate_residual=getattr(args, 'gate_residual', True),
                    use_rope=bool(getattr(args, 'use_rope', 0)),
                    rope_theta=float(getattr(args, 'rope_theta', 10000.0)),
                    rope_dim=int(getattr(args, 'rope_dim', 0)),
                )
                self.blocks.append(new_block)
        else:
            # 原始 Transformer 架构
            self.attention_layernorms = torch.nn.ModuleList()
            self.attention_layers = torch.nn.ModuleList()
            self.forward_layernorms = torch.nn.ModuleList()
            self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        # Continuous features (user/item) are passed as raw scalars and fused via tiny MLPs
        self.use_ctr = bool(getattr(args, 'use_ctr', False))
        self.use_userctr = bool(getattr(args, 'use_userctr', False)) or self.use_ctr
        self.use_itemctr = bool(getattr(args, 'use_itemctr', False)) or self.use_ctr
        self.ctr_scale_factor = float(getattr(args, 'ctr_scale_factor', 0.1))

        # determine continual dims (count of IDs)
        user_cont_dims = len(self.USER_CONTINUAL_FEAT)
        item_cont_dims = len(self.ITEM_CONTINUAL_FEAT)

        cont_user_blocks = 1 if user_cont_dims > 0 else 0
        cont_item_blocks = 1 if item_cont_dims > 0 else 0
        # Each sparse/array/id/mm/continual block contributes hidden_units after embedding/MLP
        userdim = args.hidden_units * (
            len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT) + cont_user_blocks
        )
        itemdim = args.hidden_units * (
            len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT) + len(self.ITEM_EMB_FEAT) + cont_item_blocks
        )

        # time dims (sparse via embedding; continual as raw scalar)
        timedim = args.hidden_units * (len(self.TIME_SPARSE_FEAT)) + len(self.TIME_CONTINUAL_FEAT)

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)
        # keep a safe input dim when time is disabled
        self.timednn = torch.nn.Linear(timedim if timedim > 0 else 1, args.hidden_units)

        # 根据编码器类型选择归一化层
        if self.encoder_type == 'hstu':
            if getattr(args, 'norm_type', 'layernorm') == 'rmsnorm':
                self.last_layernorm = RMSNorm(args.hidden_units)
            else:
                self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        else:
            self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 原始 Transformer 架构的初始化
        if self.encoder_type == 'transformer':
            for _ in range(args.num_blocks):
                new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
                self.attention_layernorms.append(new_attn_layernorm)

                new_attn_layer = FlashMultiHeadAttention(
                    args.hidden_units, args.num_heads, args.dropout_rate
                )
                self.attention_layers.append(new_attn_layer)

                new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
                self.forward_layernorms.append(new_fwd_layernorm)

                new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
                self.forward_layers.append(new_fwd_layer)

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

        # time sparse embedding tables (id range: 0 as padding, 1..max_bucket as valid)
        for k in self.TIME_SPARSE_FEAT:
            # TIME_SPARSE_FEAT[k] 记录桶数（不含 padding）；+1 预留 padding
            self.sparse_emb[k] = torch.nn.Embedding(self.TIME_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
      # SENet gates for dynamic feature importance (optional)
        self.use_senet = bool(getattr(args, 'use_senet', False))
        if self.use_senet:
            # Number of feature blocks per side (match actual appended blocks)
            # - item side appends: [item_id] + item_sparse[*] + item_array[*] + (1 if any item_continual else 0) + item_emb[*]
            # - user side appends: [user_id placeholder or zeros] is already accounted via user_sparse; user_id emb not included in packed path
            item_cont_blocks = 1 if len(self.ITEM_CONTINUAL_FEAT) > 0 else 0
            user_cont_blocks = 1 if len(self.USER_CONTINUAL_FEAT) > 0 else 0
            item_blocks = 1 + len(self.ITEM_SPARSE_FEAT) + len(self.ITEM_ARRAY_FEAT) + item_cont_blocks + len(self.ITEM_EMB_FEAT)
            user_blocks = 1 + len(self.USER_SPARSE_FEAT) + len(self.USER_ARRAY_FEAT) + user_cont_blocks
            reduction = int(getattr(args, 'senet_reduction', 4))
            self.item_gate = SENetGate(item_blocks, reduction)
            self.user_gate = SENetGate(user_blocks, reduction)

        # Tiny MLPs for continual features (if present)
        hidden_units = args.hidden_units
        dropout_rate = args.dropout_rate
        if user_cont_dims > 0:
            self.user_cont_mlp = torch.nn.Sequential(
                torch.nn.Linear(user_cont_dims, hidden_units),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate),
            )
        else:
            self.user_cont_mlp = None
        if item_cont_dims > 0:
            self.item_cont_mlp = torch.nn.Sequential(
                torch.nn.Linear(item_cont_dims, hidden_units),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate),
            )
        else:
            self.item_cont_mlp = None

    # -------- runtime toggles for ID embeddings --------
    def set_user_id_emb(self, enabled: bool):
        """Enable/disable user ID embedding contribution at runtime."""
        self.use_user_id_emb = bool(enabled)

    def set_item_id_emb(self, enabled: bool):
        """Enable/disable item ID embedding contribution at runtime."""
        self.use_item_id_emb = bool(enabled)

    def set_id_item_dropout(self, p: float):
        """
        Set dropout probability applied to fused sequence embeddings.
        This is used by the training script to apply stronger regularization
        (e.g., 0.5) on the first epoch, then turn it off afterwards.
        """
        p = float(max(0.0, min(1.0, p)))
        self.emb_dropout.p = p

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度
        # Time feature groups (may be empty)
        self.TIME_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types.get('time_sparse', [])}
        self.TIME_CONTINUAL_FEAT = feat_types.get('time_continual', [])

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # Sparse or Continual scalar per time step
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            is_continual = (k in self.ITEM_CONTINUAL_FEAT) or (k in self.USER_CONTINUAL_FEAT) or (k in self.TIME_CONTINUAL_FEAT)
            dtype = np.float32 if is_continual else np.int64
            batch_data = np.zeros((batch_size, max_seq_len), dtype=dtype)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            if not self.use_user_id_emb:
                user_embedding = torch.zeros_like(user_embedding)
            item_embedding = self.item_emb(item_mask * seq)
            if not self.use_item_id_emb:
                item_embedding = torch.zeros_like(item_embedding)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            if not self.use_item_id_emb:
                item_embedding = torch.zeros_like(item_embedding)
            item_feat_list = [item_embedding]

        # batch-process all feature types
        # process sparse/array first; continual handled via dedicated MLP later
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                ]
            )
            # time features only used on sequence path
            time_feat_list = []
            if len(self.TIME_SPARSE_FEAT) + len(self.TIME_CONTINUAL_FEAT) > 0:
                all_feat_types.append((self.TIME_SPARSE_FEAT, 'time_sparse', time_feat_list))
                all_feat_types.append((self.TIME_CONTINUAL_FEAT, 'time_continual', time_feat_list))

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])

            # pre-allocate tensor
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # batch-convert and transfer to GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # Dynamic feature importance via SENet (item-side)
        if getattr(self, 'use_senet', False):
            # Shape guard: only apply gate when channel count matches configured num_blocks
            try:
                if hasattr(self, 'item_gate') and isinstance(self.item_gate, SENetGate):
                    expected = int(self.item_gate.num_blocks)
                    actual = int(len(item_feat_list))
                    if expected == actual:
                        item_feat_list = self.item_gate(item_feat_list)
            except Exception:
                pass
        # merge features
        # Fuse item continual via MLP (if any)
        if len(self.ITEM_CONTINUAL_FEAT) > 0 and isinstance(feature_array, list) and len(feature_array) > 0:
            try:
                cont_tensor = self.feat2tensor(feature_array, list(self.ITEM_CONTINUAL_FEAT)[0]).unsqueeze(-1)
                # If multiple continual ids exist, stack each
                cont_list = []
                for k in self.ITEM_CONTINUAL_FEAT:
                    t = self.feat2tensor(feature_array, k).unsqueeze(-1)
                    cont_list.append(t)
                cont_cat = torch.cat(cont_list, dim=-1)  # [B, T, C_i]
                if self.item_cont_mlp is not None:
                    item_feat_list.append(self.item_cont_mlp(cont_cat))
                else:
                    item_feat_list.append(cont_cat)
            except Exception:
                pass

        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = F.gelu(self.itemdnn(all_item_emb))
        if include_user:
            # Dynamic feature importance via SENet (user-side)
            if getattr(self, 'use_senet', False):
                try:
                    if hasattr(self, 'user_gate') and isinstance(self.user_gate, SENetGate):
                        expected_u = int(self.user_gate.num_blocks)
                        actual_u = int(len(user_feat_list))
                        if expected_u == actual_u:
                            user_feat_list = self.user_gate(user_feat_list)
                except Exception:
                    pass
            # Fuse user continual via MLP (if any)
            if len(self.USER_CONTINUAL_FEAT) > 0 and isinstance(feature_array, list) and len(feature_array) > 0:
                try:
                    cont_list_u = []
                    for k in self.USER_CONTINUAL_FEAT:
                        t = self.feat2tensor(feature_array, k).unsqueeze(-1)
                        cont_list_u.append(t)
                    cont_cat_u = torch.cat(cont_list_u, dim=-1)
                    if self.user_cont_mlp is not None:
                        user_feat_list.append(self.user_cont_mlp(cont_cat_u))
                    else:
                        user_feat_list.append(cont_cat_u)
                except Exception:
                    pass
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = F.gelu(self.userdnn(all_user_emb))
            if len(time_feat_list) > 0:
                all_time = torch.cat(time_feat_list, dim=2) if len(time_feat_list) > 1 else time_feat_list[0]
                time_emb = F.gelu(self.timednn(all_time))
                seqs_emb = all_item_emb + all_user_emb + time_emb
            else:
                seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature, pos_ids=None, time_ids=None):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_ids: 位置ID，用于相对位置偏置 [batch_size, maxlen]
            time_ids: 时间ID，用于相对时间偏置 [batch_size, maxlen]

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        if self.encoder_type == 'hstu':
            # HSTU 架构：使用统一的 blocks，传递时间和位置信息
            if self.grad_checkpointing and self.training:
                # 使用梯度检查点以节省激活显存
                def _wrap(b):
                    def fn(x):
                        return b(x, attn_mask=attention_mask, pos_ids=pos_ids, time_ids=time_ids)
                    return fn
                for block in self.blocks:
                    seqs = torch.utils.checkpoint.checkpoint(_wrap(block), seqs, use_reentrant=False)
            else:
                for block in self.blocks:
                    seqs = block(seqs, attn_mask=attention_mask, pos_ids=pos_ids, time_ids=time_ids)
        else:
            # 原始 Transformer 架构
            for i in range(len(self.attention_layers)):
                if self.norm_first:
                    x = self.attention_layernorms[i](seqs)
                    mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                    seqs = seqs + mha_outputs
                    seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
                else:
                    mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                    seqs = self.attention_layernorms[i](seqs + mha_outputs)
                    seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        """
        训练时调用，返回序列表征与正负样本表征以及loss掩码，用于InfoNCE

        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            log_feats: 序列表征 [batch_size, maxlen, hidden_units]
            pos_embs: 正样本表征 [batch_size, maxlen, hidden_units]
            neg_embs: 负样本表征 [batch_size, maxlen, hidden_units]
            loss_mask: 掩码 [batch_size, maxlen]，1表示计算该位置的loss
        """
        log_feats = self.log2feats(user_item, mask, seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        return log_feats, pos_embs, neg_embs, loss_mask

    # =========================
    # Packed path (tensorized features from DataLoader workers)
    # =========================

    def feat2emb_packed(self, seq_ids, features, token_type=None, include_user=True):
        """
        seq_ids: LongTensor [B, T]
        features: dict containing tensors from collate_fn_packed
        token_type: optional LongTensor [B, T] (1=item, 2=user) for masking item id embedding
        include_user: whether to fuse user-side features
        Returns: [B, T, H]
        """
        device = self.dev
        H = self.item_emb.embedding_dim
        seq_ids = seq_ids.to(device, non_blocking=True)

        # Item id embedding (mask out non-item tokens if token_type provided)
        # Clamp item indices to valid range to avoid OOB gather
        max_item_idx = int(self.item_emb.num_embeddings) - 1
        seq_ids_safe = torch.clamp(seq_ids, 0, max_item_idx)
        item_emb0 = self.item_emb(seq_ids_safe)
        if token_type is not None:
            item_mask = (token_type.to(device) == 1).unsqueeze(-1)
            item_emb0 = item_emb0 * item_mask
        if not self.use_item_id_emb:
            item_emb0 = torch.zeros_like(item_emb0)
        item_feat_list = [item_emb0]

        # Item sparse features
        if 'item_sparse' in features and features['item_sparse'] is not None:
            x = features['item_sparse'].to(device, non_blocking=True)
            # x: [B, T, S_i] where S_i matches insertion order of self.ITEM_SPARSE_FEAT
            for j, k in enumerate(self.ITEM_SPARSE_FEAT):
                emb_tbl = self.sparse_emb[k]
                # Clamp per-feature indices
                max_idx = int(emb_tbl.num_embeddings) - 1
                idx = torch.clamp(x[..., j], 0, max_idx)
                item_feat_list.append(emb_tbl(idx))
        else:
            # keep shape alignment not necessary here (sparse dims are optional in itemdim calculation)
            pass

        # Item array features (optional, if provided); else keep dims by appending zeros for each expected array feat
        if 'item_array' in features and features['item_array'] is not None:
            arr = features['item_array'].to(device, non_blocking=True)  # [B, T, A_i, L_i]
            for j, k in enumerate(self.ITEM_ARRAY_FEAT):
                emb_tbl = self.sparse_emb[k]
                max_idx = int(emb_tbl.num_embeddings) - 1
                idx_seq = torch.clamp(arr[..., j, :], 0, max_idx)
                pooled = emb_tbl(idx_seq).mean(dim=-2)
                item_feat_list.append(pooled)
        else:
            for _ in self.ITEM_ARRAY_FEAT:
                item_feat_list.append(torch.zeros_like(item_emb0))

        # Item continual features (float), expecting shape [B, T, C_i] -> pass MLP then append
        if len(self.ITEM_CONTINUAL_FEAT) > 0:
            if 'item_cont' in features and features['item_cont'] is not None:
                cont = features['item_cont'].to(device, non_blocking=True)  # [B, T, C_i]
                if self.ctr_scale_factor != 1.0:
                    cont = cont * self.ctr_scale_factor
                if self.item_cont_mlp is not None:
                    item_feat_list.append(self.item_cont_mlp(cont))
                else:
                    item_feat_list.append(cont)
            else:
                # preserve gate input dim: append zero block
                item_feat_list.append(torch.zeros_like(item_emb0))

        # Item mm embeddings（允许缺失 -> 零填充，保证 dims 对齐）
        if 'mm' in features and features['mm']:
            included = set()
            for fid in self.ITEM_EMB_FEAT:
                if fid in features['mm'] and features['mm'][fid] is not None:
                    mm = features['mm'][fid].to(device, non_blocking=True)
                    item_feat_list.append(self.emb_transform[fid](mm))
                    included.add(fid)
            for fid in self.ITEM_EMB_FEAT:
                if fid not in included:
                    item_feat_list.append(torch.zeros_like(item_emb0))
        else:
            for _ in self.ITEM_EMB_FEAT:
                item_feat_list.append(torch.zeros_like(item_emb0))

        # Dynamic feature importance via SENet (item-side, packed)
        if getattr(self, 'use_senet', False):
            try:
                if hasattr(self, 'item_gate') and isinstance(self.item_gate, SENetGate):
                    expected = int(self.item_gate.num_blocks)
                    actual = int(len(item_feat_list))
                    if expected == actual:
                        item_feat_list = self.item_gate(item_feat_list)
            except Exception:
                pass
        # Merge item side
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = F.gelu(self.itemdnn(all_item_emb))

        if not include_user:
            return all_item_emb

        # User side features
        user_feat_list = []

        # user sparse
        if 'user_sparse' in features and features['user_sparse'] is not None:
            x = features['user_sparse'].to(device, non_blocking=True)
            for j, k in enumerate(self.USER_SPARSE_FEAT):
                emb_tbl = self.sparse_emb[k]
                max_idx = int(emb_tbl.num_embeddings) - 1
                idx = torch.clamp(x[..., j], 0, max_idx)
                user_feat_list.append(emb_tbl(idx))

        # user id embedding is not available in packed inputs; keep dims by appending one zero block to account for 
        # the "+1" term in userdim (which accounts for user embedding table)
        user_feat_list.append(torch.zeros_like(item_emb0))

        # user array features: if not provided, append zeros to preserve dims
        if 'user_array' in features and features['user_array'] is not None:
            arr = features['user_array'].to(device, non_blocking=True)  # [B, T, A_u, L_u]
            for j, k in enumerate(self.USER_ARRAY_FEAT):
                emb_tbl = self.sparse_emb[k]
                max_idx = int(emb_tbl.num_embeddings) - 1
                idx_seq = torch.clamp(arr[..., j, :], 0, max_idx)
                pooled = emb_tbl(idx_seq).mean(dim=-2)
                user_feat_list.append(pooled)
        else:
            for _ in self.USER_ARRAY_FEAT:
                user_feat_list.append(torch.zeros_like(item_emb0))

        # user continual (packed)
        if len(self.USER_CONTINUAL_FEAT) > 0:
            if 'user_cont' in features and features['user_cont'] is not None:
                uc = features['user_cont'].to(device, non_blocking=True)  # [B, T, C_u]
                if self.ctr_scale_factor != 1.0:
                    uc = uc * self.ctr_scale_factor
                if self.user_cont_mlp is not None:
                    user_feat_list.append(self.user_cont_mlp(uc))
                else:
                    user_feat_list.append(uc)
            else:
                user_feat_list.append(torch.zeros_like(item_emb0))

        # Dynamic feature importance via SENet (user-side, packed)
        if getattr(self, 'use_senet', False):
            try:
                if hasattr(self, 'user_gate') and isinstance(self.user_gate, SENetGate):
                    expected_u = int(self.user_gate.num_blocks)
                    actual_u = int(len(user_feat_list))
                    if expected_u == actual_u:
                        user_feat_list = self.user_gate(user_feat_list)
            except Exception:
                pass
        all_user_emb = torch.cat(user_feat_list, dim=2) if len(user_feat_list) > 1 else user_feat_list[0]
        all_user_emb = F.gelu(self.userdnn(all_user_emb))

        # time side (sequence only)
        time_feat_list = []
        if 'time_sparse' in features and features['time_sparse'] is not None and len(self.TIME_SPARSE_FEAT) > 0:
            x = features['time_sparse'].to(device, non_blocking=True)  # [B, T, S_t]
            for j, k in enumerate(self.TIME_SPARSE_FEAT):
                emb_tbl = self.sparse_emb[k]
                max_idx = int(emb_tbl.num_embeddings) - 1
                idx = torch.clamp(x[..., j], 0, max_idx)
                time_feat_list.append(emb_tbl(idx))
        if 'time_continual' in features and features['time_continual'] is not None and len(self.TIME_CONTINUAL_FEAT) > 0:
            xc = features['time_continual'].to(device, non_blocking=True)  # [B, T, C]
            for j, _ in enumerate(self.TIME_CONTINUAL_FEAT):
                time_feat_list.append(xc[..., j].unsqueeze(-1))

        if len(time_feat_list) > 0:
            all_time = torch.cat(time_feat_list, dim=2)
            time_emb = F.gelu(self.timednn(all_time))
            return all_item_emb + all_user_emb + time_emb
        else:
            return all_item_emb + all_user_emb

    def score_rank_from_packed(self, batch, seq_embs):
        """
        Compute rank scores for candidate packs.
        batch may contain:
          - rank_ids: [B, T, L]
          - rank_item_sparse: [B, T, L, S_i] (optional)
        Returns: (scores [B, T, L], mask [B, T, L]) where invalid ids are masked False
        """
        if 'rank_ids' not in batch:
            return None
        device = self.dev
        rank_ids = batch['rank_ids'].to(device)
        B, T, L = rank_ids.shape

        features = {}
        if 'rank_item_sparse' in batch and batch['rank_item_sparse'] is not None:
            x = batch['rank_item_sparse'].to(device)  # [B, T, L, S_i]
            features['item_sparse'] = x.view(B * T, L, x.size(-1))
        else:
            features['item_sparse'] = None

        # reshape to [B*T, L]
        seq_ids = rank_ids.view(B * T, L)
        item_embs = self.feat2emb_packed(seq_ids, features, include_user=False)  # [B*T, L, H]
        item_embs = item_embs.view(B, T, L, -1)
        item_embs = F.normalize(item_embs, dim=-1)
        seq_norm = F.normalize(seq_embs, dim=-1)

        scores = torch.einsum('bth, btlh -> btl', seq_norm, item_embs)
        mask = (rank_ids > 0)
        scores = torch.where(mask, scores, torch.full_like(scores, float('-inf')))
        return scores, mask

    def _encode_sequence_from_packed(self, seq_ids, token_type, batch_features, pos_ids=None, time_ids=None):
        """Helper to produce contextualized sequence embeddings from packed inputs."""
        device = self.dev
        # Strict parity with non-packed path: do not trim or shift sequence/window
        features = {
            'item_sparse': batch_features.get('item_sparse', None),
            'user_sparse': batch_features.get('user_sparse', None),
            'time_sparse': batch_features.get('time_sparse', None),
            'time_continual': batch_features.get('time_continual', None),
            'mm': batch_features.get('mm', {}) if isinstance(batch_features.get('mm', {}), dict) else {},
            'item_cont': batch_features.get('item_cont', None),
            'user_cont': batch_features.get('user_cont', None),
        }
        seqs = self.feat2emb_packed(seq_ids, features, token_type=token_type, include_user=True)
        seqs *= self.item_emb.embedding_dim ** 0.5
        B, T = seqs.shape[0], seqs.shape[1]
        poss = torch.arange(1, T + 1, device=device).unsqueeze(0).expand(B, -1).clone()
        poss *= (seq_ids != 0).to(device)
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # build masks
        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=device)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (token_type != 0).to(device)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        # encoder
        if self.encoder_type == 'hstu':
            # 确保pos_ids和time_ids在正确的设备上
            pos_ids_device = pos_ids.to(device) if pos_ids is not None else None
            time_ids_device = time_ids.to(device) if time_ids is not None else None
            if self.grad_checkpointing and self.training:
                def _wrap(b):
                    def fn(x):
                        return b(x, attn_mask=attention_mask, pos_ids=pos_ids_device, time_ids=time_ids_device)
                    return fn
                for block in self.blocks:
                    seqs = torch.utils.checkpoint.checkpoint(_wrap(block), seqs, use_reentrant=False)
            else:
                for block in self.blocks:
                    seqs = block(seqs, attn_mask=attention_mask, pos_ids=pos_ids_device, time_ids=time_ids_device)
        else:
            for i in range(len(self.attention_layers)):
                if self.norm_first:
                    x = self.attention_layernorms[i](seqs)
                    mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                    seqs = seqs + mha_outputs
                    seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
                else:
                    mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                    seqs = self.attention_layernorms[i](seqs + mha_outputs)
                    seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward_packed(self, batch):
        """
        batch: output of collate_fn_packed
        Returns: seq_embs, pos_embs, neg_embs, loss_mask
        """
        # Anchor embeddings
        if getattr(self, 'train_item_only', False):
            # 不使用序列/用户/时间特征，anchor 直接由 item_feat_dict 衍生的 item 特征构成
            anchors = {
                'item_sparse': batch.get('item_sparse', None),
                'mm': batch.get('mm', {}),
                'item_cont': batch.get('item_cont', None),
            }
            log_feats = self.feat2emb_packed(batch['seq_ids'], anchors, include_user=False)
        else:
            # 提取时间和位置信息
            pos_ids = batch.get('rel_pos_ids', None)
            time_ids = batch.get('rel_time_ids', None)
            # 确保pos_ids和time_ids在正确的设备上
            if pos_ids is not None:
                pos_ids = pos_ids.to(self.dev)
            if time_ids is not None:
                time_ids = time_ids.to(self.dev)
            # contextualized sequence embeddings
            log_feats = self._encode_sequence_from_packed(batch['seq_ids'], batch['token_type'], batch, pos_ids=pos_ids, time_ids=time_ids)

        # positive/negative embeddings (no user fusion, direct item features only)
        pos_features = {
            'item_sparse': batch.get('pos_item_sparse', None),
            'mm': batch.get('mm_pos', {}),
            'item_cont': batch.get('pos_item_cont', None),
        }
        neg_features = {
            'item_sparse': batch.get('neg_item_sparse', None),
            'mm': batch.get('mm_neg', {}),
            'item_cont': batch.get('neg_item_cont', None),
        }
        pos_embs = self.feat2emb_packed(batch['pos_ids'], pos_features, include_user=False)
        neg_embs = self.feat2emb_packed(batch['neg_ids'], neg_features, include_user=False)

        loss_mask = (batch['next_token_type'] == 1).to(self.dev)
        return log_feats, pos_embs, neg_embs, loss_mask

    @torch.no_grad()
    def encode_items(self, batch_items):
        """Encode items from ItemDatasetPacked batch -> [B, H] normalized."""
        device = self.dev
        item_ids = batch_items['item_id'].to(device)
        B = item_ids.shape[0]
        seq_ids = item_ids.view(B, 1)
        features = {
            'item_sparse': batch_items.get('item_sparse', None),
            'mm': batch_items.get('mm', {}),
        }
        if features['item_sparse'] is not None:
            features['item_sparse'] = features['item_sparse'].to(device).view(B, 1, -1)
        if features['mm']:
            features['mm'] = {k: v.to(device).view(B, 1, -1) for k, v in features['mm'].items()}
        embs = self.feat2emb_packed(seq_ids, features, include_user=False)[:, 0, :]
        embs = F.normalize(embs, dim=-1)
        return embs

    def predict(self, log_seqs, seq_feature, mask, pos_ids=None, time_ids=None):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
            pos_ids: 位置ID，用于相对位置偏置 [batch_size, maxlen]
            time_ids: 时间ID，用于相对时间偏置 [batch_size, maxlen]
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature, pos_ids=pos_ids, time_ids=time_ids)

        final_feat = log_feats[:, -1, :]
        final_feat = F.normalize(final_feat, p=2, dim=-1)

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            batch_emb = F.normalize(batch_emb, p=2, dim=-1)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        _save_emb_local(final_embs, Path(save_path, 'embedding.fbin'))
        _save_emb_local(final_ids, Path(save_path, 'id.u64bin'))