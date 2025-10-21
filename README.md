# HSTU: Multimodal Generative Recommendation

中文 | English

---

## 1. Overview 概览

This repository implements a multimodal generative recommendation system optimized for large-scale user/item catalogs. It replaces the traditional Transformer encoder (MHA+FFN) with a stable HSTU-style encoder, adds time-aware and CTR features, and accelerates both training and inference with packed dataloading.

本仓库实现了一个面向大规模用户/物品的多模态生成式推荐框架。核心用 HSTU 风格编码器替代传统 Transformer（MHA+FFN），并引入时间与 CTR 特征；通过“特征打包”的数据管线显著加速训练与推理。

Key ideas 关键点：
- HSTU blocks replace MHA+FFN with Pre-Norm and RMSNorm for stability; optional RoPE and relative position/time bias.
- 架构上以 HSTU blocks 统一替代 MHA+FFN，并采用 Pre-Norm 与 RMSNorm；支持 RoPE 与相对位置/时间偏置。
- Dynamic “Rank package” sampling: for each step build candidates {cold/hot click, cold/hot exposure, random} to fight head bias; optional hard-negative cache.
- 采样上构建“Rank 包”：每步从历史前缀动态采样【冷/热点击、冷/热曝光、随机】，压制头部热门，支持长尾；可维护困难负样本缓存。
- Time features: hour/weekday/week + log-bucketed time-gap continuous channel; injected during training and inference.
- 时间特征：小时/星期/周 + 时间差对数分桶连续通道；训练/推理两侧一致注入。
- Global CTR priors for user/item: precompute offline, standardize, and inject as continual features to ease cold-start.
- 全局 CTR（用户/物品）：离线预估并标准化后作为连续通道注入，有效缓解冷启动。
- Packed dataloader: move heavy list/dict→tensor conversion into workers; multi-process loading and prefetching.
- 数据加载“打包化”：将 list/dict→tensor 的繁重工作前移到 worker，配合多进程/预取显著提速。
- ID reliance mitigation: stronger dropout on ID embeddings on early epochs, then anneal to 0.
- 降低 ID 依赖：首轮给 ID/Item Embedding 较强 dropout，随后随 epoch 退火到 0。

Empirical note 实验提示（示例来源于内部实践，非保证值）：
- HSTU+PreNorm+RMSNorm (≈16 layers) improved blended score by ≈+0.008.
- Rank 包采样带来 ≈+0.03；时间通道+CTR ≈+0.02；冷启动/退火策略 ≈+0.005。


## 2. Repository structure 目录结构
- `main.py`: training loop with InfoNCE (+optional rank loss), warmup+cosine LR, gradient accumulation/checkpointing.
- `infer.py`: embedding export for users/items, ANN retrieval (FAISS or torch), candidate id mapping.
- `model.py`: HSTU encoder, SENet gates for dynamic feature importance, feature fusion (user/item/time/mm/CTR).
- `dataset.py`: dataset definitions, CTR loading/transform, packed dataloaders, rank-package construction.
- `dataset_packed.py`: infer-time packed dataset (co-located with `infer.py`).
- `precompute_ctr.py`: offline CTR estimation and standardization metadata.
- `retrieval_torch.py`: streaming cosine top-k in PyTorch (FAISS fallback available).
- `run.sh`: example pipeline (precompute CTR → train with recommended flags).


## 3. Data format 数据格式
Train 训练数据（目录记为 `$TRAIN_DATA_PATH`）：
- `seq.jsonl` + `seq_offsets.pkl`: each line is a user sequence (list). A record is a tuple:
  `[user_id|None, item_id|None, user_feat|None, item_feat|None, action_type:int, timestamp:int?]`。
- `item_feat_dict.json`: item feature dict keyed by original item id (creative_id)。
- `creative_emb/`: multimodal embeddings，文件命名如 `emb_81_32.pkl` 或 `emb_{fid}_{dim}/part-*`。
- `indexer.pkl`: dict with `i` (item re-id map), `u` (user re-id), `f` (feature id → value id map)。
- Optional 可选：`item_pop.json`（记录 head/tail 分桶，用于 Rank 包）。
- CTR artifacts CTR 产物（默认在 `$USER_CACHE_PATH/ctr` 或 `$TRAIN_DATA_PATH/ctr`）：
  - `user_ctr.json`, `item_ctr.json`, `ctr_meta.json`（含标准化/先验参数）。

Inference 推理数据（目录记为 `$EVAL_DATA_PATH`）：
- `predict_seq.jsonl` + `predict_seq_offsets.pkl`: user sequences for query generation；记录格式兼容训练。
- `predict_set.jsonl`: candidate items，每行包含 `{creative_id, retrieval_id, features}`。

Outputs 推理输出（目录 `$EVAL_RESULT_PATH`）：
- `query.fbin`: user/query embeddings; `embedding.fbin` + `id.u64bin`: item embeddings/ids。
- `retrive_id2creative_id.json`: retrieval id → creative_id 映射；检索 TopK 在内存返回。


## 4. Dependencies 依赖
- Python ≥ 3.9, PyTorch ≥ 2.0 (Flash attn API auto-used when available)
- numpy, tqdm, tensorboard, faiss-gpu (optional; fallback to torch retrieval)

Install (conda/pip 任选)：
```bash
pip install -r requirements.txt  # 如无，可手动安装：torch numpy tqdm tensorboard faiss-gpu
```


## 5. Quick start 快速开始
### 5.1 Precompute CTR 预计算 CTR
```bash
export TRAIN_DATA_PATH=/path/to/train_data
export USER_CACHE_PATH=/path/to/cache   # 可与 TRAIN_DATA_PATH 相同
python -u precompute_ctr.py \
  --seq_path ${TRAIN_DATA_PATH}/seq.jsonl \
  --output_dir ${USER_CACHE_PATH}/ctr \
  --prior auto --k_clip 10,10000
```
或直接使用脚本：
```bash
bash run.sh  # 内含：若无CTR则先预计算，再启动训练
```

### 5.2 Train 训练
设置环境变量（必需）：
```bash
export TRAIN_DATA_PATH=/path/to/train_data
export TRAIN_LOG_PATH=./logs
export TRAIN_TF_EVENTS_PATH=./tf_events
export TRAIN_CKPT_PATH=./ckpt
export USER_CACHE_PATH=/path/to/cache
```
示例命令（与 `run.sh` 一致的核心配置）：
```bash
python -u main.py \
  --use_ctr --use_userctr --use_itemctr \
  --ctr_dir ${USER_CACHE_PATH}/ctr \
  --use_timefeat --use_rel_bias=1 --norm_first --norm_type=rmsnorm \
  --encoder_type=hstu --dqk=32 --dv=64 \
  --num_blocks=10 --num_heads=8 --dropout_rate=0.2 \
  --batch_size=64 --lr=0.002 --warmup_steps=2666 \
  --gradient_accumulation_steps=4 --clip_norm=1.0 --temperature=0.03 \
  --num_epochs=8 --pin_memory
```
要点：
- InfoNCE 对比学习，支持动作感知加权/双温度/焦点损失：`--nce_weight_scheme {none,fixed,cb,focal}`、`--two_tau --tau_click --tau_expo`、`--focal_gamma`。
- 可选排序损失：`--use_rankloss=1 --rankloss_mode={pairwise,listwise} --lambda_rank 0.6`。
- 首个 epoch 对 ID/Item Embedding 施加强正则：`--first_epoch_id_item_dropout`（自动退火到0）。
- 训练使用 packed dataloader：多进程 + 预取，显著降低CPU瓶颈。

### 5.3 Inference 推理与检索
设置环境变量：
```bash
export EVAL_DATA_PATH=/path/to/eval_data
export EVAL_RESULT_PATH=./eval_out
# 指向包含 "model.pt" 的目录（训练保存于 ${TRAIN_CKPT_PATH}/global_step*/model.pt）
export MODEL_OUTPUT_PATH=/path/to/checkpoint_dir
export USER_CACHE_PATH=/path/to/cache  # 若要注入CTR
```
运行（函数式入口）：
```bash
python - <<'PY'
import os
from infer import infer
os.makedirs(os.environ['EVAL_RESULT_PATH'], exist_ok=True)
# 可调：FAISS/torch、top_k、use_timefeat/use_rel_bias等，见 infer.get_args()
res = infer()
print('Done, binaries saved under EVAL_RESULT_PATH')
PY
```
或使用 FAISS：命令行参数 `--retrieval_backend faiss`（未安装将自动回退为 torch）。


## 6. Model & features 模型与特征
- Encoder 编码器：`--encoder_type hstu`（默认）。HSTU block 仅保留稳定的 softmax 注意力，Pre-Norm + RMSNorm；可选 RoPE（`--use_rope`）。
- Relative bias 相对偏置：`--use_rel_bias`，位置桶 `--rel_pos_buckets`，时间桶 `--rel_time_buckets`。
- Time features 时间通道：小时/星期/周 + log1p 时间差；训练/推理对齐，packed 路径中自动构造。
- Multimodal 多模态：支持特征ID {81..86} 的 mm 向量，经线性变换后与 ID/sparse/array/continual 融合。
- CTR continual：用户(150)/物品(151) 两侧可独立开关（训练 `--use_ctr/--use_userctr/--use_itemctr`，推理同名开关）。
- SENet gate：`--use_senet` 动态通道权重，分别在 user/item 侧对特征块做门控。


## 7. Dataset internals 细节提示
- Packed dataloader 会输出全张量字典（参见 `dataset.py::collate_fn_packed`），训练/推理的模型前向均走 packed 路径。
- Rank 包构造：基于时间前缀统计“点击/曝光×冷热”，生成 5 槽候选 `[cold_click, hot_click, cold_expo, hot_expo, random]`，可与排序损失联动。
- 困难负样本缓存：`USER_CACHE_PATH/hard_negative_cache.pkl` 自动构建/持久化（仅训练）。


## 8. Environment variables 环境变量一览
- Train 训练：`TRAIN_DATA_PATH`, `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH`, `TRAIN_CKPT_PATH`, `USER_CACHE_PATH`。
- Infer 推理：`EVAL_DATA_PATH`, `EVAL_RESULT_PATH`, `MODEL_OUTPUT_PATH`, `USER_CACHE_PATH`。
- Optional 可选：`FORCE_USE_TIMEFEAT=0/1` 强制（覆写 CLI）。


## 9. Tips & performance 实用建议
- Prefer GPU with bf16/TF32 enabled for best throughput; PyTorch 2.x recommended.
- 建议使用 GPU 并开启 bf16/TF32；推荐 PyTorch 2.x。
- If FAISS is unavailable, the torch streaming top-k remains robust for large `Ni` via chunking。
- 若未安装 FAISS，可使用内置的 PyTorch 分块流式检索，稳定内存占用。


## 10. License & acknowledgement 许可与致谢
Code is provided for research/engineering reference. Portions of design are inspired by recent sequence modeling and multimodal retrieval literature.

代码供研究与工程参考使用。部分设计思路参考了近期序列建模与多模态检索相关工作。

---

If this helps your work, please consider starring the repo. 欢迎 Star 支持！
