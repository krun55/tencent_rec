import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import logging
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import save_emb
from model import BaselineModel
from retrieval_torch import topk_cosine_torch_streaming
from seed_utils import fix_random_seeds, make_worker_init_fn, make_generator


logger = logging.getLogger("infer")
if not logger.handlers:
    # ensure logs go to stdout so platform log collectors capture them
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Prefer local dataset_packed (ships with infer) to ensure acceleration path
try:
    from dataset_packed import PredictSeqDatasetPacked, collate_fn_packed  # type: ignore
except Exception:
    # Fallback to dataset if it already contains packed classes
    try:
        from dataset import PredictSeqDatasetPacked, collate_fn_packed  # type: ignore
    except Exception as e:
        try:
            import dataset as _ds  # type: ignore
            ds_path = getattr(_ds, "__file__", "<unknown>")
            available = [n for n in dir(_ds) if n.endswith("Packed") or n.endswith("collate_fn")]
            raise ImportError(
                f"PredictSeqDatasetPacked not found in dataset module ({ds_path}).\n"
                f"Available names: {available}.\n"
                f"Please include dataset_packed.py with PredictSeqDatasetPacked/collate_fn_packed in the infer package, or deploy the updated dataset.py."
            ) from e
        except Exception:
            raise


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=256, type=int)
    parser.add_argument('--num_blocks', default=16, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.01, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true', default=True)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true', default=True)
    parser.add_argument('--temperature', default=0.04, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--clip_norm', default=1.0, type=float)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--prefetch_factor', default=4, type=int)
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--tf32', action='store_true')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--deterministic', action='store_true')

    # Retrieval options
    parser.add_argument('--retrieval_backend', default='torch', type=str, choices=['faiss', 'torch'], help='ANN backend')
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--overfetch_factor', default=2, type=int, help='Fetch more than top_k to backfill unmapped')

    # HSTU 架构参数
    parser.add_argument('--encoder_type', default='hstu', type=str, choices=['transformer', 'hstu'])
    parser.add_argument('--dqk', default=32, type=int, help='每头 QK 维度')
    parser.add_argument('--dv', default=64, type=int, help='每头 V/U 维度')
    parser.add_argument('--use_rel_bias', default=1, type=int, choices=[0, 1], help='是否使用相对位置/时间偏置')
    parser.add_argument('--rel_pos_buckets', default=32, type=int, help='相对位置分桶数')
    parser.add_argument('--rel_time_buckets', default=32, type=int, help='相对时间分桶数（0表示不使用）')
    parser.add_argument('--hstu_activation', default='silu', type=str, choices=['silu'], help='HSTU 激活函数')
    parser.add_argument('--norm_type', default='rmsnorm', type=str, choices=['layernorm', 'rmsnorm'], help='归一化类型')
    
    # RoPE 相关参数
    parser.add_argument('--use_rope', default=1, type=int, choices=[0, 1], help='是否在注意力中使用RoPE')
    parser.add_argument('--rope_theta', default=10000.0, type=float, help='RoPE 基数 theta')
    parser.add_argument('--rope_dim', default=0, type=int, help='RoPE 作用维度，0表示等于dqk')

    # Dynamic feature importance (SENet)
    parser.add_argument('--use_senet', action='store_true', default=True, help='启用SENet动态特征重要性')
    parser.add_argument('--senet_reduction', default=4, type=int, help='SENet通道缩减比')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # Time feature switches and params (must be defined BEFORE parse_args)
    parser.add_argument('--use_timefeat', action='store_true',
                        help='Enable time features (hour/weekday/week/time-diff)')
    parser.add_argument('--time_diff_log_base', type=float, default=2.0,
                        help='Log base for diff bucketing (>1); used if bucketization enabled')
    parser.add_argument('--time_diff_max_bucket', type=int, default=63,
                        help='Max bucket id for time diff (0..max_bucket)')

    # 与训练保持一致的 user token 位置策略
    parser.add_argument('--user_token_pos', default='keep', choices=['first', 'keep'],
                        help="推理时 user token 位置策略（同训练）")
    
    parser.add_argument('--q_chunk', default=4096, type=int)
    parser.add_argument('--i_chunk', default=327680, type=int)
    parser.add_argument('--use_bf16', default=1, type=int, choices=[0, 1])        # 原 store_true
    parser.add_argument('--log_interval_sec', default=5, type=int, help='Seconds between ETA log lines')

    # CTR features (inference)
    parser.add_argument('--use_ctr', action='store_true', default=True,help='Enable user/item CTR features at inference')
    parser.add_argument('--use_userctr', action='store_true', default=True, help='Enable user CTR at inference')
    parser.add_argument('--use_itemctr', action='store_true', default=True, help='Enable item CTR at inference')
    parser.add_argument('--ctr_dir', type=str, default=None, help='Directory for CTR artifacts; defaults to $USER_CACHE_PATH/ctr or $EVAL_DATA_PATH/ctr if not set')


    args = parser.parse_args()

    # 推理默认启用 SENet（除非明确关闭）
    if not hasattr(args, 'use_senet') or args.use_senet is None:
        args.use_senet = True

    # 尊重 CLI，并允许通过环境变量覆盖：FORCE_USE_TIMEFEAT=0/1/true/false
    env_tf = os.environ.get('FORCE_USE_TIMEFEAT', None)
    if env_tf is not None:
        try:
            args.use_timefeat = str(env_tf).strip().lower() not in ('0', 'false', 'no')
        except Exception:
            args.use_timefeat = bool(env_tf)

    # 仅在启用时间特征时，确保 time diff 参数有值
    if getattr(args, 'use_timefeat', False):
        if not hasattr(args, 'time_diff_log_base') or args.time_diff_log_base is None:
            args.time_diff_log_base = 2.0
        if not hasattr(args, 'time_diff_max_bucket') or args.time_diff_max_bucket is None:
            args.time_diff_max_bucket = 63

    # 不强制开启相对偏置，遵从 CLI；校正 rel_time_buckets 为非负
    try:
        args.rel_time_buckets = max(0, int(args.rel_time_buckets))
    except Exception:
        args.rel_time_buckets = 32

    # 日志确认
    print(f"[INFO] user_token_pos = {args.user_token_pos}")
    print(f"[INFO] use_timefeat = {args.use_timefeat}")
    print(f"[INFO] use_rel_bias = {args.use_rel_bias}")
    print(f"[INFO] rel_time_buckets = {args.rel_time_buckets}")
    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def load_fbin(fbin_path: Path) -> np.ndarray:
    """Load vectors saved by save_emb (uint32 header: N, D; then float32 data)."""
    with open(fbin_path, 'rb') as f:
        num_points = struct.unpack('I', f.read(4))[0]
        num_dims = struct.unpack('I', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32, count=num_points * num_dims)
    return data.reshape(num_points, num_dims)


def load_u64bin(id_path: Path) -> np.ndarray:
    """Load ids saved by save_emb (uint32 header: N, 1; then uint64 data)."""
    with open(id_path, 'rb') as f:
        num_points = struct.unpack('I', f.read(4))[0]
        _ = struct.unpack('I', f.read(4))[0]
        data = np.fromfile(f, dtype=np.uint64, count=num_points)
    return data


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model, args=None):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}
    oov_count = 0
    total_items = 0

    # Optional: load CTR artifacts for items to enrich continual feature '151'
    I_CTR_ID = '151'
    # Hygiene: respect --ctr_dir first, then USER_CACHE_PATH/ctr, then EVAL_DATA_PATH/ctr
    ctr_dir = None
    if args is not None and getattr(args, 'ctr_dir', None) and os.path.isdir(args.ctr_dir):
        ctr_dir = args.ctr_dir
    if ctr_dir is None:
        env_base = os.environ.get('USER_CACHE_PATH', None)
        if env_base is not None:
            maybe = os.path.join(env_base, 'ctr')
            if os.path.isdir(maybe):
                ctr_dir = maybe
    if ctr_dir is None:
        maybe_eval_ctr = Path(os.environ.get('EVAL_DATA_PATH', str(Path(candidate_path).parent)), 'ctr')
        ctr_dir = str(maybe_eval_ctr) if maybe_eval_ctr.is_dir() else None
    # Respect ablation: only enable candidate-side item CTR if explicitly enabled at inference
    enable_item_ctr = bool(getattr(args, 'use_itemctr', False)) if args is not None else False
    item_ctr = None
    ctr_meta = None
    if enable_item_ctr and ctr_dir is not None:
        try:
            with open(Path(ctr_dir, 'item_ctr.json'), 'r', encoding='utf-8') as f:
                item_ctr = {int(k): float(v) for k, v in json.load(f).items()}
            with open(Path(ctr_dir, 'ctr_meta.json'), 'r', encoding='utf-8') as f:
                ctr_meta = json.load(f)
            eps = float(ctr_meta['transform']['eps'])
            logit_flag = bool(ctr_meta['transform']['logit'])
            i_mean = float(ctr_meta['item']['mean'])
            i_std = float(ctr_meta['item']['std'])
            a_i = float(ctr_meta['item']['alpha'])
            b_i = float(ctr_meta['item']['beta'])
            i_prior = a_i / max(a_i + b_i, 1e-12)

            def ctr_transform(x):
                x = float(min(max(x, eps), 1.0 - eps))
                if logit_flag:
                    x = math.log(x / (1.0 - x))
                denom = i_std if i_std and i_std > 0 else 1.0
                return float((x - i_mean) / denom)

            def get_item_ctr_val(item_reid: int) -> float:
                if item_ctr is None:
                    return 0.0
                if int(item_reid) in item_ctr:
                    return ctr_transform(item_ctr[int(item_reid)])
                # 未读到CTR时，直接返回0（不做变换）
                return 0.0
            # Log candidate-side CTR stats and key coverage
            try:
                logger.info(
                    f"[CTR-cand] #keys={len(item_ctr)} i_prior={i_prior:.6f} "
                    f"mean/std={i_mean:.6f}/{i_std:.6f} logit={logit_flag} eps={eps} dir={ctr_dir}"
                )
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"CTR artifacts not loaded for candidates: {e}")
            item_ctr, ctr_meta = None, None

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            if item_id == 0:
                oov_count += 1
            total_items += 1
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            # inject item CTR continual if available and enabled
            if enable_item_ctr and I_CTR_ID in feat_types['item_continual'] and item_ctr is not None and ctr_meta is not None:
                try:
                    # map creative_id -> item reid
                    item_reid = indexer.get(creative_id, 0)
                    if item_reid > 0:
                        feature[I_CTR_ID] = get_item_ctr_val(item_reid)
                    else:
                        feature[I_CTR_ID] = 0
                except Exception:
                    feature[I_CTR_ID] = 0

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    try:
        ratio = (oov_count / max(1, total_items))
        logger.info(f"candidate_index_oov: {oov_count}/{total_items} ({ratio:.2%})")
    except Exception:
        pass
    return retrieve_id2creative_id


def infer():
    args = get_args()
    # set seeds very early
    fix_random_seeds(args.seed, deterministic=args.deterministic)
    data_path = os.environ.get('EVAL_DATA_PATH')

    # Use packed dataset for inference to reuse training-time acceleration
    test_dataset = PredictSeqDatasetPacked(data_path, args)
    # 记录时间特征配置与维度
    try:
        ts = test_dataset.feature_types.get('time_sparse', [])
        tc = test_dataset.feature_types.get('time_continual', [])
        logger.info(
            f"[TIME] use_timefeat={getattr(test_dataset,'use_timefeat', None)} "
            f"S_t={len(ts)} C_t={len(tc)} "
            f"log_base={getattr(test_dataset,'time_diff_log_base', None)} "
            f"max_bucket={getattr(test_dataset,'time_diff_max_bucket', None)}"
        )
    except Exception:
        pass
    # Log indexer and sequence-side CTR transform params for parity check
    try:
        logger.info(
            f"[Indexer] itemnum={test_dataset.itemnum} usernum={test_dataset.usernum}"
        )
    except Exception:
        pass
    try:
        logger.info(
            f"[CTR-seq] mean/std user={getattr(test_dataset,'u_mean',None)}/{getattr(test_dataset,'u_std',None)} "
            f"item={getattr(test_dataset,'i_mean',None)}/{getattr(test_dataset,'i_std',None)} "
            f"logit={getattr(test_dataset,'logit',None)} eps={getattr(test_dataset,'eps',None)}"
        )
    except Exception:
        pass

    # Precompute ordered user_id list aligned with uid [0..N-1]
    uid2user = {}
    def uid_to_user_id(uid: int) -> str:
        if uid in uid2user:
            return uid2user[uid]
        try:
            user_sequence = test_dataset._load_user_data(uid)
            user_id = None
            for record in user_sequence:
                u = record[0] if len(record) >= 1 else None
                if u:
                    if isinstance(u, str):
                        user_id = u
                    else:
                        user_id = test_dataset.indexer_u_rev.get(u, str(u))
                    break
            if user_id is None:
                user_id = str(uid)
            uid2user[uid] = user_id
            return user_id
        except Exception:
            uid2user[uid] = str(uid)
            return uid2user[uid]

    num_users = len(test_dataset)
    ordered_user_ids = [uid_to_user_id(i) for i in range(num_users)]

    worker_init_fn = make_worker_init_fn(args.seed)
    dl_generator = make_generator(args.seed, rank=0)

    # Build DataLoader with safe kwargs for num_workers=0
    loader_kwargs = {
        'dataset': test_dataset,
        'batch_size': args.batch_size,
        'shuffle': False,
        'pin_memory': args.pin_memory,
        'generator': dl_generator,
        'collate_fn': collate_fn_packed,
    }

    if args.num_workers and args.num_workers > 0:
        loader_kwargs['num_workers'] = args.num_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
        loader_kwargs['persistent_workers'] = args.persistent_workers
        loader_kwargs['worker_init_fn'] = worker_init_fn
    else:
        loader_kwargs['num_workers'] = 0

    test_loader = DataLoader(**loader_kwargs)
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()
    # 记录模型侧时间特征通道数
    try:
        logger.info(
            f"[TIME-model] S_t={len(getattr(model,'TIME_SPARSE_FEAT', {}))} "
            f"C_t={len(getattr(model,'TIME_CONTINUAL_FEAT', []))}"
        )
    except Exception:
        pass

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    all_embs = []
    norm_sum = 0.0
    norm_sq_sum = 0.0
    zero_count = 0
    total_count = 0

    # Progress bar with real-time ETA
    total_batches = len(test_loader)
    total_users = len(test_dataset)
    start_time = time.time()
    last_log_time = start_time
    processed_users = 0
    pbar = tqdm(total=total_batches, dynamic_ncols=True, desc="Infer")
    # immediate summary line to stdout
    print(
        f"Infer start: users={total_users}, batches={total_batches}, batch_size={args.batch_size}, workers={args.num_workers}",
        flush=True,
    )
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            pos_ids = batch.get('rel_pos_ids', None)
            time_ids = batch.get('rel_time_ids', None)
            # 首个batch打印时间特征形状与相对时间偏置状态
            if step == 0:
                try:
                    ts = batch.get('time_sparse', None)
                    tc = batch.get('time_continual', None)
                    ts_shape = None if ts is None else list(ts.shape)
                    tc_shape = None if tc is None else list(tc.shape)
                    rel_time_on = bool(getattr(args, 'use_rel_bias', 0)) and int(getattr(args, 'rel_time_buckets', 0)) > 0
                    logger.info(
                        f"[TIME-batch0] use_timefeat={args.use_timefeat} time_sparse_shape={ts_shape} time_cont_shape={tc_shape} "
                        f"rel_time_bias_enabled={rel_time_on}"
                    )
                except Exception:
                    pass
            seq_embs = model._encode_sequence_from_packed(
                batch['seq_ids'], batch['token_type'], batch, pos_ids=pos_ids, time_ids=time_ids
            )
            user_embs = F.normalize(seq_embs[:, -1, :], p=2, dim=-1)

            # Strict shape checks on first batch to fail fast
            if step == 0:
                # MM dims alignment
                mm_batch = batch.get('mm', {}) or {}
                for fid, exp_dim in getattr(model, 'ITEM_EMB_FEAT', {}).items():
                    if fid in mm_batch:
                        got = int(mm_batch[fid].shape[-1])
                        if got != int(exp_dim):
                            raise RuntimeError(
                                f"MM feature dim mismatch for fid={fid}: got {got}, expected {exp_dim}"
                            )
                # Ensure continual CTR tensors are present when model expects them
                if len(getattr(model, 'ITEM_CONTINUAL_FEAT', [])) > 0:
                    assert ('item_cont' in batch) and (batch['item_cont'] is not None), "missing item_cont in inference batch"
                if len(getattr(model, 'USER_CONTINUAL_FEAT', [])) > 0:
                    assert ('user_cont' in batch) and (batch['user_cont'] is not None), "missing user_cont in inference batch"
                # Verify DNN input dims match feature block counts
                H = int(model.item_emb.embedding_dim)
                expected_item_blocks = 1 + len(model.ITEM_SPARSE_FEAT) + len(model.ITEM_ARRAY_FEAT) + len(model.ITEM_EMB_FEAT) + (1 if len(model.ITEM_CONTINUAL_FEAT) > 0 else 0)
                expected_user_blocks = len(model.USER_SPARSE_FEAT) + 1 + len(model.USER_ARRAY_FEAT) + (1 if len(model.USER_CONTINUAL_FEAT) > 0 else 0)
                assert int(model.itemdnn.in_features) == int(expected_item_blocks * H), f"itemdnn.in_features={model.itemdnn.in_features}, expected={expected_item_blocks * H}"
                assert int(model.userdnn.in_features) == int(expected_user_blocks * H), f"userdnn.in_features={model.userdnn.in_features}, expected={expected_user_blocks * H}"

        batch_norms = torch.linalg.norm(user_embs, ord=2, dim=-1)
        bn = batch_norms.detach().cpu().numpy()
        norm_sum += float(bn.sum())
        norm_sq_sum += float((bn ** 2).sum())
        zero_count += int((bn < 1e-6).sum())
        total_count += int(bn.size)

        emb_np = user_embs.detach().cpu().numpy().astype(np.float32)
        all_embs.append(emb_np)

        # Update progress/ETA by users processed
        processed_users += user_embs.shape[0]
        elapsed = max(1e-6, time.time() - start_time)
        ips = processed_users / elapsed  # items per second
        remaining_users = max(0, total_users - processed_users)
        eta_sec = remaining_users / max(1e-6, ips)
        # format ETA as H:MM:SS
        h = int(eta_sec // 3600)
        m = int((eta_sec % 3600) // 60)
        s = int(eta_sec % 60)
        eta_str = f"{h:d}:{m:02d}:{s:02d}"
        pbar.set_postfix({"ETA": eta_str, "ips": f"{ips:.1f}"})
        pbar.update(1)

        # Periodic explicit log line (newline) to ensure real-time log visibility
        now = time.time()
        if (now - last_log_time) >= max(1, int(args.log_interval_sec)) or (step + 1) == total_batches or step == 0:
            # use stdout printing to ensure visibility in external log systems
            print(
                f"Infer progress: {processed_users}/{total_users} users | {ips:.2f} u/s | ETA {eta_str}",
                flush=True,
            )
            last_log_time = now

    pbar.close()

    # 生成候选库的embedding 以及 id文件（记录耗时以便对比）
    t_item_start = time.time()
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
        args,
    )
    print(f"Item embedding built in {time.time()-t_item_start:.2f}s", flush=True)
    all_embs = np.concatenate(all_embs, axis=0)

    # Sanity check: ensure alignment counts
    if all_embs.shape[0] != len(ordered_user_ids):
        raise RuntimeError(
            f"Query/user count mismatch: query_embs={all_embs.shape[0]} vs users={len(ordered_user_ids)}"
        )

    # Check embedding variance across users (detect degenerate constant vectors)
    try:
        row_var = float(all_embs.var(axis=1).mean())
        if not np.isfinite(row_var) or row_var < 1e-8:
            raise RuntimeError(
                f"Abnormal query embeddings variance (mean row var={row_var:.3e}). Possible degenerate outputs."
            )
    except Exception:
        pass

    if total_count > 0:
        mean_norm = norm_sum / total_count
        var_norm = max(norm_sq_sum / total_count - mean_norm * mean_norm, 0.0)
        std_norm = math.sqrt(var_norm)
        logger.info(
            f"query_emb_norms: mean={mean_norm:.4f}, std={std_norm:.4f}, zero_cnt={zero_count}/{total_count}"
        )
    # 保存query文件（可选）
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))

    # 内置 ANN：优先 FAISS（GPU 可用则上 GPU），否则退化到 PyTorch 版
    EVAL_DIR = Path(os.environ.get('EVAL_RESULT_PATH'))
    item_vecs = load_fbin(EVAL_DIR / 'embedding.fbin')   # [Ni, D]
    item_ids = load_u64bin(EVAL_DIR / 'id.u64bin')       # [Ni]

    # Consistency checks for candidate library
    if item_vecs.shape[0] != item_ids.shape[0]:
        raise RuntimeError(
            f"Item vectors/ids count mismatch: vecs={item_vecs.shape[0]} vs ids={item_ids.shape[0]}"
        )
    if item_vecs.shape[1] != all_embs.shape[1]:
        raise RuntimeError(
            f"Embedding dim mismatch: items D={item_vecs.shape[1]} vs queries D={all_embs.shape[1]}"
        )
    if len(retrieve_id2creative_id) != int(item_ids.shape[0]):
        raise RuntimeError(
            f"retrive_id2creative_id size mismatch: mapping={len(retrieve_id2creative_id)} vs ids={item_ids.shape[0]}"
        )
    query_vecs = all_embs                                # [Nq, D]

    top_k = int(args.top_k)
    overfetch = max(1, int(args.overfetch_factor))
    k_fetch = max(top_k, top_k * overfetch)
    if args.retrieval_backend == 'faiss':
        try:
            import faiss
            d = item_vecs.shape[1]
            cpu_index = faiss.IndexFlatIP(d)
            # GPU if available
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(cpu_index)
            else:
                index = cpu_index
            index.add(item_vecs.astype(np.float32))
            D, I = index.search(query_vecs.astype(np.float32), k_fetch)
            retrieved_ids = item_ids[I]
        except Exception as e:
            logger.warning(f"FAISS fallback to torch retrieval due to: {e}")
            q = torch.from_numpy(query_vecs)
            it = torch.from_numpy(item_vecs)
            idx, _ = topk_cosine_torch_streaming(
                q,
                it,
                k=k_fetch,
                q_chunk=int(args.q_chunk),
                i_chunk=int(args.i_chunk),
                device=args.device,
                use_bf16=bool(args.use_bf16),
            )
            retrieved_ids = item_ids[idx.numpy()]
    else:
        q = torch.from_numpy(query_vecs)
        it = torch.from_numpy(item_vecs)
        idx, _ = topk_cosine_torch_streaming(
            q,
            it,
            k=k_fetch,
            q_chunk=int(args.q_chunk),
            i_chunk=int(args.i_chunk),
            device=args.device,
            use_bf16=bool(args.use_bf16),
        )
        retrieved_ids = item_ids[idx.numpy()]

    # 映射到 creative_id，避免回退到0；优先用有映射的，必要时用原始retrieval_id补齐
    retrieved_ids = retrieved_ids.astype(np.int64)
    num_queries = int(retrieved_ids.shape[0])
    mapped_total = 0
    cid_zero_count = 0
    topks = []
    for row in retrieved_ids:
        mapped_row = []
        mapped_true_cnt = 0
        # 优先选择有有效映射的
        for rid in row:
            cid = retrieve_id2creative_id.get(int(rid))
            if cid is None:
                continue
            try:
                cid_int = int(cid)
            except Exception:
                continue
            if cid_int == 0:
                cid_zero_count += 1
                continue
            mapped_row.append(cid_int)
            mapped_true_cnt += 1
            if len(mapped_row) >= top_k:
                break
        # 若不足top_k，用原始retrieval_id补齐
        if len(mapped_row) < top_k:
            for rid in row:
                if len(mapped_row) >= top_k:
                    break
                mapped_row.append(int(rid))
        topks.append(mapped_row[:top_k])
        mapped_total += min(mapped_true_cnt, top_k)

    # 日志：映射命中率与cid==0出现次数
    denom = max(1, num_queries * top_k)
    map_hit = mapped_total / denom
    logger.info(f"map_hit_rate: {map_hit:.2%} ({mapped_total}/{denom}), cid_eq_0_cnt={cid_zero_count}")

    return topks, ordered_user_ids
