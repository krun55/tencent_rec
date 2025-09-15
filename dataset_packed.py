import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Reuse base dataset utilities
from dataset import MyDataset


_SHAPE_DICT_MM = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}


class UserSeqDatasetPacked(MyDataset):
    """
    用户序列数据集（Packed 版本）
    - 在 worker 进程中完成 list/dict -> tensor 的转换
    - 返回全张量；collate 仅做 stack
    - 兼容现有的数据与特征定义；如未指定 array 的固定长度，则不输出 array 特征键
    """

    def __init__(self, data_dir, args, array_max_lens=None):
        super().__init__(data_dir, args)
        self.array_max_lens = array_max_lens

        # 预先缓存键顺序，便于张量打包（需与 model 中的 Embedding Dict 顺序一致）
        self.user_sparse_keys = list(self.feature_types['user_sparse'])
        self.item_sparse_keys = list(self.feature_types['item_sparse'])
        self.user_array_keys = list(self.feature_types['user_array']) if self.array_max_lens else []
        self.item_array_keys = list(self.feature_types['item_array']) if self.array_max_lens else []
        self.mm_feat_keys = list(self.feature_types['item_emb'])

        self.user_cont_keys = list(self.feature_types['user_continual'])
        self.item_cont_keys = list(self.feature_types['item_continual'])

        self.time_sparse_keys = list(self.feature_types.get('time_sparse', []))
        self.time_cont_keys = list(self.feature_types.get('time_continual', []))

    def _build_rank_packs_for_user(self, ext_user_sequence):
        L = len(ext_user_sequence)
        if L <= 1:
            return []
        packs = [[0, 0, 0, 0, 0] for _ in range(L - 1)]

        prefix_clicks, prefix_expos = set(), set()

        def pick(cands, forbid):
            from random import choice
            pool = [x for x in cands if x not in forbid and str(x) in self.item_feat_dict]
            return choice(pool) if pool else 0

        for idx in range(L - 1):
            cur = ext_user_sequence[idx]
            nxt = ext_user_sequence[idx + 1]
            if len(cur) >= 5:
                cur_id, _, cur_type, _ = cur[0], cur[1], cur[2], cur[3]
            else:
                cur_id, _, cur_type, _ = cur
            if len(nxt) >= 5:
                next_i, _, next_type, _ = nxt[0], nxt[1], nxt[2], nxt[3]
            else:
                next_i, _, next_type, _ = nxt

            if cur_type == 1 and cur_id:
                act = cur[3]
                if act is not None:
                    try:
                        a = int(act)
                        if a in self.click_set:
                            prefix_clicks.add(cur_id)
                        elif a in self.expo_set:
                            prefix_expos.add(cur_id)
                    except Exception:
                        pass

            if next_type == 1:
                tail_clicks = [x for x in prefix_clicks if self.pop_bucket.get(x, 'head') == 'tail']
                head_clicks = [x for x in prefix_clicks if self.pop_bucket.get(x, 'head') == 'head']
                tail_expos = [x for x in (prefix_expos - prefix_clicks) if self.pop_bucket.get(x, 'head') == 'tail']
                head_expos = [x for x in (prefix_expos - prefix_clicks) if self.pop_bucket.get(x, 'head') == 'head']
                forbid = prefix_clicks | prefix_expos

                c_cold = pick(tail_clicks, set())
                c_hot = pick(head_clicks, set())
                e_cold = pick(tail_expos, set())
                e_hot = pick(head_expos, set())
                rnd = self._random_neq(1, self.itemnum + 1, forbid) if len(forbid) < (self.itemnum - 1) else 0

                packs[idx] = [c_cold or 0, c_hot or 0, e_cold or 0, e_hot or 0, rnd or 0]

        return packs

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        # 构造交错的 user/item token 流
        ext_user_sequence = []
        for record_tuple in user_sequence:
            if len(record_tuple) >= 6:
                u, i, user_feat, item_feat, action_type, ts = record_tuple
            else:
                u, i, user_feat, item_feat, action_type = record_tuple
                ts = 0
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, ts))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, ts))

        T = self.maxlen + 1

        # 基本 ID/类型张量
        seq = np.zeros([T], dtype=np.int64)
        pos = np.zeros([T], dtype=np.int64)
        neg = np.zeros([T], dtype=np.int64)
        token_type = np.zeros([T], dtype=np.int64)
        next_token_type = np.zeros([T], dtype=np.int64)
        next_action_type = np.zeros([T], dtype=np.int64)

        S_i = len(self.item_sparse_keys)
        S_u = len(self.user_sparse_keys)

        seq_item_sparse = np.zeros([T, S_i], dtype=np.int64) if S_i > 0 else None
        seq_user_sparse = np.zeros([T, S_u], dtype=np.int64) if S_u > 0 else None

        pos_item_sparse = np.zeros([T, S_i], dtype=np.int64) if S_i > 0 else None
        neg_item_sparse = np.zeros([T, S_i], dtype=np.int64) if S_i > 0 else None

        # 多模态: fid -> [T, D]
        seq_mm = {fid: np.zeros([T, _SHAPE_DICT_MM[fid]], dtype=np.float32) for fid in self.mm_feat_keys}
        pos_mm = {fid: np.zeros([T, _SHAPE_DICT_MM[fid]], dtype=np.float32) for fid in self.mm_feat_keys}
        neg_mm = {fid: np.zeros([T, _SHAPE_DICT_MM[fid]], dtype=np.float32) for fid in self.mm_feat_keys}

        # Rank 包
        rank_packs = self._build_rank_packs_for_user(ext_user_sequence)

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        nxt = ext_user_sequence[-1]
        idx = T - 1

        rank_ids = np.zeros([T, 5], dtype=np.int64)
        rank_item_sparse = np.zeros([T, 5, S_i], dtype=np.int64) if S_i > 0 else None

        # 时间特征张量（若启用）
        S_t = len(self.time_sparse_keys)
        C_t = len(self.time_cont_keys)
        seq_time_sparse = np.zeros([T, S_t], dtype=np.int64) if S_t > 0 else None
        seq_time_cont = np.zeros([T, C_t], dtype=np.float32) if C_t > 0 else None

        prev_ts = None
        pos_ids = np.zeros([T], dtype=np.int64)
        time_ids = np.zeros([T], dtype=np.int64)
        # CTR continual tensors (optional)
        Cu = len(self.user_cont_keys)
        Ci = len(self.item_cont_keys)
        seq_user_cont = np.zeros([T, Cu], dtype=np.float32) if Cu > 0 else None
        seq_item_cont = np.zeros([T, Ci], dtype=np.float32) if Ci > 0 else None

        for rev_k, record_tuple in enumerate(reversed(ext_user_sequence)):
            i_or_u, feat, type_, act_type, cur_ts = record_tuple
            next_i, next_feat, next_type, next_act_type, _ = nxt

            # 特征补全
            feat = self.fill_missing_feat(feat, i_or_u if type_ == 1 else 0)
            next_feat = self.fill_missing_feat(next_feat, next_i if next_type == 1 else 0)

            if type_ == 1:
                seq[idx] = i_or_u
            else:
                seq[idx] = 0
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type

            pos_ids[idx] = idx + 1
            time_ids[idx] = cur_ts
            # 注入 CTR continual 到当前 token（user/item）
            if seq_user_cont is not None and type_ == 2 and getattr(self, 'use_userctr', False) and getattr(self, '_ctr_loaded', False):
                try:
                    if isinstance(i_or_u, str):
                        uid_reid = self.indexer['u'].get(i_or_u, 0)
                    else:
                        uid_reid = int(i_or_u)
                    raw_u = self.user_ctr.get(int(uid_reid))
                    if raw_u is None:
                        seq_user_cont[idx, 0] = 0.0
                    else:
                        seq_user_cont[idx, 0] = self._ctr_transform(raw_u, self.u_mean, self.u_std)
                except Exception:
                    seq_user_cont[idx, 0] = 0.0
            if seq_item_cont is not None and type_ == 1 and getattr(self, 'use_itemctr', False) and getattr(self, '_ctr_loaded', False):
                try:
                    raw_i = self.item_ctr.get(int(i_or_u))
                    if raw_i is None:
                        seq_item_cont[idx, 0] = 0.0
                    else:
                        seq_item_cont[idx, 0] = self._ctr_transform(raw_i, self.i_mean, self.i_std)
                except Exception:
                    seq_item_cont[idx, 0] = 0.0

            if seq_user_sparse is not None:
                for j, k in enumerate(self.user_sparse_keys):
                    seq_user_sparse[idx, j] = int(feat.get(k, 0))
            if type_ == 1 and seq_item_sparse is not None:
                for j, k in enumerate(self.item_sparse_keys):
                    seq_item_sparse[idx, j] = int(feat.get(k, 0))
            if type_ == 1:
                for fid in self.mm_feat_keys:
                    vec = feat.get(fid, None)
                    if isinstance(vec, np.ndarray):
                        seq_mm[fid][idx] = vec.astype(np.float32, copy=False)

            if (seq_time_sparse is not None or seq_time_cont is not None) and self.use_timefeat:
                if prev_ts is None:
                    dt_hours = 0.0
                else:
                    try:
                        dt_hours = max(0.0, float(abs(cur_ts - prev_ts)) / 3600.0)
                    except Exception:
                        dt_hours = 0.0
                base = self.time_diff_log_base if self.time_diff_log_base > 1.0 else 2.0
                try:
                    t_log = np.log1p(dt_hours) / np.log(base)
                except Exception:
                    t_log = 0.0
                t_log = float(np.clip(t_log, 0.0, float(self.time_diff_max_bucket)))
                bucket = int(min(int(np.floor(t_log) + 1), int(self.time_diff_max_bucket))) if t_log > 0 else 0
                if seq_time_sparse is not None and S_t > 0:
                    if S_t >= 1:
                        seq_time_sparse[idx, 0] = bucket
                    try:
                        dt = datetime.utcfromtimestamp(int(cur_ts))
                        hour = dt.hour
                        weekday = dt.weekday()
                        week = int(dt.strftime('%U'))
                        week = max(1, min(53, week))
                    except Exception:
                        hour, weekday, week = 0, 0, 1
                    if S_t >= 2:
                        seq_time_sparse[idx, 1] = hour
                    if S_t >= 3:
                        seq_time_sparse[idx, 2] = weekday
                    if S_t >= 4:
                        seq_time_sparse[idx, 3] = week
                if seq_time_cont is not None and C_t > 0:
                    seq_time_cont[idx, 0] = t_log

            # 注入 CTR continual 到当前 token（与训练路径保持一致）
            if seq_user_cont is not None and type_ == 2 and getattr(self, 'use_userctr', False) and getattr(self, '_ctr_loaded', False):
                try:
                    if isinstance(i_or_u, str):
                        uid_reid = self.indexer['u'].get(i_or_u, 0)
                    else:
                        uid_reid = int(i_or_u)
                    raw_u = self.user_ctr.get(int(uid_reid))
                    if raw_u is None:
                        seq_user_cont[idx, 0] = 0.0
                    else:
                        seq_user_cont[idx, 0] = self._ctr_transform(raw_u, self.u_mean, self.u_std)
                except Exception:
                    seq_user_cont[idx, 0] = 0.0
            if seq_item_cont is not None and type_ == 1 and getattr(self, 'use_itemctr', False) and getattr(self, '_ctr_loaded', False):
                try:
                    raw_i = self.item_ctr.get(int(i_or_u))
                    if raw_i is None:
                        seq_item_cont[idx, 0] = 0.0
                    else:
                        seq_item_cont[idx, 0] = self._ctr_transform(raw_i, self.i_mean, self.i_std)
                except Exception:
                    seq_item_cont[idx, 0] = 0.0

            if next_type == 1:
                pos[idx] = next_i
                if pos_item_sparse is not None:
                    for j, k in enumerate(self.item_sparse_keys):
                        pos_item_sparse[idx, j] = int(next_feat.get(k, 0))
                for fid in self.mm_feat_keys:
                    nvec = next_feat.get(fid, None)
                    if isinstance(nvec, np.ndarray):
                        pos_mm[fid][idx] = nvec.astype(np.float32, copy=False)

                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
                if neg_item_sparse is not None:
                    for j, k in enumerate(self.item_sparse_keys):
                        neg_item_sparse[idx, j] = int(neg_feat.get(k, 0))
                for fid in self.mm_feat_keys:
                    zvec = neg_feat.get(fid, None)
                    if isinstance(zvec, np.ndarray):
                        neg_mm[fid][idx] = zvec.astype(np.float32, copy=False)

            orig_pos = (len(ext_user_sequence) - 2 - rev_k)
            if 0 <= orig_pos < len(rank_packs):
                pack = rank_packs[orig_pos]
                if pack and any(int(x) > 0 for x in pack):
                    rank_ids[idx, :] = np.array(pack, dtype=np.int64)
                    if rank_item_sparse is not None:
                        for l, cid in enumerate(pack):
                            if cid > 0:
                                try:
                                    cfeat = self.fill_missing_feat(self.item_feat_dict[str(cid)], cid)
                                    for j, k in enumerate(self.item_sparse_keys):
                                        rank_item_sparse[idx, l, j] = int(cfeat.get(k, 0))
                                except Exception:
                                    pass

            nxt = record_tuple
            prev_ts = cur_ts
            idx -= 1
            if idx == -1:
                break

        out = {
            "seq_ids": torch.from_numpy(seq.astype(np.int64)),
            "pos_ids": torch.from_numpy(pos.astype(np.int64)),
            "neg_ids": torch.from_numpy(neg.astype(np.int64)),
            "token_type": torch.from_numpy(token_type.astype(np.int64)),
            "next_token_type": torch.from_numpy(next_token_type.astype(np.int64)),
            "next_action_type": torch.from_numpy(next_action_type.astype(np.int64)),
            "rel_pos_ids": torch.from_numpy(pos_ids.astype(np.int64)),
            "rel_time_ids": torch.from_numpy(time_ids.astype(np.int64)),
        }

        if seq_item_sparse is not None:
            out["item_sparse"] = torch.from_numpy(seq_item_sparse)
        if seq_user_sparse is not None:
            out["user_sparse"] = torch.from_numpy(seq_user_sparse)

        out["mm"] = {fid: torch.from_numpy(seq_mm[fid]) for fid in self.mm_feat_keys}
        out["mm_pos"] = {fid: torch.from_numpy(pos_mm[fid]) for fid in self.mm_feat_keys}
        out["mm_neg"] = {fid: torch.from_numpy(neg_mm[fid]) for fid in self.mm_feat_keys}

        if pos_item_sparse is not None:
            out["pos_item_sparse"] = torch.from_numpy(pos_item_sparse)
        if neg_item_sparse is not None:
            out["neg_item_sparse"] = torch.from_numpy(neg_item_sparse)

        if seq_time_sparse is not None:
            out["time_sparse"] = torch.from_numpy(seq_time_sparse)
        if seq_time_cont is not None:
            out["time_continual"] = torch.from_numpy(seq_time_cont)
        # 输出 CTR continual（若启用）
        if seq_item_cont is not None:
            out["item_cont"] = torch.from_numpy(seq_item_cont)
        if seq_user_cont is not None:
            out["user_cont"] = torch.from_numpy(seq_user_cont)

        out["rank_ids"] = torch.from_numpy(rank_ids)
        if rank_item_sparse is not None:
            out["rank_item_sparse"] = torch.from_numpy(rank_item_sparse)

        return out


def collate_fn_packed(batch):
    keys_to_stack = [
        "seq_ids",
        "pos_ids",
        "neg_ids",
        "token_type",
        "next_token_type",
        "next_action_type",
        "rel_pos_ids",
        "rel_time_ids",
        "item_sparse",
        "user_sparse",
        "time_sparse",
        "time_continual",
        "item_cont",
        "user_cont",
        "pos_item_sparse",
        "neg_item_sparse",
        "rank_ids",
        "rank_item_sparse",
        "uid",
    ]
    out = {}
    for k in keys_to_stack:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    for mmk in ["mm", "mm_pos", "mm_neg"]:
        if mmk in batch[0]:
            fids = list(batch[0][mmk].keys())
            out[mmk] = {fid: torch.stack([b[mmk][fid] for b in batch], dim=0) for fid in fids}

    return out


class PredictSeqDatasetPacked(UserSeqDatasetPacked):
    def __init__(self, data_dir, args, array_max_lens=None):
        try:
            args.use_hard_negative = False
        except Exception:
            pass
        super().__init__(data_dir, args, array_max_lens=array_max_lens)

    def _load_data_and_offsets(self):
        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        self.data_file = None
        self._data_file_pid = None
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def __getitem__(self, uid):
        # 与训练版不同：推理包含最后一个事件进入上下文
        user_sequence = self._load_user_data(uid)

        # 构造交错的 user/item token 流
        ext_user_sequence = []
        for record_tuple in user_sequence:
            if len(record_tuple) >= 6:
                u, i, user_feat, item_feat, action_type, ts = record_tuple
            else:
                u, i, user_feat, item_feat, action_type = record_tuple
                ts = 0
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, ts))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, ts))

        T = self.maxlen + 1

        # 基本 ID/类型张量
        seq = np.zeros([T], dtype=np.int64)
        pos = np.zeros([T], dtype=np.int64)
        neg = np.zeros([T], dtype=np.int64)
        token_type = np.zeros([T], dtype=np.int64)
        next_token_type = np.zeros([T], dtype=np.int64)
        next_action_type = np.zeros([T], dtype=np.int64)

        S_i = len(self.item_sparse_keys)
        S_u = len(self.user_sparse_keys)

        seq_item_sparse = np.zeros([T, S_i], dtype=np.int64) if S_i > 0 else None
        seq_user_sparse = np.zeros([T, S_u], dtype=np.int64) if S_u > 0 else None

        pos_item_sparse = np.zeros([T, S_i], dtype=np.int64) if S_i > 0 else None
        neg_item_sparse = np.zeros([T, S_i], dtype=np.int64) if S_i > 0 else None

        # 多模态: fid -> [T, D]
        seq_mm = {fid: np.zeros([T, _SHAPE_DICT_MM[fid]], dtype=np.float32) for fid in self.mm_feat_keys}
        pos_mm = {fid: np.zeros([T, _SHAPE_DICT_MM[fid]], dtype=np.float32) for fid in self.mm_feat_keys}
        neg_mm = {fid: np.zeros([T, _SHAPE_DICT_MM[fid]], dtype=np.float32) for fid in self.mm_feat_keys}

        # Rank 包
        rank_packs = self._build_rank_packs_for_user(ext_user_sequence)

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # 注意：包含最后一个事件
        nxt = ext_user_sequence[-1]
        idx = T - 1

        rank_ids = np.zeros([T, 5], dtype=np.int64)
        rank_item_sparse = np.zeros([T, 5, S_i], dtype=np.int64) if S_i > 0 else None

        # 时间特征张量（若启用）
        S_t = len(self.time_sparse_keys)
        C_t = len(self.time_cont_keys)
        seq_time_sparse = np.zeros([T, S_t], dtype=np.int64) if S_t > 0 else None
        seq_time_cont = np.zeros([T, C_t], dtype=np.float32) if C_t > 0 else None

        # CTR continual tensors (optional)
        Cu = len(self.user_cont_keys)
        Ci = len(self.item_cont_keys)
        seq_user_cont = np.zeros([T, Cu], dtype=np.float32) if Cu > 0 else None
        seq_item_cont = np.zeros([T, Ci], dtype=np.float32) if Ci > 0 else None

        prev_ts = None
        pos_ids = np.zeros([T], dtype=np.int64)
        time_ids = np.zeros([T], dtype=np.int64)

        for rev_k, record_tuple in enumerate(reversed(ext_user_sequence)):
            i_or_u, feat, type_, act_type, cur_ts = record_tuple
            next_i, next_feat, next_type, next_act_type, _ = nxt

            # 特征补全
            feat = self.fill_missing_feat(feat, i_or_u if type_ == 1 else 0)
            next_feat = self.fill_missing_feat(next_feat, next_i if next_type == 1 else 0)

            if type_ == 1:
                seq[idx] = i_or_u
            else:
                seq[idx] = 0
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type

            pos_ids[idx] = idx + 1
            time_ids[idx] = cur_ts

            if seq_user_sparse is not None:
                for j, k in enumerate(self.user_sparse_keys):
                    seq_user_sparse[idx, j] = int(feat.get(k, 0))
            if type_ == 1 and seq_item_sparse is not None:
                for j, k in enumerate(self.item_sparse_keys):
                    seq_item_sparse[idx, j] = int(feat.get(k, 0))
            if type_ == 1:
                for fid in self.mm_feat_keys:
                    vec = feat.get(fid, None)
                    if isinstance(vec, np.ndarray):
                        seq_mm[fid][idx] = vec.astype(np.float32, copy=False)

            if (seq_time_sparse is not None or seq_time_cont is not None) and self.use_timefeat:
                if prev_ts is None:
                    dt_hours = 0.0
                else:
                    try:
                        dt_hours = max(0.0, float(abs(cur_ts - prev_ts)) / 3600.0)
                    except Exception:
                        dt_hours = 0.0
                base = self.time_diff_log_base if self.time_diff_log_base > 1.0 else 2.0
                try:
                    t_log = np.log1p(dt_hours) / np.log(base)
                except Exception:
                    t_log = 0.0
                t_log = float(np.clip(t_log, 0.0, float(self.time_diff_max_bucket)))
                bucket = int(min(int(np.floor(t_log) + 1), int(self.time_diff_max_bucket))) if t_log > 0 else 0
                if seq_time_sparse is not None and S_t > 0:
                    if S_t >= 1:
                        seq_time_sparse[idx, 0] = bucket
                    try:
                        dt = datetime.utcfromtimestamp(int(cur_ts))
                        hour = dt.hour
                        weekday = dt.weekday()
                        week = int(dt.strftime('%U'))
                        week = max(1, min(53, week))
                    except Exception:
                        hour, weekday, week = 0, 0, 1
                    if S_t >= 2:
                        seq_time_sparse[idx, 1] = hour
                    if S_t >= 3:
                        seq_time_sparse[idx, 2] = weekday
                    if S_t >= 4:
                        seq_time_sparse[idx, 3] = week
                if seq_time_cont is not None and C_t > 0:
                    seq_time_cont[idx, 0] = t_log

            # 注入 CTR continual 到当前 token（与训练路径保持一致）
            if seq_user_cont is not None and type_ == 2 and getattr(self, 'use_userctr', False) and getattr(self, '_ctr_loaded', False):
                try:
                    if isinstance(i_or_u, str):
                        uid_reid = self.indexer['u'].get(i_or_u, 0)
                    else:
                        uid_reid = int(i_or_u)
                    raw_u = self.user_ctr.get(int(uid_reid))
                    if raw_u is None:
                        seq_user_cont[idx, 0] = 0.0
                    else:
                        seq_user_cont[idx, 0] = self._ctr_transform(raw_u, self.u_mean, self.u_std)
                except Exception:
                    seq_user_cont[idx, 0] = 0.0
            if seq_item_cont is not None and type_ == 1 and getattr(self, 'use_itemctr', False) and getattr(self, '_ctr_loaded', False):
                try:
                    raw_i = self.item_ctr.get(int(i_or_u))
                    if raw_i is None:
                        seq_item_cont[idx, 0] = 0.0
                    else:
                        seq_item_cont[idx, 0] = self._ctr_transform(raw_i, self.i_mean, self.i_std)
                except Exception:
                    seq_item_cont[idx, 0] = 0.0

            if next_type == 1:
                pos[idx] = next_i
                if pos_item_sparse is not None:
                    for j, k in enumerate(self.item_sparse_keys):
                        pos_item_sparse[idx, j] = int(next_feat.get(k, 0))
                for fid in self.mm_feat_keys:
                    nvec = next_feat.get(fid, None)
                    if isinstance(nvec, np.ndarray):
                        pos_mm[fid][idx] = nvec.astype(np.float32, copy=False)

                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
                if neg_item_sparse is not None:
                    for j, k in enumerate(self.item_sparse_keys):
                        neg_item_sparse[idx, j] = int(neg_feat.get(k, 0))
                for fid in self.mm_feat_keys:
                    zvec = neg_feat.get(fid, None)
                    if isinstance(zvec, np.ndarray):
                        neg_mm[fid][idx] = zvec.astype(np.float32, copy=False)

            # rank 包位置映射
            orig_pos = (len(ext_user_sequence) - 1 - rev_k)
            if 0 <= orig_pos - 1 < len(rank_packs):
                pack = rank_packs[orig_pos - 1]
                if pack and any(int(x) > 0 for x in pack):
                    rank_ids[idx, :] = np.array(pack, dtype=np.int64)
                    if rank_item_sparse is not None:
                        for l, cid in enumerate(pack):
                            if cid > 0:
                                try:
                                    cfeat = self.fill_missing_feat(self.item_feat_dict[str(cid)], cid)
                                    for j, k in enumerate(self.item_sparse_keys):
                                        rank_item_sparse[idx, l, j] = int(cfeat.get(k, 0))
                                except Exception:
                                    pass

            nxt = record_tuple
            prev_ts = cur_ts
            idx -= 1
            if idx == -1:
                break

        out = {
            "seq_ids": torch.from_numpy(seq.astype(np.int64)),
            "pos_ids": torch.from_numpy(pos.astype(np.int64)),
            "neg_ids": torch.from_numpy(neg.astype(np.int64)),
            "token_type": torch.from_numpy(token_type.astype(np.int64)),
            "next_token_type": torch.from_numpy(next_token_type.astype(np.int64)),
            "next_action_type": torch.from_numpy(next_action_type.astype(np.int64)),
            "rel_pos_ids": torch.from_numpy(pos_ids.astype(np.int64)),
            "rel_time_ids": torch.from_numpy(time_ids.astype(np.int64)),
        }

        if seq_item_sparse is not None:
            out["item_sparse"] = torch.from_numpy(seq_item_sparse)
        if seq_user_sparse is not None:
            out["user_sparse"] = torch.from_numpy(seq_user_sparse)

        out["mm"] = {fid: torch.from_numpy(seq_mm[fid]) for fid in self.mm_feat_keys}
        out["mm_pos"] = {fid: torch.from_numpy(pos_mm[fid]) for fid in self.mm_feat_keys}
        out["mm_neg"] = {fid: torch.from_numpy(neg_mm[fid]) for fid in self.mm_feat_keys}

        if pos_item_sparse is not None:
            out["pos_item_sparse"] = torch.from_numpy(pos_item_sparse)
        if neg_item_sparse is not None:
            out["neg_item_sparse"] = torch.from_numpy(neg_item_sparse)

        if seq_time_sparse is not None:
            out["time_sparse"] = torch.from_numpy(seq_time_sparse)
        if seq_time_cont is not None:
            out["time_continual"] = torch.from_numpy(seq_time_cont)

        # 输出 CTR continual（若启用）
        if seq_item_cont is not None:
            out["item_cont"] = torch.from_numpy(seq_item_cont)
        if seq_user_cont is not None:
            out["user_cont"] = torch.from_numpy(seq_user_cont)

        out["rank_ids"] = torch.from_numpy(rank_ids)
        if rank_item_sparse is not None:
            out["rank_item_sparse"] = torch.from_numpy(rank_item_sparse)

        out["uid"] = torch.tensor(uid, dtype=torch.long)
        return out

    def _load_user_data(self, uid):
        data = super(PredictSeqDatasetPacked, self)._load_user_data(uid)
        fixed = []
        for record in data:
            try:
                if len(record) >= 6:
                    u, i, user_feat, item_feat, action_type, ts = record
                else:
                    u, i, user_feat, item_feat, action_type = record
                    ts = 0
                if i and isinstance(i, int) and i > self.itemnum:
                    i = 0
                fixed.append((u, i, user_feat, item_feat, action_type, ts))
            except Exception:
                continue
        return fixed