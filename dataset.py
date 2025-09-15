import json
import os
import pickle
import struct
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        # 时间特征开关与参数（用于 t_gap_bucket 与 t_diff_log）
        self.use_timefeat = getattr(args, 'use_timefeat', False)
        self.time_diff_log_base = float(getattr(args, 'time_diff_log_base', 2.0))
        self.time_diff_max_bucket = int(getattr(args, 'time_diff_max_bucket', 63))

        # 可配置的每个位置负样本数量（默认4）
        self.num_negatives = getattr(args, 'num_negatives', 4)
        # 负采样策略配置
        self.use_hard_negative = getattr(args, 'use_hard_negative', False)
        self.hard_negative_ratio = getattr(args, 'hard_negative_ratio', 0.5)  # 困难样本比例 3/4
        self.top_k_popular = getattr(args, 'top_k_popular', 10000)
        # 可选：载入 popularity 分桶（优雅降级）
        self.pop_bucket = {}
        pop_file = Path(self.data_dir, 'item_pop.json')
        if pop_file.exists():
            try:
                with open(pop_file, 'r', encoding='utf-8') as f:
                    j = json.load(f)
                    for k, v in j.items():
                        try:
                            self.pop_bucket[int(k)] = v.get('bucket', 'head')
                        except Exception:
                            pass
            except Exception:
                self.pop_bucket = {}

        # 行为口径（点击/曝光 action id 集合）
        try:
            self.click_set = set(int(x) for x in str(getattr(args, 'click_actions', '1')).split(',') if x != '')
            self.expo_set = set(int(x) for x in str(getattr(args, 'expo_actions', '0')).split(',') if x != '')
        except Exception:
            self.click_set = {1}
            self.expo_set = {0}

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        # CTR feature switches and artifacts
        self.use_ctr = bool(getattr(args, 'use_ctr', False))
        # In inference, allow independent ablation of user/item CTR regardless of use_ctr
        self.force_ctr_channels = bool(getattr(args, 'inference_only', False))
        if self.force_ctr_channels:
            self.use_userctr = bool(getattr(args, 'use_userctr', False))
            self.use_itemctr = bool(getattr(args, 'use_itemctr', False))
        else:
            self.use_userctr = bool(getattr(args, 'use_userctr', False)) or self.use_ctr
            self.use_itemctr = bool(getattr(args, 'use_itemctr', False)) or self.use_ctr
        # IDs for continual CTR features (string keys align with existing feature-id convention)
        self.U_CTR_ID = '150'
        self.I_CTR_ID = '151'

        # CTR artifacts (loaded lazily after _init_feat_info sets types)
        self.user_ctr = None
        self.item_ctr = None
        self.ctr_meta = None
        self._ctr_loaded = False

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        # Load CTR dictionaries and transform params if enabled
        if self.use_userctr or self.use_itemctr:
            self._load_ctr_artifacts(args)
        
        # 初始化负采样相关数据结构
        if self.use_hard_negative:
            self._init_hard_negative_sampling()

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        # 延迟在 worker 内打开文件，避免多进程共享同一文件描述符导致竞态
        self.data_file_path = self.data_dir / "seq.jsonl"
        self.data_file = None
        self._data_file_pid = None
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_ctr_artifacts(self, args):
        """Load CTR tables and meta; default path under USER_CACHE_PATH/ctr or data_dir/ctr."""
        if self._ctr_loaded:
            return
        # default directory resolution
        ctr_dir = getattr(args, 'ctr_dir', None)
        if not ctr_dir or not os.path.isdir(ctr_dir):
            base = os.environ.get('USER_CACHE_PATH', str(self.data_dir))
            ctr_dir = os.path.join(base, 'ctr')
        try:
            with open(os.path.join(ctr_dir, 'ctr_meta.json'), 'r', encoding='utf-8') as f:
                self.ctr_meta = json.load(f)
            if self.use_userctr:
                with open(os.path.join(ctr_dir, 'user_ctr.json'), 'r', encoding='utf-8') as f:
                    tmp = json.load(f)
                # keys maybe str
                self.user_ctr = {int(k): float(v) for k, v in tmp.items()}
            if self.use_itemctr:
                with open(os.path.join(ctr_dir, 'item_ctr.json'), 'r', encoding='utf-8') as f:
                    tmp = json.load(f)
                self.item_ctr = {int(k): float(v) for k, v in tmp.items()}

            # transform params
            self.eps = float(self.ctr_meta['transform']['eps'])
            self.logit = bool(self.ctr_meta['transform']['logit'])
            self.u_mean = float(self.ctr_meta['user']['mean'])
            self.u_std = float(self.ctr_meta['user']['std'])
            self.i_mean = float(self.ctr_meta['item']['mean'])
            self.i_std = float(self.ctr_meta['item']['std'])
            a_u = float(self.ctr_meta['user']['alpha'])
            b_u = float(self.ctr_meta['user']['beta'])
            a_i = float(self.ctr_meta['item']['alpha'])
            b_i = float(self.ctr_meta['item']['beta'])
            self.u_prior = a_u / max(a_u + b_u, 1e-12)
            self.i_prior = a_i / max(a_i + b_i, 1e-12)
            self._ctr_loaded = True
            print(
                f"[CTR] loaded from {ctr_dir}: user(k={self.ctr_meta['user']['k']:.2f}, mean={self.u_mean:.4f}, std={self.u_std:.4f}) "
                f"item(k={self.ctr_meta['item']['k']:.2f}, mean={self.i_mean:.4f}, std={self.i_std:.4f}), logit={self.logit}, eps={self.eps}"
            )
        except Exception as e:
            print(f"[CTR] failed to load from {ctr_dir}: {e}")
            # disable if load failed
            self.use_userctr = False
            self.use_itemctr = False

    def _ctr_transform(self, x, mean, std):
        import math
        x = min(max(float(x), getattr(self, 'eps', 1e-5)), 1.0 - getattr(self, 'eps', 1e-5))
        if getattr(self, 'logit', False):
            x = math.log(x / (1.0 - x))
        denom = std if std and std > 0 else 1.0
        return float((x - mean) / denom)

    def _ensure_open_file(self):
        """确保在当前进程中打开独立的文件句柄。"""
        current_pid = os.getpid()
        if self.data_file is None or self._data_file_pid != current_pid or getattr(self.data_file, 'closed', False):
            # 重新打开，绑定到当前进程
            self.data_file = open(self.data_file_path, 'rb')
            self._data_file_pid = current_pid

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self._ensure_open_file()
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        current_user_id = None  # 追踪当前用户ID
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                current_user_id = u  # 更新当前用户ID
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        # K 负样本：[T, K]
        neg = np.zeros([self.maxlen + 1, self.num_negatives], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        # 负样本特征：[T, K]
        neg_feat = np.empty([self.maxlen + 1, self.num_negatives], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                
                # 使用新的负采样策略：困难样本+随机样本混合
                if self.use_hard_negative and current_user_id is not None:
                    neg_samples = self._get_negative_samples_new(current_user_id, ts, self.num_negatives)
                    for k, neg_id in enumerate(neg_samples):
                        if k < self.num_negatives:  # 防止越界
                            neg[idx, k] = neg_id
                            neg_feat[idx, k] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
                else:
                    # 原始采样策略：采样K个不在序列中的负样本（并尽量去重）
                    used_neg = set()
                    for k in range(self.num_negatives):
                        neg_id = self._random_neq(1, self.itemnum + 1, ts.union(used_neg))
                        used_neg.add(neg_id)
                        neg[idx, k] = neg_id
                        neg_feat[idx, k] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            ##'111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        # 启用时间特征：
        # - time_sparse: t_gap_bucket（对数分桶离散特征，用于Embedding）
        # - time_continual: t_diff_log（log1p 归一化连续特征）
        if getattr(self, 'use_timefeat', False):
            # 为时间特征分配特征ID：
            #   - 130: t_gap_bucket（对数分桶）
            #   - 132: hour（0..23）
            #   - 133: weekday（0..6，周一=0）
            #   - 134: week_of_year（1..53，0 作为 padding 起点）
            #   - 131: t_diff_log（连续）
            T_GAP_BUCKET_ID = '130'
            T_HOUR_ID = '132'
            T_WEEKDAY_ID = '133'
            T_WEEK_ID = '134'
            T_DIFF_LOG_ID = '131'
            feat_types['time_sparse'] = [T_GAP_BUCKET_ID, T_HOUR_ID, T_WEEKDAY_ID, T_WEEK_ID]
            feat_types['time_continual'] = [T_DIFF_LOG_ID]
            # 默认值
            feat_default_value[T_GAP_BUCKET_ID] = 0
            feat_default_value[T_HOUR_ID] = 0
            feat_default_value[T_WEEKDAY_ID] = 0
            feat_default_value[T_WEEK_ID] = 0
            feat_default_value[T_DIFF_LOG_ID] = 0
            # 统计量（不含 padding 0，模型端 +1 作为 padding_idx）：
            # t_gap_bucket: 1..max_bucket；hour: 0..23；weekday: 0..6；week: 1..53
            feat_statistics[T_GAP_BUCKET_ID] = int(self.time_diff_max_bucket)
            feat_statistics[T_HOUR_ID] = 24
            feat_statistics[T_WEEKDAY_ID] = 7
            feat_statistics[T_WEEK_ID] = 53

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        # Append CTR continual feature IDs
        # - If in inference_only mode: always include to keep model shapes aligned with training, values may be zeroed later
        # - Otherwise (training): include only when explicitly enabled
        if getattr(self, 'force_ctr_channels', False) or getattr(self, 'use_userctr', False):
            if self.U_CTR_ID not in feat_types['user_continual']:
                feat_types['user_continual'].append(self.U_CTR_ID)
        if getattr(self, 'force_ctr_channels', False) or getattr(self, 'use_itemctr', False):
            if self.I_CTR_ID not in feat_types['item_continual']:
                feat_types['item_continual'].append(self.I_CTR_ID)

        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for key, feat_type in self.feature_types.items():
            # 排除时间特征键，时间特征单独在 Packed 路径中生成
            if key in ('time_sparse', 'time_continual'):
                continue
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    def _init_hard_negative_sampling(self):
        """
        初始化Hard Negative采样所需的数据结构
        """
        # 使用环境变量USER_CACHE_PATH作为缓存目录
        cache_dir = Path(os.environ.get('USER_CACHE_PATH', str(self.data_dir)))
        cache_file = cache_dir / "hard_negative_cache.pkl"
        
        # 确保缓存目录存在
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_file.exists():
            print("Loading hard negative sampling cache...")
            self._load_hard_negative_cache(cache_file)
        else:
            print("Building hard negative sampling cache...")
            self._build_hard_negative_cache()
            self._save_hard_negative_cache(cache_file)
    
    def _load_hard_negative_cache(self, cache_file):
        """加载Hard Negative缓存"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.item_popularity = cache_data['item_popularity']
        self.popular_items = cache_data['popular_items']
        self.user_history_clicks = cache_data['user_history_clicks']
        self.user_hard_samples = cache_data['user_hard_samples']
        
        print(f"Loaded cache: {len(self.popular_items)} popular items, "
              f"{len(self.user_history_clicks)} users with history")
    
    def _save_hard_negative_cache(self, cache_file):
        """保存Hard Negative缓存"""
        cache_data = {
            'item_popularity': self.item_popularity,
            'popular_items': self.popular_items,
            'user_history_clicks': self.user_history_clicks,
            'user_hard_samples': self.user_hard_samples
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved hard negative cache to {cache_file}")
    
    def _build_hard_negative_cache(self):
        """
        从seq.jsonl构建Hard Negative采样所需的数据结构
        """
        # 初始化数据结构
        self.item_popularity = Counter()
        self.user_history_clicks = defaultdict(set)
        
        # 直接遍历seq.jsonl文件
        seq_file = self.data_dir / "seq.jsonl"
        if not seq_file.exists():
            raise FileNotFoundError(f"seq.jsonl not found in {self.data_dir}")
        
        print("Analyzing seq.jsonl for hard negative sampling...")
        total_lines = 0
        total_interactions = 0
        error_lines = 0
        
        with open(seq_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Processing seq.jsonl"), 1):
                total_lines += 1
                try:
                    user_sequence = json.loads(line.strip())
                    current_user_id = None
                    
                    for record in user_sequence:
                        if len(record) >= 6:
                            u_id, i_id, _, _, action_type, _ = record[:6]
                            
                            # 更新当前用户ID
                            if u_id is not None:
                                current_user_id = u_id
                            
                            # 处理物品交互
                            if i_id is not None and current_user_id is not None:
                                # 统计物品流行度（所有交互类型）
                                self.item_popularity[i_id] += 1
                                total_interactions += 1
                                
                                # 记录用户历史点击（包括点击和曝光）
                                self.user_history_clicks[current_user_id].add(i_id)
                                
                except (json.JSONDecodeError, ValueError) as e:
                    error_lines += 1
                    if error_lines <= 5:  # 只打印前5个错误
                        print(f"Error parsing line {line_num}: {e}")
                    continue
        
        # 遍历完成，打印统计信息
        print(f"\n=== seq.jsonl 遍历完成 ===")
        print(f"总处理行数: {total_lines}")
        print(f"错误行数: {error_lines}")
        print(f"总交互次数: {total_interactions}")
        print(f"唯一物品数: {len(self.item_popularity)}")
        print(f"唯一用户数: {len(self.user_history_clicks)}")
        if self.item_popularity:
            avg_interactions = total_interactions / len(self.item_popularity)
            print(f"平均每个物品交互次数: {avg_interactions:.2f}")
        if self.user_history_clicks:
            avg_user_items = sum(len(items) for items in self.user_history_clicks.values()) / len(self.user_history_clicks)
            print(f"平均每个用户交互物品数: {avg_user_items:.2f}")
        
        # 选择Top-K热门物品
        popular_items_list = self.item_popularity.most_common(self.top_k_popular)
        self.popular_items = {item_id for item_id, _ in popular_items_list}
        
        # 为每个用户构建Hard Sample集合
        print("Building hard samples for each user...")
        self.user_hard_samples = {}
        for user_id, history in self.user_history_clicks.items():
            # Hard Sample = 热门物品 - 用户历史点击集合
            self.user_hard_samples[user_id] = self.popular_items - history
        
        print(f"Built hard negative cache: "
              f"{len(self.item_popularity)} total items, "
              f"{len(self.popular_items)} popular items, "
              f"{len(self.user_history_clicks)} users with history")
    
    def _get_negative_samples_new(self, user_id, exclude_items, num_samples=4):
        """
        新的负采样策略：困难样本 + 随机样本混合
        
        Args:
            user_id: 用户ID
            exclude_items: 需要排除的物品集合
            num_samples: 总负样本数量
            
        Returns:
            List[int]: 负样本ID列表
        """
        if not self.use_hard_negative:
            # 如果不使用Hard Negative，回退到原始随机采样
            return self._get_random_negative_samples(exclude_items, num_samples)
        
        # 计算困难样本和随机样本数量
        num_hard = int(num_samples * self.hard_negative_ratio)
        num_random = num_samples - num_hard
        
        negative_samples = []
        used_samples = set(exclude_items)
        
        # 1. 采样困难样本
        if num_hard > 0 and user_id in self.user_hard_samples:
            hard_candidates = self.user_hard_samples[user_id] - used_samples
            # 过滤掉不在item_feat_dict中的物品
            hard_candidates = {item for item in hard_candidates if str(item) in self.item_feat_dict}
            
            if hard_candidates:
                hard_samples = list(hard_candidates)
                np.random.shuffle(hard_samples)
                
                actual_hard = min(num_hard, len(hard_samples))
                selected_hard = hard_samples[:actual_hard]
                negative_samples.extend(selected_hard)
                used_samples.update(selected_hard)
                
                # 如果困难样本不足，剩余的用随机样本补充
                num_random += (num_hard - actual_hard)
        else:
            # 如果没有困难样本，全部用随机样本
            num_random += num_hard
        
        # 2. 采样随机样本
        if num_random > 0:
            random_samples = self._get_random_negative_samples(used_samples, num_random)
            negative_samples.extend(random_samples)
        
        return negative_samples[:num_samples]
    
    def _get_random_negative_samples(self, exclude_items, num_samples):
        """
        获取随机负样本 - 保持原有逻辑不变
        
        Args:
            exclude_items: 排除的物品集合
            num_samples: 样本数量
            
        Returns:
            List[int]: 随机负样本列表
        """
        samples = []
        used_neg = set(exclude_items)
        for _ in range(num_samples):
            neg_id = self._random_neq(1, self.itemnum + 1, used_neg)
            samples.append(neg_id)
            used_neg.add(neg_id)
        return samples

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        # 测试数据集禁用Hard Negative采样
        args.use_hard_negative = False
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        # 预测文件同样按 worker 懒打开
        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        self.data_file = None
        self._data_file_pid = None
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
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

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            # 与原数据保持兼容：若记录中无时间戳字段则以0占位
            if len(record_tuple) >= 6:
                u, i, user_feat, item_feat, action_type, ts = record_tuple
            else:
                u, i, user_feat, item_feat, action_type = record_tuple
                ts = 0
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, ts))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1, action_type, ts))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        
        # 位置和时间ID张量（用于相对偏置）
        pos_ids = np.zeros([self.maxlen + 1], dtype=np.int64)
        time_ids = np.zeros([self.maxlen + 1], dtype=np.int64)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # 准备用户reid（用于 user CTR 注入）
        try:
            uid_reid = self.indexer['u'].get(user_id, 0)
        except Exception:
            uid_reid = 0

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, action_type, cur_ts = record_tuple
            feat = self.fill_missing_feat(feat, i)

            # 注入 CTR 连续特征到 seq_feat（与训练保持一致）
            if self._ctr_loaded:
                # user 侧 CTR（固定使用该用户的全局 CTR）
                if type_ == 2 and getattr(self, 'use_userctr', False) and self.U_CTR_ID in self.feature_types['user_continual']:
                    try:
                        raw_u = self.user_ctr.get(int(uid_reid))
                        if raw_u is None:
                            feat[self.U_CTR_ID] = 0.0
                        else:
                            feat[self.U_CTR_ID] = self._ctr_transform(raw_u, self.u_mean, self.u_std)
                    except Exception:
                        feat[self.U_CTR_ID] = 0.0
                # item 侧 CTR（按当前 item id 注入）
                if type_ == 1 and getattr(self, 'use_itemctr', False) and self.I_CTR_ID in self.feature_types['item_continual']:
                    try:
                        raw_i = self.item_ctr.get(int(i)) if int(i) > 0 else None
                        if raw_i is None:
                            feat[self.I_CTR_ID] = 0.0
                        else:
                            feat[self.I_CTR_ID] = self._ctr_transform(raw_i, self.i_mean, self.i_std)
                    except Exception:
                        feat[self.I_CTR_ID] = 0.0
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            
            # 填充位置和时间ID（用于相对偏置）
            pos_ids[idx] = idx + 1  # 位置从1开始
            time_ids[idx] = cur_ts  # 时间戳
            
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id, pos_ids, time_ids

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
            pos_ids: 位置ID, torch.Tensor形式
            time_ids: 时间ID, torch.Tensor形式
        """
        seq, token_type, seq_feat, user_id, pos_ids, time_ids = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)
        pos_ids = torch.from_numpy(np.array(pos_ids))
        time_ids = torch.from_numpy(np.array(time_ids))

        return seq, token_type, seq_feat, user_id, pos_ids, time_ids


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('part-*'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict


# =========================
# Packed Datasets (tensorized inside workers)
# =========================

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
        self.array_max_lens = array_max_lens  # dict: feat_id -> max_len, 可为 None

        # 预先缓存键顺序，便于张量打包（需与 model 中的 Embedding Dict 顺序一致）
        self.user_sparse_keys = list(self.feature_types['user_sparse'])
        self.item_sparse_keys = list(self.feature_types['item_sparse'])
        self.user_array_keys = list(self.feature_types['user_array']) if self.array_max_lens else []
        self.item_array_keys = list(self.feature_types['item_array']) if self.array_max_lens else []
        self.mm_feat_keys = list(self.feature_types['item_emb'])  # 多模态仅在 item 侧

        # 连续特征占位（当前为空列表，保留接口）
        self.user_cont_keys = list(self.feature_types['user_continual'])
        self.item_cont_keys = list(self.feature_types['item_continual'])

        # 时间特征键（若未启用则为空）
        self.time_sparse_keys = list(self.feature_types.get('time_sparse', []))
        self.time_cont_keys = list(self.feature_types.get('time_continual', []))

    def _build_rank_packs_for_user(self, ext_user_sequence):
        """
        基于时间前缀构造五级候选包：冷点 > 热点 > 冷曝 > 热曝 > 随机未曝
        输入：ext_user_sequence: [(id, feat, type, action_type)] 按时间顺序
        输出：长度 len(ext_user_sequence)-1 的列表，每个元素为5槽 [i1,i2,i3,i4,i5]（0表示缺）
        仅在 next_type==1 的位置生效，其余位置为全0
        """
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
            # 兼容含时间戳的五元组：(id, feat, type, action_type, ts)
            if len(cur) >= 5:
                cur_id, _, cur_type, _ = cur[0], cur[1], cur[2], cur[3]
            else:
                cur_id, _, cur_type, _ = cur
            if len(nxt) >= 5:
                next_i, _, next_type, _ = nxt[0], nxt[1], nxt[2], nxt[3]
            else:
                next_i, _, next_type, _ = nxt

            # 更新前缀集合（当前记录属于过去）
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

            # 仅当下一个是 item 时构造 rank 包
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
            # 与原数据保持兼容：若记录中无时间戳字段则以0占位
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

        # 基本 ID/类型张量（numpy -> 最终转 torch）
        seq = np.zeros([T], dtype=np.int64)
        pos = np.zeros([T], dtype=np.int64)
        neg = np.zeros([T], dtype=np.int64)
        token_type = np.zeros([T], dtype=np.int64)
        next_token_type = np.zeros([T], dtype=np.int64)
        next_action_type = np.zeros([T], dtype=np.int64)

        # 三种特征视角：当前 token（seq_*），下一个正样本（pos_*），随机负样本（neg_*）
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

        # Rank 包初始化（基于时间前缀）
        rank_packs = self._build_rank_packs_for_user(ext_user_sequence)

        # 负采样候选集合（加速 _random_neq）
        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # 逐步填充（left-padding）
        nxt = ext_user_sequence[-1]
        idx = T - 1

        # 预分配 rank 张量
        rank_ids = np.zeros([T, 5], dtype=np.int64)
        rank_item_sparse = np.zeros([T, 5, S_i], dtype=np.int64) if S_i > 0 else None

        # 时间特征张量（若启用）
        S_t = len(self.time_sparse_keys)
        C_t = len(self.time_cont_keys)
        seq_time_sparse = np.zeros([T, S_t], dtype=np.int64) if S_t > 0 else None
        seq_time_cont = np.zeros([T, C_t], dtype=np.float32) if C_t > 0 else None

        # 时间差计算辅助
        prev_ts = None
        
        # 位置和时间ID张量（用于相对偏置）
        pos_ids = np.zeros([T], dtype=np.int64)
        time_ids = np.zeros([T], dtype=np.int64)

        # CTR continual tensors (optional)
        Cu = len(self.user_cont_keys)
        Ci = len(self.item_cont_keys)
        seq_user_cont = np.zeros([T, Cu], dtype=np.float32) if Cu > 0 else None
        seq_item_cont = np.zeros([T, Ci], dtype=np.float32) if Ci > 0 else None
        pos_item_cont = np.zeros([T, Ci], dtype=np.float32) if Ci > 0 else None
        neg_item_cont = np.zeros([T, Ci], dtype=np.float32) if Ci > 0 else None
        rank_item_cont = np.zeros([T, 5, Ci], dtype=np.float32) if Ci > 0 else None

        for rev_k, record_tuple in enumerate(reversed(ext_user_sequence[:-1])):
            i_or_u, feat, type_, act_type, cur_ts = record_tuple
            next_i, next_feat, next_type, next_act_type, _ = nxt

            # 特征补全（包含 user/item sparse/array/mm/continual 键）
            feat = self.fill_missing_feat(feat, i_or_u if type_ == 1 else 0)
            next_feat = self.fill_missing_feat(next_feat, next_i if next_type == 1 else 0)

            # 基本序列/类型
            if type_ == 1:
                seq[idx] = i_or_u  # 仅保留 item id，user 位保持 0
            else:
                seq[idx] = 0
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            
            # 填充位置和时间ID（用于相对偏置）
            pos_ids[idx] = idx + 1  # 位置从1开始
            time_ids[idx] = cur_ts  # 时间戳

            # 注入 CTR continual 到当前 token
            if Cu > 0 and type_ == 2 and self.use_userctr and self._ctr_loaded:
                try:
                    raw = self.user_ctr.get(int(i_or_u))
                    if raw is None:
                        seq_user_cont[idx, 0] = 0.0
                    else:
                        seq_user_cont[idx, 0] = self._ctr_transform(raw, self.u_mean, self.u_std)
                except Exception:
                    seq_user_cont[idx, 0] = 0.0
            if Ci > 0 and type_ == 1 and self.use_itemctr and self._ctr_loaded:
                try:
                    raw_i = self.item_ctr.get(int(i_or_u))
                    if raw_i is None:
                        seq_item_cont[idx, 0] = 0.0
                    else:
                        seq_item_cont[idx, 0] = self._ctr_transform(raw_i, self.i_mean, self.i_std)
                except Exception:
                    seq_item_cont[idx, 0] = 0.0

            # 当前 token 的 item/user sparse + mm（item 位才写入 item 侧）
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

            # 时间特征（基于相邻事件的时间差，单位小时；对数分桶 + 连续log值）
            if (seq_time_sparse is not None or seq_time_cont is not None) and self.use_timefeat:
                if prev_ts is None:
                    dt_hours = 0.0
                else:
                    try:
                        dt_hours = max(0.0, float(abs(cur_ts - prev_ts)) / 3600.0)
                    except Exception:
                        dt_hours = 0.0

                # 连续特征：log1p 归一化到以 base 为底，并裁剪到 max_bucket
                base = self.time_diff_log_base if self.time_diff_log_base > 1.0 else 2.0
                try:
                    t_log = np.log1p(dt_hours) / np.log(base)
                except Exception:
                    t_log = 0.0
                t_log = float(np.clip(t_log, 0.0, float(self.time_diff_max_bucket)))

                # 离散桶：取 floor(t_log)+1（0 作为 padding），上限为 time_diff_max_bucket
                bucket = int(min(int(np.floor(t_log) + 1), int(self.time_diff_max_bucket))) if t_log > 0 else 0

                # time_sparse 列顺序严格按照 self.time_sparse_keys
                if seq_time_sparse is not None and S_t > 0:
                    # 1) gap_bucket -> 第一个列
                    if S_t >= 1:
                        seq_time_sparse[idx, 0] = bucket
                    # 2) hour/weekday/week -> 其后列
                    try:
                        dt = datetime.utcfromtimestamp(int(cur_ts))
                        hour = dt.hour
                        weekday = (dt.weekday())  # Monday=0..Sunday=6
                        week = int(dt.strftime('%U'))  # week number, 00-53 -> 使用 1..53，有效值 1..53
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
                    # 连续 t_diff_log -> 第一个列
                    seq_time_cont[idx, 0] = t_log

            # 正样本（仅 item 侧）
            if next_type == 1:
                pos[idx] = next_i
                if pos_item_sparse is not None:
                    for j, k in enumerate(self.item_sparse_keys):
                        pos_item_sparse[idx, j] = int(next_feat.get(k, 0))
                for fid in self.mm_feat_keys:
                    nvec = next_feat.get(fid, None)
                    if isinstance(nvec, np.ndarray):
                        pos_mm[fid][idx] = nvec.astype(np.float32, copy=False)
                if Ci > 0 and self.use_itemctr and self._ctr_loaded:
                    try:
                        raw_pos = self.item_ctr.get(int(next_i))
                        if raw_pos is None:
                            pos_item_cont[idx, 0] = 0.0
                        else:
                            pos_item_cont[idx, 0] = self._ctr_transform(raw_pos, self.i_mean, self.i_std)
                    except Exception:
                        pos_item_cont[idx, 0] = 0.0

                # 负样本（随机）
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
                if Ci > 0 and self.use_itemctr and self._ctr_loaded:
                    try:
                        raw_neg = self.item_ctr.get(int(neg_id))
                        if raw_neg is None:
                            neg_item_cont[idx, 0] = 0.0
                        else:
                            neg_item_cont[idx, 0] = self._ctr_transform(raw_neg, self.i_mean, self.i_std)
                    except Exception:
                        neg_item_cont[idx, 0] = 0.0

            # 写入 rank 包（根据原序列的正向位置）
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
                    if rank_item_cont is not None and self.use_itemctr and self._ctr_loaded:
                        for l, cid in enumerate(pack):
                            if cid > 0:
                                try:
                                    raw_c = self.item_ctr.get(int(cid))
                                    if raw_c is None:
                                        rank_item_cont[idx, l, 0] = 0.0
                                    else:
                                        rank_item_cont[idx, l, 0] = self._ctr_transform(raw_c, self.i_mean, self.i_std)
                                except Exception:
                                    rank_item_cont[idx, l, 0] = 0.0

            nxt = record_tuple
            # 更新 prev_ts 为当前事件时间戳
            prev_ts = cur_ts
            idx -= 1
            if idx == -1:
                break

        # 构建输出张量
        out = {
            "seq_ids": torch.from_numpy(seq.astype(np.int64)),
            "pos_ids": torch.from_numpy(pos.astype(np.int64)),
            "neg_ids": torch.from_numpy(neg.astype(np.int64)),
            "token_type": torch.from_numpy(token_type.astype(np.int64)),
            "next_token_type": torch.from_numpy(next_token_type.astype(np.int64)),
            "next_action_type": torch.from_numpy(next_action_type.astype(np.int64)),
            "rel_pos_ids": torch.from_numpy(pos_ids.astype(np.int64)),  # 位置ID用于相对位置偏置
            "rel_time_ids": torch.from_numpy(time_ids.astype(np.int64)),  # 时间ID用于相对时间偏置
        }

        if seq_item_sparse is not None:
            out["item_sparse"] = torch.from_numpy(seq_item_sparse)
        if seq_user_sparse is not None:
            out["user_sparse"] = torch.from_numpy(seq_user_sparse)

        # 多模态（当前/pos/neg）
        out["mm"] = {fid: torch.from_numpy(seq_mm[fid]) for fid in self.mm_feat_keys}
        out["mm_pos"] = {fid: torch.from_numpy(pos_mm[fid]) for fid in self.mm_feat_keys}
        out["mm_neg"] = {fid: torch.from_numpy(neg_mm[fid]) for fid in self.mm_feat_keys}

        if pos_item_sparse is not None:
            out["pos_item_sparse"] = torch.from_numpy(pos_item_sparse)
        if neg_item_sparse is not None:
            out["neg_item_sparse"] = torch.from_numpy(neg_item_sparse)

        # 时间特征输出（若启用）
        if seq_time_sparse is not None:
            out["time_sparse"] = torch.from_numpy(seq_time_sparse)
        if seq_time_cont is not None:
            out["time_continual"] = torch.from_numpy(seq_time_cont)

        # 若指定了 array_max_lens，可在此补充 array/continual 的打包（目前默认不输出，保持最小可用）

        # 追加 rank 包键（如启用 rankloss 时使用）
        out["rank_ids"] = torch.from_numpy(rank_ids)
        if rank_item_sparse is not None:
            out["rank_item_sparse"] = torch.from_numpy(rank_item_sparse)

        # 输出 CTR continual 张量
        if seq_item_cont is not None:
            out["item_cont"] = torch.from_numpy(seq_item_cont)
        if seq_user_cont is not None:
            out["user_cont"] = torch.from_numpy(seq_user_cont)
        if pos_item_cont is not None:
            out["pos_item_cont"] = torch.from_numpy(pos_item_cont)
        if neg_item_cont is not None:
            out["neg_item_cont"] = torch.from_numpy(neg_item_cont)
        if rank_item_cont is not None:
            out["rank_item_cont"] = torch.from_numpy(rank_item_cont)

        return out


def collate_fn_packed(batch):
    """仅做堆叠，避免主进程做 list->tensor 的重活"""
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
        "pos_item_sparse",
        "neg_item_sparse",
        "rank_ids",
        "rank_item_sparse",
        "item_cont",
        "user_cont",
        "pos_item_cont",
        "neg_item_cont",
        "rank_item_cont",
    ]
    out = {}
    for k in keys_to_stack:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    # mm family
    for mmk in ["mm", "mm_pos", "mm_neg"]:
        if mmk in batch[0]:
            fids = list(batch[0][mmk].keys())
            out[mmk] = {fid: torch.stack([b[mmk][fid] for b in batch], dim=0) for fid in fids}

    return out


class ItemDatasetPacked(torch.utils.data.Dataset):
    """离线生成 item embedding 的数据集，只读 item 侧特征并张量化"""

    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.mm_emb_ids = args.mm_emb_id
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        # mm dict（与训练集一致）
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)

        # 参考 MyDataset 的 feat_types 初始化（仅用到 item_sparse/item_emb）
        # 这里简化：直接重用 MyDataset 的静态定义
        # 由于 keys 顺序在 model 中也一致，保持原有列表顺序
        self.item_sparse_keys = [
            '100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116'
        ]

        # CTR (optional for offline item encoding)
        self.use_ctr = bool(getattr(args, 'use_ctr', False)) or bool(getattr(args, 'use_itemctr', False))
        self.I_CTR_ID = '151'
        self.item_ctr = None
        self.ctr_meta = None
        if self.use_ctr:
            ctr_dir = getattr(args, 'ctr_dir', None)
            if not ctr_dir or not os.path.isdir(ctr_dir):
                base = os.environ.get('USER_CACHE_PATH', str(self.data_dir))
                ctr_dir = os.path.join(base, 'ctr')
            try:
                with open(os.path.join(ctr_dir, 'ctr_meta.json'), 'r', encoding='utf-8') as f:
                    self.ctr_meta = json.load(f)
                with open(os.path.join(ctr_dir, 'item_ctr.json'), 'r', encoding='utf-8') as f:
                    tmp = json.load(f)
                self.item_ctr = {int(k): float(v) for k, v in tmp.items()}
                self.eps = float(self.ctr_meta['transform']['eps'])
                self.logit = bool(self.ctr_meta['transform']['logit'])
                self.i_mean = float(self.ctr_meta['item']['mean'])
                self.i_std = float(self.ctr_meta['item']['std'])
                a_i = float(self.ctr_meta['item']['alpha'])
                b_i = float(self.ctr_meta['item']['beta'])
                self.i_prior = a_i / max(a_i + b_i, 1e-12)
            except Exception as e:
                print(f"[CTR][ItemDatasetPacked] failed to load CTR: {e}")
                self.use_ctr = False

    def _ctr_transform(self, x, mean, std, eps=1e-5, logit=False):
        import math
        x = min(max(float(x), eps), 1.0 - eps)
        if logit:
            x = math.log(x / (1.0 - x))
        denom = std if std and std > 0 else 1.0
        return float((x - mean) / denom)

    def __len__(self):
        return self.itemnum

    def __getitem__(self, idx):
        # idx 从 0 开始，对应的 item_id 从 1 开始
        item_id = idx + 1
        feat = self.item_feat_dict.get(str(item_id), {})
        # 稀疏
        S_i = len(self.item_sparse_keys)
        item_sparse = np.zeros([S_i], dtype=np.int64) if S_i > 0 else None
        if item_sparse is not None:
            for j, k in enumerate(self.item_sparse_keys):
                item_sparse[j] = int(feat.get(k, 0))
        # 多模态
        mm = {}
        for fid in self.mm_emb_ids:
            vec = None
            # feat 中若不存在，则尝试从 mm_emb_dict 映射
            if fid in feat and isinstance(feat[fid], np.ndarray):
                vec = feat[fid]
            else:
                # 旧版 mm_emb_dict 以匿名 creative_id 索引；离线时可能不可用，保持 0 向量
                pass
            D = _SHAPE_DICT_MM[fid]
            if isinstance(vec, np.ndarray) and vec.shape[-1] == D:
                mm[fid] = torch.from_numpy(vec.astype(np.float32, copy=False))
            else:
                mm[fid] = torch.zeros(D, dtype=torch.float32)

        out = {
            "item_id": torch.tensor(item_id, dtype=torch.long),
        }
        if item_sparse is not None:
            out["item_sparse"] = torch.from_numpy(item_sparse)
        out["mm"] = mm
        if self.use_ctr and self.item_ctr is not None:
            try:
                raw = self.item_ctr.get(int(item_id))
                if raw is None:
                    val = 0.0
                else:
                    val = self._ctr_transform(raw, self.i_mean, self.i_std, eps=getattr(self, 'eps', 1e-5), logit=getattr(self, 'logit', False))
            except Exception:
                val = 0.0
            out["item_cont"] = torch.tensor([val], dtype=torch.float32)
        return out


class PredictSeqDatasetPacked(UserSeqDatasetPacked):
    """推理数据集（Packed），与训练保持相同特征路径，禁用Hard Negatives。"""

    def __init__(self, data_dir, args, array_max_lens=None):
        # 禁用 Hard Negative
        args.use_hard_negative = False
        super().__init__(data_dir, args, array_max_lens=array_max_lens)

    def _load_data_and_offsets(self):
        # 预测文件同样按 worker 懒打开
        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        self.data_file = None
        self._data_file_pid = None
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def __len__(self):
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)
