import argparse
import json
import math
import os
from collections import defaultdict


def scan_counts(seq_path):
    U_expo = defaultdict(int)
    U_click = defaultdict(int)
    I_expo = defaultdict(int)
    I_click = defaultdict(int)
    total_expo = 0
    total_click = 0

    with open(seq_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                arr = json.loads(line)
            except Exception:
                continue
            for rec in arr:
                # Expect: [user_id, item_id, user_feat, item_feat, action_type, timestamp]
                if len(rec) < 6:
                    # fallback older format
                    u, i, _, _, act = rec + [None] * (5 - len(rec))
                else:
                    u, i, _, _, act, _ = rec[:6]
                if i is None:
                    continue
                if act is None:
                    continue
                total_expo += 1
                U_expo[u] += 1
                I_expo[i] += 1
                if int(act) == 1:
                    total_click += 1
                    U_click[u] += 1
                    I_click[i] += 1
    return U_expo, U_click, I_expo, I_click, total_expo, total_click


def method_of_moments_k(expo_dict, click_dict, k_clip=(10.0, 10000.0)):
    expos = []
    phats = []
    total_expo = 0
    total_click = 0
    for e in expo_dict.values():
        total_expo += e
    for key, e in expo_dict.items():
        c = click_dict.get(key, 0)
        phat = c / max(e, 1)
        phats.append(phat)
        expos.append(e)
        total_click += c
    m = total_click / max(total_expo, 1)

    mean_phat = sum(phats) / max(len(phats), 1)
    var_raw = sum((p - mean_phat) ** 2 for p in phats) / max(len(phats), 1)
    noise = sum(m * (1 - m) / max(e, 1) for e in expos) / max(len(expos), 1)
    v_between = max(var_raw - noise, 1e-12)
    # Beta-Bernoulli: Var(p) = m(1-m)/(k+1) => k = m(1-m)/v_between - 1
    k = m * (1 - m) / v_between - 1.0
    k = float(max(k_clip[0], min(k, k_clip[1])))
    return m, k


def smooth(pairs, alpha, beta):
    out = {}
    for _id, (c, e) in pairs.items():
        out[str(_id)] = (c + alpha) / (e + alpha + beta)
    return out


def standardize(values, eps=1e-12):
    arr = list(values)
    if not arr:
        return 0.0, 1.0
    mean = sum(arr) / len(arr)
    var = sum((x - mean) ** 2 for x in arr) / max(len(arr), 1)
    std = math.sqrt(max(var, eps))
    return float(mean), float(std)


def maybe_logit(x, eps=1e-5):
    x = min(max(x, eps), 1.0 - eps)
    return math.log(x / (1.0 - x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--prior', type=str, default='auto')  # 'auto' or 'a,b'
    p.add_argument('--k_clip', type=str, default='10,10000')
    p.add_argument('--eps', type=float, default=1e-5)
    p.add_argument('--logit', action='store_true')
    args = p.parse_args()

    if args.output_dir is None:
        base = os.environ.get('USER_CACHE_PATH', os.path.dirname(args.seq_path))
        args.output_dir = os.path.join(base, 'ctr')
    k_clip = tuple(map(float, args.k_clip.split(',')))

    U_expo, U_click, I_expo, I_click, total_expo, total_click = scan_counts(args.seq_path)
    global_m = total_click / max(total_expo, 1)

    m_u, k_u = method_of_moments_k(U_expo, U_click, k_clip=k_clip)
    m_i, k_i = method_of_moments_k(I_expo, I_click, k_clip=k_clip)

    if args.prior != 'auto':
        a_str, b_str = args.prior.split(',')
        a_u = a_i = float(a_str)
        b_u = b_i = float(b_str)
    else:
        a_u = m_u * k_u
        b_u = (1 - m_u) * k_u
        a_i = m_i * k_i
        b_i = (1 - m_i) * k_i

    U_pairs = {u: (U_click.get(u, 0), e) for u, e in U_expo.items()}
    I_pairs = {i: (I_click.get(i, 0), e) for i, e in I_expo.items()}

    user_ctr = smooth(U_pairs, a_u, b_u)
    item_ctr = smooth(I_pairs, a_i, b_i)

    def transform(dic):
        vals = [min(max(v, args.eps), 1.0 - args.eps) for v in dic.values()]
        if args.logit:
            vals = [maybe_logit(v, eps=args.eps) for v in vals]
        mean, std = standardize(vals)
        return mean, std

    mean_u, std_u = transform(user_ctr)
    mean_i, std_i = transform(item_ctr)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'user_ctr.json'), 'w', encoding='utf-8') as f:
        json.dump(user_ctr, f)
    with open(os.path.join(args.output_dir, 'item_ctr.json'), 'w', encoding='utf-8') as f:
        json.dump(item_ctr, f)

    meta = {
        'global_ctr': global_m,
        'user': {'alpha': a_u, 'beta': b_u, 'k': k_u, 'mean': mean_u, 'std': std_u},
        'item': {'alpha': a_i, 'beta': b_i, 'k': k_i, 'mean': mean_i, 'std': std_i},
        'transform': {'eps': args.eps, 'logit': bool(args.logit)},
    }
    with open(os.path.join(args.output_dir, 'ctr_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"CTR artifacts saved to: {args.output_dir}")


if __name__ == '__main__':
    main()


