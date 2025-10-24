# X, Y: (n, d), (m, d) — ví dụ d=64
import numpy as np
from dataclasses import dataclass
import numpy as np
from bs.normalize import *

# ============================== Utils ==============================
def _norm_rows(a):
    """Euclidean norm dọc theo trục đặc trưng (cuối). a[..., D] -> norm(..., )."""
    return np.linalg.norm(a, axis=-1)

def _eps_backbone_nd(X, eps0=0.0, eps1=1.0, p=1):
    """
    X: (n, d). eps[t] = eps0 + eps1 * ||X[t+1]-X[t]||^p, t=0..n-2
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X phải có shape (n, d).")
    n = X.shape[0]
    if n <= 1:
        return np.empty(0, dtype=float)
    dif = _norm_rows(X[1:] - X[:-1])
    if p == 2:
        dif = dif * dif
    return eps0 + eps1 * dif

def _phi_j_to_ix_vectorized(n, m):
    """Ánh xạ j -> chỉ số neo khởi đầu phi_j (linspace theo chỉ số thời gian)."""
    if m <= 1:
        return np.zeros(m, dtype=int)
    return np.rint(np.linspace(0, n - 1, m)).astype(int)

# ============================ RMQ (Range-Max) ============================
def _rmq_build_max(arr):
    """
    Sparse Table cho max-range trên 1D arr (len=n-1).
    Trả (st, lg) để query max nhanh.
    """
    arr = np.asarray(arr, dtype=float).ravel()
    n = arr.size
    if n == 0:
        return np.empty((1, 0), dtype=float), np.zeros(1, dtype=int)
    K = int(np.floor(np.log2(n))) + 1
    st = np.empty((K, n), dtype=arr.dtype)
    st[0, :] = arr
    j = 1
    while (1 << j) <= n:
        span = 1 << (j - 1)
        upto = n - (1 << j) + 1
        st[j, :upto] = np.maximum(st[j - 1, :upto], st[j - 1, span: span + upto])
        j += 1
    lg = np.zeros(n + 1, dtype=int)
    for i in range(2, n + 1):
        lg[i] = lg[i // 2] + 1
    return st, lg

def _rmq_query_max_batch(st, lg, Ls, Rs):
    """
    Vector hoá: trả max(arr[L..R]) cho từng cặp (L,R). Nếu L>R -> 0.
    """
    Ls = np.asarray(Ls, dtype=int); Rs = np.asarray(Rs, dtype=int)
    out = np.zeros_like(Ls, dtype=float)
    mask = (Ls <= Rs)
    if not mask.any():
        return out
    lens = Rs[mask] - Ls[mask] + 1
    k = lg[lens]
    left = st[k, Ls[mask]]
    right = st[k, Rs[mask] - (1 << k) + 1]
    out[mask] = np.maximum(left, right)
    return out

# ====================== Prefix cho q-norm của epsilon ====================
def _build_prefix_pow_eps(eps, q):
    """
    Prefix-sum cho eps^q (q>0, hữu hạn).
    pref[0]=0, pref[k]=sum_{t<k} eps[t]^q
    """
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q phải hữu hạn và > 0 (dùng q=np.inf cho max).")
    eps = np.asarray(eps, float).ravel()
    pow_eps = np.power(np.maximum(eps, 0.0), q)
    pref = np.concatenate([[0.0], np.cumsum(pow_eps)])
    return pref

def _range_pnorm_from_prefix(pref, Ls, Rs, q):
    """
    Tính ((sum_{t=L..R} eps[t]^q))^(1/q) theo prefix (Ls,Rs vector).
    Nếu L>R -> 0.
    """
    Ls = np.asarray(Ls, int); Rs = np.asarray(Rs, int)
    out = np.zeros_like(Ls, dtype=float)
    mask = (Ls <= Rs)
    if not mask.any():
        return out
    Lc = np.clip(Ls[mask], 0, pref.size - 2)
    Rc = np.clip(Rs[mask], 0, pref.size - 2)
    sum_pow = pref[Rc + 1] - pref[Lc]
    out[mask] = np.power(np.maximum(sum_pow, 0.0), 1.0 / float(q))
    return out

# ========================= Backbone Tree (Union-Find) ====================
@dataclass
class _Tree:
    parent: np.ndarray   # (#nodes,)
    w: np.ndarray        # cạnh node -> parent
    children: list       # list[list]
    leaf_x: np.ndarray   # node-id của lá x[i]
    height: np.ndarray

def _build_backbone_tree_from_X(X, lam=1.0, eps0=0.0, eps1=1.0, p=1):
    """
    X: (n, d). Xây "cột sống" theo Union-Find trên các cạnh (t, t+1),
    độ cao H = 0.5 * lam * eps[t], nối theo eps giảm dần.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X phải có shape (n, d).")
    n = X.shape[0]
    if n == 0:
        raise ValueError("Chuỗi X rỗng.")
    if n == 1:
        parent = np.array([-1, 0], dtype=int)
        w = np.array([0.0, 0.0], dtype=float)
        children = [[1], []]
        leaf_x = np.array([1], dtype=int)
        height = np.array([0.0, 0.0], dtype=float)
        return _Tree(parent, w, children, leaf_x, height)

    eps = _eps_backbone_nd(X, eps0=eps0, eps1=eps1, p=p)  # (n-1,)
    parent = [-1] * n
    w = [0.0] * n
    children = [[] for _ in range(n)]
    height = [0.0] * n
    leaf_x = np.arange(n, dtype=int)

    uf_parent = list(range(n))
    comp_root_node = list(range(n))
    comp_height = [0.0] * n

    def uf_find(a):
        while uf_parent[a] != a:
            uf_parent[a] = uf_parent[uf_parent[a]]
            a = uf_parent[a]
        return a

    def uf_union(a, b, H):
        ra, rb = uf_find(a), uf_find(b)
        if ra == rb:
            return ra
        new_id = len(parent)
        parent.append(-1)
        w.append(0.0)
        children.append([])
        height.append(H)
        na = comp_root_node[ra]
        nb = comp_root_node[rb]
        parent[na] = new_id
        parent[nb] = new_id
        w[na] = max(0.0, H - height[na])
        w[nb] = max(0.0, H - height[nb])
        children[new_id].append(na)
        children[new_id].append(nb)
        uf_parent[rb] = ra
        comp_root_node[ra] = new_id
        comp_height[ra] = H
        return ra

    order = np.argsort(eps) # tăng dần
    for t in order:
        H = 0.5 * lam * float(eps[t])
        uf_union(t, t + 1, H)

    r = uf_find(0)
    root_node = comp_root_node[r]
    parent[root_node] = -1

    return _Tree(
        parent=np.array(parent, dtype=int),
        w=np.array(w, dtype=float),
        children=children,
        leaf_x=leaf_x,
        height=np.array(height, dtype=float),
    )

# ========================== Anchor selection (MAP/MC) ====================
def _choose_anchors_MAP_nd(X, Y, eps, lam=1.0, alpha=1.0, w_band=10, q_U=np.inf):
    """
    MAP trên dải [phi-w, phi+w] với bottleneck U(i|phi):
      - q_U = np.inf -> U = lam * max eps (RMQ)
      - q_U hữu hạn > 0 -> U = lam * (sum eps^q)^(1/q)
    cost(i,j) = alpha*||X[i]-Y[j]|| + U(i|phi_j).
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, d = X.shape; m = Y.shape[0]
    phi = _phi_j_to_ix_vectorized(n, m)  # (m,)
    W = int(2 * w_band + 1)

    base = np.arange(-w_band, w_band + 1, dtype=int)[None, :] + phi[:, None]  # (m,W)
    idx = np.clip(base, 0, n - 1)
    valid = (base >= 0) & (base <= (n - 1))

    l = np.minimum(idx, phi[:, None])
    r = np.maximum(idx, phi[:, None]) - 1

    if np.isinf(q_U):
        st, lg = _rmq_build_max(eps)
        U = lam * _rmq_query_max_batch(st, lg, l.ravel(), r.ravel()).reshape(m, W)
    else:
        pref = _build_prefix_pow_eps(eps, q_U)
        U = lam * _range_pnorm_from_prefix(pref, l.ravel(), r.ravel(), q_U).reshape(m, W)

    Xi = X[idx]                                 # (m,W,d)
    dist = np.linalg.norm(Xi - Y[:, None, :], axis=2)  # (m,W)

    cost = alpha * dist + U
    cost = np.where(valid, cost, np.inf)
    k_min = np.argmin(cost, axis=1)
    anchors = idx[np.arange(m), k_min]
    return anchors.astype(int)

def _choose_anchors_MC_nd(
    X, Y, eps, lam=1.0, alpha=1.0, w_band=10, T=0.25, rng=None, q_U=np.inf,
    share_kernel: bool = False,
):
    """
    MC sampling theo softmax(-cost/T) bằng 1 lần "kernel" (Uniform[0,1)).
    - Nếu share_kernel=True: dùng một u chung cho mọi j trong lần gọi này.
    - Nếu share_kernel=False (mặc định): mỗi j dùng một u riêng (độc lập).
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, d = X.shape; m = Y.shape[0]
    phi = _phi_j_to_ix_vectorized(n, m)
    W = int(2 * w_band + 1)

    base = np.arange(-w_band, w_band + 1, dtype=int)[None, :] + phi[:, None]  # (m, W)
    idx = np.clip(base, 0, n - 1)
    valid = (base >= 0) & (base <= (n - 1))                                   # (m, W)

    # biên trái/phải để tính bottleneck U(i | phi_j)
    l = np.minimum(idx, phi[:, None])
    r = np.maximum(idx, phi[:, None]) - 1

    if np.isinf(q_U):
        st, lg = _rmq_build_max(eps)
        U = lam * _rmq_query_max_batch(st, lg, l.ravel(), r.ravel()).reshape(m, W)
    else:
        pref = _build_prefix_pow_eps(eps, q_U)
        U = lam * _range_pnorm_from_prefix(pref, l.ravel(), r.ravel(), q_U).reshape(m, W)

    # khoảng cách dữ liệu
    Xi = X[idx]  # (m, W, d)
    dist = np.linalg.norm(Xi - Y[:, None, :], axis=2)  # (m, W)

    # cost và logits
    cost = alpha * dist + U
    logits = -cost / float(T)

    # mask các vị trí invalid
    logits = np.where(valid, logits, -np.inf)

    # softmax theo từng hàng (ổn định số)
    z = logits - np.nanmax(logits, axis=1, keepdims=True)   # max có thể là -inf nếu hàng toàn invalid
    expz = np.exp(z, where=np.isfinite(z), out=np.zeros_like(z))
    expz *= valid
    sums = expz.sum(axis=1, keepdims=True)

    # fallback nếu một hàng không có mass hợp lệ (sums == 0): rơi về MAP trên các vị trí hợp lệ
    best_k_map = np.argmin(np.where(valid, cost, np.inf), axis=1)  # (m,)

    # chuẩn hóa xác suất
    probs = np.divide(expz, sums, out=np.zeros_like(expz), where=(sums > 0))

    # CDF và “thả kernel”
    cdf = np.cumsum(probs, axis=1)  # (m, W), mỗi hàng kết thúc ở 1.0

    if share_kernel:
        u = float(rng.random())
        # dùng cùng u cho mọi hàng
        k = np.sum(cdf < u, axis=1)
    else:
        u = rng.random(m)  # (m,)
        k = np.sum(cdf < u[:, None], axis=1)

    # nếu hàng đó sums==0 (không có xác suất hợp lệ) thì dùng k_MAP
    k = np.where(sums.ravel() > 0, k, best_k_map)

    anchors = idx[np.arange(m), k]
    return anchors.astype(int)

# ========================= Soft anchor probabilities =========================
def _soft_anchor_probs_nd(
    X, Y, eps, lam=1.0, alpha=1.0, w_band=10, T=0.25, q_U=np.inf, eps_y0=0.0, p=1
):
    """
    Tính p(i|j) = softmax(-cost/T) trong băng quanh phi_j và trả về:
      - q: (n,) với q_i = sum_j b_j * p(i|j), b_j = 1/m (Y đều)
      - term_y: kỳ vọng [(alpha*||X[i]-Y[j]|| + eps_y0)^p] theo p(i|j) và b_j
    cost(i,j) = alpha*||X[i]-Y[j]|| + lam * U_q(i | phi_j)
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, d = X.shape; m = Y.shape[0]
    assert m >= 1 and n >= 1
    phi = _phi_j_to_ix_vectorized(n, m)
    W = int(2 * w_band + 1)

    base = np.arange(-w_band, w_band + 1, dtype=int)[None, :] + phi[:, None]  # (m, W)
    idx = np.clip(base, 0, n - 1)
    valid = (base >= 0) & (base <= (n - 1))

    # biên để tính bottleneck U(i | phi_j)
    l = np.minimum(idx, phi[:, None])
    r = np.maximum(idx, phi[:, None]) - 1

    if np.isinf(q_U):
        st, lg = _rmq_build_max(eps)
        U = lam * _rmq_query_max_batch(st, lg, l.ravel(), r.ravel()).reshape(m, W)
    else:
        pref = _build_prefix_pow_eps(eps, q_U)
        U = lam * _range_pnorm_from_prefix(pref, l.ravel(), r.ravel(), q_U).reshape(m, W)

    Xi = X[idx]  # (m, W, d)
    dist = np.linalg.norm(Xi - Y[:, None, :], axis=2)  # (m, W)

    cost = alpha * dist + U
    logits = -cost / float(T)
    logits = np.where(valid, logits, -np.inf)

    z = logits - np.nanmax(logits, axis=1, keepdims=True)
    expz = np.exp(z, where=np.isfinite(z), out=np.zeros_like(z))
    expz *= valid
    sums = expz.sum(axis=1, keepdims=True)

    # fallback: hàng không hợp lệ -> one-hot tại MAP trong băng
    best_k_map = np.argmin(np.where(valid, cost, np.inf), axis=1)
    probs = np.divide(expz, sums, out=np.zeros_like(expz), where=(sums > 0))
    empty = (sums.ravel() == 0)
    if np.any(empty):
        probs[empty, :] = 0.0
        probs[empty, best_k_map[empty]] = 1.0

    # q_i = (1/m) * sum_j p(i|j)
    q = np.zeros(n, dtype=float)
    weights = (1.0 / float(m)) * probs  # (m, W)
    np.add.at(q, idx, weights)

    # term_y = E_{j~b, i~p(.|j)} [ (alpha*||X[i]-Y[j]|| + eps_y0)^p ]
    pw = float(p)
    term_y = np.sum(weights * (alpha * dist + float(eps_y0)) ** pw, dtype=float)

    return q, term_y


# ====== Tree part (no-attach) từ q: W_p^p(tree) = sum_e (w[e]^p) * |mass_e| ======
def _wp_pow_tree_from_q(tree_x: _Tree, a, q, p=1):
    """
    Đồng bộ với định nghĩa hiện tại trong _wp_pow_no_attach_nd:
    dùng w[v]**p * |mass[v]| (không đổi công thức cũ để nhất quán MAP/MC).
    mass_leaf = a - q, dồn bottom-up.
    """
    pw = float(p)
    parent = tree_x.parent
    w = tree_x.w
    n_nodes = parent.size

    mass = np.zeros(n_nodes, dtype=float)
    mass[tree_x.leaf_x] = a - q  # (n,)

    total = 0.0
    for v in range(n_nodes - 1, 0, -1):
        pv = parent[v]
        if pv >= 0:
            total += (w[v] ** pw) * abs(mass[v])
            mass[pv] += mass[v]
    return float(total)

def _wp_pow_tree_from_q(tree_x: _Tree, a, q, p=1):
    """
    Đồng bộ với định nghĩa hiện tại trong _wp_pow_no_attach_nd:
    dùng w[v]**p * |mass[v]| (không đổi công thức cũ để nhất quán MAP/MC).
    mass_leaf = a - q, dồn bottom-up.
    """
    pw = float(p)
    parent = tree_x.parent
    w = tree_x.w
    n_nodes = parent.size

    mass = np.zeros(n_nodes, dtype=float)
    mass[tree_x.leaf_x] = a - q  # (n,)

    total = 0.0
    for v in range(n_nodes - 1, 0, -1):
        pv = parent[v]
        if pv >= 0:
            total += (w[v] ** pw) * abs(mass[v])
            mass[pv] += mass[v]
    return float(total)
# ========================= OT on Tree: no-attach =========================
def _wp_pow_no_attach_nd(tree_x: _Tree, anchors, X, Y, a, b, alpha=1.0, eps_y0=0.0, p=1):
    """
    W_p^p nhanh không gắn lá Y.
      term_Y = sum_j (alpha*||X[anc[j]]-Y[j]|| + eps_y0)^p * b_j
      mass tại lá X: a_i - sum_{j neo vào i} b_j, rồi bottom-up 1 pass.
    """
    pw = float(p)

    dif = np.linalg.norm(X[anchors] - Y, axis=1)  # (m,)
    term_y = np.sum((alpha * dif + float(eps_y0)) ** pw * b, dtype=float)

    n = tree_x.leaf_x.size
    sum_b_per_leaf = np.zeros(n, dtype=float)
    np.add.at(sum_b_per_leaf, anchors, b)

    parent = tree_x.parent
    w = tree_x.w
    n_nodes = parent.size
    mass = np.zeros(n_nodes, dtype=float)
    mass[tree_x.leaf_x] = a - sum_b_per_leaf

    total = 0.0
    for v in range(n_nodes - 1, 0, -1):
        pv = parent[v]
        if pv >= 0:
            total += (w[v] ** pw) * abs(mass[v])
            mass[pv] += mass[v]
    return float(total + term_y)

# ============================== Public API ===============================
def bbs_tree_ot_distance_nd(
    X, Y,                     # (n,d), (m,d)
    num_samples=16,           # cho MC
    alpha=1.0,
    lam=5.0,
    eps0=0.0, eps1=1.0, p_eps=1,
    anchor_mode="MC",         # "MC", "MAP", hoặc "SOFT"
    w_band=10,
    T_mc=0.25,
    eps_y0=0.0,
    p_ot=2,
    random_state=None,
    reducer="mean",           # "mean" hoặc "median"
    return_details=False,
    q_U=np.inf,               # q-norm cho U (np.inf => max)
):
    """
    Khoảng cách BBS giữa 2 chuỗi đa chiều X,Y.
    Trả về W_p (không phải W_p^p).
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X, Y phải có shape (n,d) và (m,d).")
    n, d = X.shape
    m, d2 = Y.shape
    if d != d2:
        raise ValueError("Số chiều đặc trưng của X và Y phải bằng nhau.")
    if n == 0 or m == 0:
        raise ValueError("Chuỗi rỗng.")
    if p_ot < 1:
        raise ValueError("p_ot phải ≥ 1.")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Dữ liệu chứa NaN/Inf.")

    # Backbone & RMQ cho X
    eps = _eps_backbone_nd(X, eps0=eps0, eps1=eps1, p=p_eps)  # (n-1,)
    tree_x = _build_backbone_tree_from_X(X, lam=1.0, eps0=eps0, eps1=eps1, p=p_eps)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)

    def _calc_once(local_rng):
        mode = anchor_mode.upper()
        if mode == "MAP":
            anchors = _choose_anchors_MAP_nd(X, Y, eps, lam=lam, alpha=alpha, w_band=w_band, q_U=q_U)
            w_p_p = _wp_pow_no_attach_nd(tree_x, anchors, X, Y, a, b,
                                         alpha=alpha, eps_y0=eps_y0, p=p_ot)
            return (w_p_p ** (1.0 / p_ot)) if p_ot != 1 else w_p_p, anchors

        elif mode == "MC":
            anchors = _choose_anchors_MC_nd(X, Y, eps, lam=lam, alpha=alpha, w_band=w_band,
                                            T=T_mc, rng=local_rng, q_U=q_U)
            w_p_p = _wp_pow_no_attach_nd(tree_x, anchors, X, Y, a, b,
                                         alpha=alpha, eps_y0=eps_y0, p=p_ot)
            return (w_p_p ** (1.0 / p_ot)) if p_ot != 1 else w_p_p, anchors

        elif mode == "SOFT":
            # Neo mềm: p(i|j) -> q_i, cộng với kỳ vọng term_y; phần cây lấy từ q
            q, term_y = _soft_anchor_probs_nd(
                X, Y, eps, lam=lam, alpha=alpha, w_band=w_band,
                T=T_mc, q_U=q_U, eps_y0=eps_y0, p=p_ot
            )
            tree_part = _wp_pow_tree_from_q(tree_x, a, q, p=p_ot)
            w_p_p = term_y + tree_part
            return (w_p_p ** (1.0 / p_ot)) if p_ot != 1 else w_p_p, None

        else:
            raise ValueError("anchor_mode phải là 'MAP', 'MC' hoặc 'SOFT'.")

    mode = anchor_mode.upper()
    if mode in ("MAP", "SOFT"):
        val, anchors = _calc_once(rng)
        if not return_details:
            return float(val)
        return float(val), {
            "all_distances": [float(val)],
            "anchors_list": [anchors],
            "params": dict(num_samples=1, alpha=alpha, lam=lam, eps0=eps0, eps1=eps1, p_eps=p_eps,
                           anchor_mode=anchor_mode, w_band=w_band, T_mc=T_mc, eps_y0=eps_y0, p_ot=p_ot,
                           reducer=reducer, q_U=q_U)
        }

    # MC: lặp num_samples
    vals = []
    anchors_list = [] if return_details else None
    for _ in range(int(num_samples)):
        rs = int(rng.integers(1e9))
        v, anchors = _calc_once(np.random.default_rng(rs))
        vals.append(float(v))
        if anchors_list is not None:
            anchors_list.append(anchors)

    val = float(np.median(vals) if reducer == "median" else np.mean(vals))
    if not return_details:
        return val
    return val, {
        "all_distances": vals,
        "anchors_list": anchors_list,
        "params": dict(num_samples=num_samples, alpha=alpha, lam=lam, eps0=eps0, eps1=eps1, p_eps=p_eps,
                       anchor_mode=anchor_mode, w_band=w_band, T_mc=T_mc, eps_y0=eps_y0, p_ot=p_ot,
                       reducer=reducer, q_U=q_U)
    }

# ============================== Arclength utils ===============================
def _resample_by_arclength(X, n_new=None):
    """
    Trả về X' được lấy mẫu đều theo độ dài cung (arclength).
    - Nếu n_new None -> giữ nguyên số điểm ban đầu.
    - Ổn với dữ liệu đa chiều (n, d).
    """
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X phải có shape (n, d).")
    n, d = X.shape
    if n <= 1:
        return X.copy()

    if n_new is None:
        n_new = n

    seg = np.linalg.norm(X[1:] - X[:-1], axis=1)       # (n-1,)
    s = np.concatenate([[0.0], np.cumsum(seg)])        # (n,)
    L = s[-1]
    if L <= 0:
        # Chuỗi đứng yên -> trả bản sao
        return X.copy()

    # Lưới tham số đều theo chiều dài (n_new điểm)
    t = np.linspace(0.0, L, n_new)

    # Nội suy từng chiều theo s
    out = np.empty((n_new, d), float)
    for k in range(d):
        out[:, k] = np.interp(t, s, X[:, k])
    return out

def bs_distance_between_series(
    X, Y,
    num_trees=16,
    alpha=10.0,
    lam=10.0,
    eps0=1.0, eps1=1.0, p_eps=1,
    anchor_mode="MC",
    T_mc=5.0,
    eps_y0=0.0,
    p_ot=2,
    random_state=None,
    reducer="mean",
    return_details=False,
    q_U=np.inf,   # <--- NEW
):
    """
    Wrapper cho bbs_tree_ot_distance_nd để tương thích với tên hàm cũ.
    """
    if X.shape[1] < 10 or Y.shape[1] < 10:
        X, Y, norm_stats = normalize_pair(X, Y, mode="whiten")  # chuẩn hoá trước khi tính BBS
        
    return bbs_tree_ot_distance_nd(
        X, Y,
        num_samples=num_trees,
        alpha=alpha,
        lam=lam,
        eps0=eps0, eps1=eps1, p_eps=p_eps,
        anchor_mode=anchor_mode,
        w_band=15,
        T_mc=T_mc,
        eps_y0=eps_y0,
        p_ot=p_ot,
        random_state=random_state,
        reducer=reducer,
        return_details=return_details,
        q_U=q_U,   # truyền q-norm xuống
    )
