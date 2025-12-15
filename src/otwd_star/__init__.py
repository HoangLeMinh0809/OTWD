# ctwd_api.py  — RAGGED-FRIENDLY
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Sequence, Union
import heapq

# (tuỳ chọn) dùng SciPy để tăng tốc SpMM; nếu không có vẫn chạy được
try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

BIG = 1e12

# =========================
# 0) HELPERS (ragged / dense)
# =========================
def _as_ragged_list(M: Union[np.ndarray, Sequence[np.ndarray]]) -> Tuple[List[np.ndarray], int]:
    """
    Chuẩn hoá đầu vào về list các mảng (n_i, d).
    Trả về (list_seq, d)
    """
    if isinstance(M, np.ndarray):
        if M.ndim != 3:
            raise ValueError("If ndarray, expect shape (m, n, d).")
        m, n, d = M.shape
        seqs = [M[i] for i in range(m)]
        return seqs, d
    # list/tuple các chuỗi (n_i, d)
    seqs = []
    d = None
    for i, xi in enumerate(M):
        xi = np.asarray(xi, dtype=float)
        if xi.ndim != 2:
            raise ValueError(f"Sequence {i} must have shape (n_i, d).")
        if d is None:
            d = xi.shape[1]
        elif xi.shape[1] != d:
            raise ValueError("All sequences must have the same feature dimension d.")
        seqs.append(xi)
    if d is None:
        raise ValueError("Empty sequence list.")
    return seqs, d

def _linearize_points_ragged(M: Union[np.ndarray, Sequence[np.ndarray]]):
    """
    Hỗ trợ ragged: 
      - P: (N, d) các điểm ghép lại
      - Sidx: (N,) id chuỗi
      - Tpos: (N,) thời gian chuẩn hoá trong [0,1) cho mỗi điểm (i / n_i)
      - lengths: (m_seq,) độ dài từng chuỗi
      - d: số kênh
    """
    seqs, d = _as_ragged_list(M)
    m_seq = len(seqs)
    lengths = np.array([xi.shape[0] for xi in seqs], dtype=int)
    # ghép điểm
    P = np.vstack(seqs) if m_seq > 0 else np.zeros((0, d))
    # id chuỗi
    Sidx = np.repeat(np.arange(m_seq, dtype=int), lengths)
    # vị trí thời gian chuẩn hoá (không dùng endpoint=1 để tránh trùng 1.0)
    Tpos_list = [ (np.arange(n_i, dtype=float) / max(n_i,1)) for n_i in lengths ]
    Tpos = np.concatenate(Tpos_list) if m_seq > 0 else np.zeros((0,), float)
    return P, Sidx, Tpos, m_seq, lengths, d

def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Khoảng cách 'lai':
      Euclid^2 trên các đặc trưng (trừ cột cuối)  +  |orderA - orderB|   (cột cuối)
    """
    assert A.ndim == 2 and B.ndim == 2, "Expect 2D arrays"
    assert A.shape[1] == B.shape[1], "Dim mismatch"
    D = A.shape[1]
    if D == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=float)
    if D == 1:
        a_ord = A[:, 0]; b_ord = B[:, 0]
        return np.abs(a_ord[:, None] - b_ord[None, :])
    Af = A[:, :-1]; Bf = B[:, :-1]
    aa = (Af * Af).sum(1)[:, None]
    bb = (Bf * Bf).sum(1)[None, :]
    Dsq = np.clip(aa + bb - 2 * (Af @ Bf.T), 0.0, None)
    a_ord = A[:, -1]; b_ord = B[:, -1]
    Pen = np.abs(a_ord[:, None] - b_ord[None, :])
    return Dsq + Pen

@dataclass
class _Node:
    idx: np.ndarray
    height: float
    left: Optional[int]
    right: Optional[int]
    parent: Optional[int]
    is_leaf: bool

@dataclass
class CTWDModel:
    # shared runtime fields
    P: np.ndarray                  # (N, d_aug) nếu TamLe, hoặc (N, d) nếu Banded
    Sidx: np.ndarray               # (N,) id chuỗi
    Tpos: np.ndarray               # (N,) thời gian chuẩn hoá (0..1)
    lengths: np.ndarray            # (m_seq,)
    m_seq: int
    d: int                         # số kênh gốc (chưa augment)
    nodes: List[_Node]
    leaf_ids: List[int]
    leaf_index_map: Dict[int, int]
    edges: List[Tuple[int, int, float]]     # (parent, child, w_e)
    S_edge_leaf: object                     # (E, L) dense hoặc sp.csr_matrix
    centroids: np.ndarray                   # (num_nodes, D_aug)
    # meta
    mode: str                               # "tamle" | "banded"
    lam_time: float = 0.0
    lam_idx: float = 0.0
    W: float = 0.0                           # band width (tỉ lệ 0..1 cho ragged)
    # caches
    point_leaf: Optional[np.ndarray] = None  # (N,)
    H: Optional[np.ndarray] = None           # (L, m_seq)
    M: Optional[np.ndarray] = None           # (E, m_seq)
    w: Optional[np.ndarray] = None           # (E,)

# =========================
# 1) BOX TREE + GONZALEZ
# =========================
class _KDBoxTree:
    def __init__(self, leaf_size=64, max_depth=24):
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.boxes = []
        self.X = None

    def _build(self, idx, depth):
        X = self.X[idx]
        c = X.mean(axis=0)
        r = float(np.sqrt(((X - c) ** 2).sum(1).max())) if X.shape[0] else 0.0
        bid = len(self.boxes)
        self.boxes.append({"idx": idx, "c": c, "r": r, "L": None, "R": None, "leaf": False})
        if idx.size <= self.leaf_size or depth >= self.max_depth or r == 0.0:
            self.boxes[bid]["leaf"] = True
            return bid
        var = X.var(axis=0)
        d = int(np.argmax(var))
        med = np.median(X[:, d])
        mask = X[:, d] <= med
        if mask.all() or (~mask).all():
            mid = idx.size // 2
            Lidx = idx[:mid]; Ridx = idx[mid:]
        else:
            Lidx = idx[mask]; Ridx = idx[~mask]
        L = self._build(Lidx, depth + 1)
        R = self._build(Ridx, depth + 1)
        self.boxes[bid]["L"] = L; self.boxes[bid]["R"] = R
        return bid

    def fit(self, X):
        self.X = X
        self.boxes = []
        self._build(np.arange(X.shape[0]), 0)

def _bounds_box(box, centers):
    if centers.size == 0: return 0.0, float("inf")
    d = np.sqrt(((centers - box["c"][None, :]) ** 2).sum(1))
    dmin = float(d.min())
    r = box["r"]
    return max(0.0, dmin - r), dmin + r

def _farthest_point_by_boxes(X, centers, kdt: _KDBoxTree, gap_tol=1e-6):
    if centers.size == 0: return 0, 0.0
    heap = []
    L0, U0 = _bounds_box(kdt.boxes[0], centers)
    heapq.heappush(heap, (-U0, 0, L0))
    best_idx, best_val = None, -1.0
    while heap:
        negU, bid, Lb = heapq.heappop(heap)
        Ub = -negU
        L2, U2 = _bounds_box(kdt.boxes[bid], centers)
        if U2 < Ub - 1e-12 or L2 > Lb + 1e-12:
            heapq.heappush(heap, (-U2, bid, L2)); continue
        if best_val >= U2 - 1e-15: break
        box = kdt.boxes[bid]
        if box["leaf"] or (U2 - L2) <= gap_tol:
            pts = kdt.X[box["idx"]]
            D = _pairwise_sqdist(pts, centers)  # KHÔNG sqrt
            dmin = D.min(axis=1)
            imax = int(np.argmax(dmin)); val = float(dmin[imax])
            if val > best_val: best_val, best_idx = val, int(box["idx"][imax])
            continue
        for child in (box["L"], box["R"]):
            Lc, Uc = _bounds_box(kdt.boxes[child], centers)
            heapq.heappush(heap, (-Uc, child, Lc))
    return best_idx, best_val

def _gonzalez_box_nlogk(X: np.ndarray, k: int, seed: int,
                        box_leaf_size=64, box_max_depth=24, gap_tol=1e-6):
    rng = np.random.default_rng(seed)
    n = X.shape[0]; assert 1 <= k <= n
    kdt = _KDBoxTree(leaf_size=box_leaf_size, max_depth=box_max_depth); kdt.fit(X)
    i0 = int(rng.integers(0, n)); centers = X[i0:i0+1]; C = [i0]
    for _ in range(1, k):
        idx, _ = _farthest_point_by_boxes(X, centers, kdt, gap_tol)
        C.append(idx); centers = X[np.array(C)]
    return np.array(C, dtype=int)

# =========================
# 1.1) ROUTING & PRECOMPUTE
# =========================
def _route_all_points_vectorized(model: CTWDModel) -> np.ndarray:
    N = model.P.shape[0]
    leaf_of_point = np.empty(N, dtype=np.int32)
    stack = [(0, np.arange(N, dtype=np.int32))]
    nodes = model.nodes; C = model.centroids; P = model.P
    while stack:
        nid, idxs = stack.pop()
        nd = nodes[nid]
        if nd.is_leaf:
            j = model.leaf_index_map[nid]; leaf_of_point[idxs] = j; continue
        L = nd.left; R = nd.right
        X = P[idxs]
        dl = np.linalg.norm(X - C[L], axis=1)
        dr = np.linalg.norm(X - C[R], axis=1)
        go_left = dl <= dr
        if go_left.any():    stack.append((L, idxs[go_left]))
        if (~go_left).any(): stack.append((R, idxs[~go_left]))
    return leaf_of_point

def _precompute_H_M(model: CTWDModel):
    m_seq = model.m_seq
    N = model.P.shape[0]
    L = len(model.leaf_ids)
    E = len(model.edges)
    # 1) route tất cả điểm -> lá
    point_leaf = _route_all_points_vectorized(model)  # (N,)
    model.point_leaf = point_leaf
    # 2) H (L, m_seq) — histogram mỗi chuỗi
    H = np.zeros((L, m_seq), dtype=np.float32)
    for s in range(m_seq):
        mask = (model.Sidx == s)
        if not np.any(mask): continue
        counts = np.bincount(point_leaf[mask], minlength=L).astype(np.float32)
        tot = counts.sum()
        if tot > 0: counts /= tot
        H[:, s] = counts
    model.H = H
    # 3) S_edge_leaf -> CSR (nếu có SciPy) và M = S @ H
    if _HAS_SCIPY:
        SpS = sp.csr_matrix(model.S_edge_leaf)
        model.S_edge_leaf = SpS
        M = (SpS @ H).astype(np.float32)  # (E, m_seq)
    else:
        M = (model.S_edge_leaf @ H).astype(np.float32)
    model.M = M
    # 4) Trọng số cạnh
    model.w = np.array([we for _, _, we in model.edges], dtype=np.float32)

# =========================
# 1.2) CTWD — TAM LE (ragged OK)
# =========================
def _augment_points(seq: np.ndarray, lam_time: float) -> np.ndarray:
    n = seq.shape[0]
    t = (np.arange(n, dtype=float) / max(n, 1))[:, None] * np.sqrt(lam_time)
    return np.hstack([seq, t])

def build_ctwd_tamle(
    M: Union[np.ndarray, Sequence[np.ndarray]],
    lam_time: float = 5.0,
    leaf_size: int = 16,
    max_depth: int = 20,
    seed: int = 0,
    k_split: int = 2,
    box_leaf_size: int = 64,
    box_max_depth: int = 24,
) -> CTWDModel:
    """
    Xây 1 cây global theo TamLe (augment theo thời gian chuẩn hoá → ragged friendly).
    """
    P_raw, Sidx, Tpos, m_seq, lengths, d = _linearize_points_ragged(M)
    # augment từng chuỗi rồi ghép
    P_aug_list = []
    start = 0
    for s in range(m_seq):
        n_i = lengths[s]
        seq = P_raw[start:start+n_i]
        P_aug_list.append(_augment_points(seq, lam_time))
        start += n_i
    P_aug = np.vstack(P_aug_list) if P_aug_list else np.zeros((0, d+1))
    # build tree
    nodes: List[_Node] = []; leaf_ids: List[int] = []
    def _euclid_radius(X):
        if X.shape[0] <= 1: return 0.0
        if X.shape[0] > 1024:
            I = np.random.default_rng(0).choice(X.shape[0], 1024, replace=False); Y = X[I]
        else: Y = X
        j0 = 0; d0 = np.linalg.norm(Y - Y[j0], axis=1); j1 = int(np.argmax(d0))
        d1 = np.linalg.norm(Y - Y[j1], axis=1); return 0.5 * float(d1.max())
    def build(idx: np.ndarray, depth: int, parent: Optional[int], seed_: int) -> int:
        Xsub = P_aug[idx]; h = _euclid_radius(Xsub)
        nid = len(nodes); nodes.append(_Node(idx, h, None, None, parent, False))
        if idx.size <= leaf_size or depth >= max_depth or h == 0.0:
            nodes[nid].is_leaf = True; leaf_ids.append(nid); return nid
        C = _gonzalez_box_nlogk(Xsub, k=k_split, seed=seed_,
                                box_leaf_size=box_leaf_size, box_max_depth=box_max_depth)
        centers = Xsub[C]
        lab = np.argmin(_pairwise_sqdist(Xsub, centers), axis=1)
        if k_split == 2:
            left_idx = idx[lab == 0]; right_idx = idx[lab != 0]
        else:
            cnt = np.bincount(lab, minlength=k_split); main = int(np.argmax(cnt))
            left_idx = idx[lab == main]; right_idx = idx[lab != main]
        if left_idx.size == 0 or right_idx.size == 0:
            mid = idx.size // 2; left_idx = idx[:mid]; right_idx = idx[mid:]
        L = build(left_idx, depth+1, nid, seed_+1); R = build(right_idx, depth+1, nid, seed_+2)
        nodes[nid].left, nodes[nid].right = L, R; return nid
    _ = build(np.arange(P_aug.shape[0]), 0, None, seed)
    # edges & structures
    edges = []
    for cid, nd in enumerate(nodes):
        if nd.parent is not None:
            p = nodes[nd.parent]; w = max(0.0, p.height - nd.height)
            edges.append((nd.parent, cid, w))
    leaf_index_map = {nid: i for i, nid in enumerate(leaf_ids)}
    E, L = len(edges), len(leaf_ids)
    S_edge_leaf = np.zeros((E, L), dtype=np.float32)
    def collect_leaves(nid, out):
        nd = nodes[nid]
        if nd.is_leaf: out.append(nid); return
        if nd.left is not None: collect_leaves(nd.left, out)
        if nd.right is not None: collect_leaves(nd.right, out)
    for e, (pid, cid, _) in enumerate(edges):
        leaves = []; collect_leaves(cid, leaves)
        for ln in leaves:
            j = leaf_index_map[ln]; S_edge_leaf[e, j] = 1.0
    centroids = np.vstack([P_aug[nd.idx].mean(axis=0) for nd in nodes])
    model = CTWDModel(
        P=P_aug, Sidx=Sidx, Tpos=Tpos, lengths=lengths, m_seq=m_seq, d=d,
        nodes=nodes, leaf_ids=leaf_ids, leaf_index_map=leaf_index_map,
        edges=edges, S_edge_leaf=S_edge_leaf, centroids=centroids,
        mode="tamle", lam_time=lam_time
    )
    _precompute_H_M(model)
    return model

# =========================
# 2) CTWD — BANDED (ragged OK, W là TỈ LỆ 0..1)
# =========================
def build_ctwd_banded(
    M: Union[np.ndarray, Sequence[np.ndarray]],
    W: Union[int, float] = 0.15,   # nếu <=1 → coi là tỉ lệ; nếu >1 → sẽ chuyển sang tỉ lệ theo n_i
    lam_idx: float = 5.0,
    lam_tree: float = 10.0,
    leaf_size: int = 32,
    max_depth: int = 20,
    seed: int = 0,
) -> CTWDModel:
    # Linearize (không augment)
    P, Sidx, Tpos, m_seq, lengths, d = _linearize_points_ragged(M)
    # Chuyển W về tỉ lệ (0..1)
    if isinstance(W, (int, float)):
        if W <= 1.0:
            W_frac = float(W)
        else:
            # với ragged: xấp xỉ bằng trung bình W/n_i
            W_frac = float(W) / max(1, int(np.median(lengths)))
    else:
        W_frac = 0.15

    nodes: List[_Node] = []; leaf_ids: List[int] = []

    def _euclid_radius(X):
        if X.shape[0] <= 1: return 0.0
        if X.shape[0] > 1024:
            I = np.random.default_rng(0).choice(X.shape[0], 1024, replace=False); Y = X[I]
        else: Y = X
        j0 = 0; d0 = np.linalg.norm(Y - Y[j0], axis=1); j1 = int(np.argmax(d0))
        d1 = np.linalg.norm(Y - Y[j1], axis=1); return 0.5 * float(d1.max())

    def _gonz2_banded(idx_local, seed_):
        Xsub = P[idx_local]; Tsub = Tpos[idx_local]
        rng = np.random.default_rng(seed_)
        i0 = int(rng.integers(0, Xsub.shape[0]))
        # farthest từ i0 với phạt theo thời gian chuẩn hoá
        eu = np.linalg.norm(Xsub - Xsub[i0], axis=1)
        dt = np.abs(Tsub - Tsub[i0])
        c = eu + lam_idx * (dt ** 2)
        c[dt > W_frac] = BIG
        i1 = int(np.argmax(c))
        # farthest từ i1
        eu = np.linalg.norm(Xsub - Xsub[i1], axis=1)
        dt = np.abs(Tsub - Tsub[i1])
        c = eu + lam_idx * (dt ** 2)
        c[dt > W_frac] = BIG
        i2 = int(np.argmax(c))
        return i1, i2

    def build(idx: np.ndarray, depth: int, parent: Optional[int], seed_: int) -> int:
        Xsub = P[idx]; h = lam_tree * _euclid_radius(Xsub)
        nid = len(nodes); nodes.append(_Node(idx, h, None, None, parent, False))
        if idx.size <= leaf_size or depth >= max_depth or h == 0.0:
            nodes[nid].is_leaf = True; leaf_ids.append(nid); return nid
        i1_local, i2_local = _gonz2_banded(idx, seed_)
        c1, t1 = P[idx[i1_local]], Tpos[idx[i1_local]]
        c2, t2 = P[idx[i2_local]], Tpos[idx[i2_local]]
        eu1 = np.linalg.norm(P[idx] - c1, axis=1); eu2 = np.linalg.norm(P[idx] - c2, axis=1)
        dt1 = np.abs(Tpos[idx] - t1); dt2 = np.abs(Tpos[idx] - t2)
        cost1 = eu1 + lam_idx * (dt1 ** 2); cost2 = eu2 + lam_idx * (dt2 ** 2)
        cost1[dt1 > W_frac] = BIG; cost2[dt2 > W_frac] = BIG
        both_big = (cost1 >= BIG) & (cost2 >= BIG)
        # fallback nếu cả hai ngoài băng: bỏ ràng buộc
        cost1[both_big] = eu1[both_big]; cost2[both_big] = eu2[both_big]
        left_mask = cost1 <= cost2
        if left_mask.all() or (~left_mask).all():
            mid = idx.size // 2; left_idx = idx[:mid]; right_idx = idx[mid:]
        else:
            left_idx = idx[left_mask]; right_idx = idx[~left_mask]
        L = build(left_idx, depth+1, nid, seed_+1); R = build(right_idx, depth+1, nid, seed_+2)
        nodes[nid].left, nodes[nid].right = L, R; return nid

    _ = build(np.arange(P.shape[0]), 0, None, seed)

    edges = []
    for cid, nd in enumerate(nodes):
        if nd.parent is not None:
            p = nodes[nd.parent]; w = max(0.0, p.height - nd.height)
            edges.append((nd.parent, cid, w))

    leaf_index_map = {nid: i for i, nid in enumerate(leaf_ids)}
    E, L = len(edges), len(leaf_ids)
    S_edge_leaf = np.zeros((E, L), dtype=np.float32)
    def collect_leaves(nid, out):
        nd = nodes[nid]
        if nd.is_leaf: out.append(nid); return
        if nd.left is not None: collect_leaves(nd.left, out)
        if nd.right is not None: collect_leaves(nd.right, out)
    for e, (pid, cid, _) in enumerate(edges):
        leaves = []; collect_leaves(cid, leaves)
        for ln in leaves:
            j = leaf_index_map[ln]; S_edge_leaf[e, j] = 1.0

    centroids = np.vstack([P[nd.idx].mean(axis=0) for nd in nodes])

    model = CTWDModel(
        P=P, Sidx=Sidx, Tpos=Tpos, lengths=lengths, m_seq=m_seq, d=d,
        nodes=nodes, leaf_ids=leaf_ids, leaf_index_map=leaf_index_map,
        edges=edges, S_edge_leaf=S_edge_leaf, centroids=centroids,
        mode="banded", lam_idx=lam_idx, W=float(W_frac)
    )
    _precompute_H_M(model)
    return model

# =========================
# 3) DISTANCE APIs
# =========================
def ctwd_between_series_fast(model: CTWDModel, s_ref: int, s_cmp: int) -> float:
    """
    CTWD(s_ref, s_cmp) với cache:
      cost = sum_e w_e * |M[e, s_ref] - M[e, s_cmp]|
    """
    w = model.w; M = model.M
    diff = np.abs(M[:, s_ref] - M[:, s_cmp])
    return float((w * diff).sum())

def ctwd_between_series(model: CTWDModel, s_ref: int, s_cmp: int, p: int = 1) -> float:
    assert p == 1, "Hiện tại hỗ trợ p=1 (W1 trên cây)."
    return ctwd_between_series_fast(model, s_ref, s_cmp)
