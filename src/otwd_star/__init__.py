# ctwd_api.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq

# =========================
# 0) COMMON UTILITIES
# =========================
BIG = 1e12

def _linearize_points(M: np.ndarray):
    # M: (m_seq, n, d) -> P: (N, d)
    m_seq, n, d = M.shape
    P = M.reshape(m_seq * n, d)
    Sidx = np.repeat(np.arange(m_seq), n)  # series id
    Tidx = np.tile(np.arange(n), m_seq)    # time index in series
    return P, Sidx, Tidx, (m_seq, n, d)

def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = (A*A).sum(1)[:, None]; bb = (B*B).sum(1)[None, :]
    return np.clip(aa + bb - 2*A@B.T, 0.0, None)

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
    P: np.ndarray
    Sidx: np.ndarray
    Tidx: np.ndarray
    shape: Tuple[int,int,int]
    nodes: List[_Node]
    leaf_ids: List[int]
    leaf_index_map: dict
    edges: List[Tuple[int,int,float]]       # (parent, child, w_e)
    S_edge_leaf: np.ndarray                 # (E, L)
    centroids: np.ndarray                   # (num_nodes, D_aug)
    # meta (for reference)
    mode: str                               # "tamle" or "banded"
    lam_time: float                         # used by tamle (augment)
    lam_idx: float                          # used by banded
    W: int                                  # used by banded

# =========================
# 1) CTWD — TAM LE (nlogk, Box Composition)
# =========================
def _augment_points(seq: np.ndarray, lam_time: float) -> np.ndarray:
    # seq: (n,d) -> (n,d+1) with last dim = sqrt(lam_time) * (i/n)
    n = seq.shape[0]
    t = (np.arange(n, dtype=float) / max(n, 1))[:, None] * np.sqrt(lam_time)
    return np.hstack([seq, t])

class _KDBoxTree:
    def __init__(self, leaf_size=64, max_depth=24):
        self.leaf_size = leaf_size; self.max_depth = max_depth
        self.boxes = []; self.X = None
    def _build(self, idx, depth):
        X = self.X[idx]; c = X.mean(axis=0)
        r = float(np.sqrt(((X - c)**2).sum(1).max())) if X.shape[0] else 0.0
        bid = len(self.boxes); self.boxes.append({"idx": idx, "c": c, "r": r,
                                                  "L": None, "R": None, "leaf": False})
        if idx.size <= self.leaf_size or depth >= self.max_depth or r == 0.0:
            self.boxes[bid]["leaf"] = True; return bid
        var = X.var(axis=0); d = int(np.argmax(var)); med = np.median(X[:, d])
        mask = X[:, d] <= med
        if mask.all() or (~mask).all():
            mid = idx.size // 2; Lidx = idx[:mid]; Ridx = idx[mid:]
        else:
            Lidx = idx[mask]; Ridx = idx[~mask]
        L = self._build(Lidx, depth+1); R = self._build(Ridx, depth+1)
        self.boxes[bid]["L"] = L; self.boxes[bid]["R"] = R
        return bid
    def fit(self, X):
        self.X = X; self.boxes = []; self._build(np.arange(X.shape[0]), 0)

def _bounds_box(box, centers):
    if centers.size == 0: return 0.0, float("inf")
    d = np.sqrt(((centers - box["c"][None,:])**2).sum(1))
    dmin = float(d.min()); r = box["r"]
    return max(0.0, dmin - r), dmin + r

def _farthest_point_by_boxes(X, centers, kdt: _KDBoxTree, gap_tol=1e-6):
    if centers.size == 0: return 0, 0.0
    heap = []
    L0,U0 = _bounds_box(kdt.boxes[0], centers)
    heapq.heappush(heap, (-U0, 0, L0))
    best_idx, best_val = None, -1.0
    while heap:
        negU, bid, Lb = heapq.heappop(heap)
        Ub = -negU
        # lazy tighten
        L2,U2 = _bounds_box(kdt.boxes[bid], centers)
        if U2 < Ub - 1e-12 or L2 > Lb + 1e-12:
            heapq.heappush(heap, (-U2, bid, L2)); continue
        if best_val >= U2 - 1e-15: break
        box = kdt.boxes[bid]
        if box["leaf"] or (U2 - L2) <= gap_tol:
            pts = kdt.X[box["idx"]]
            D = _pairwise_sqdist(pts, centers)
            dmin = np.sqrt(D.min(axis=1))
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

def build_ctwd_tamle(
    M: np.ndarray,
    lam_time: float = 5.0,
    leaf_size: int = 16,
    max_depth: int = 20,
    seed: int = 0,
    k_split: int = 2,
    box_leaf_size: int = 64,
    box_max_depth: int = 24,
) -> CTWDModel:
    """
    Xây 1 cây global theo Tam Le (augment + Box Composition ~ nlogk).
    """
    # augment toàn bộ điểm rồi build tree bằng Gonzalez-Box
    m_seq, n, d = M.shape
    P_aug = np.vstack([_augment_points(M[s], lam_time) for s in range(m_seq)])
    Sidx = np.repeat(np.arange(m_seq), n)
    Tidx = np.tile(np.arange(n), m_seq)

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

    edges = []
    for cid, nd in enumerate(nodes):
        if nd.parent is not None:
            p = nodes[nd.parent]; w = max(0.0, p.height - nd.height)
            edges.append((nd.parent, cid, w))

    leaf_index_map = {nid: i for i, nid in enumerate(leaf_ids)}
    E, L = len(edges), len(leaf_ids)
    S_edge_leaf = np.zeros((E, L))
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

    return CTWDModel(
        P=P_aug, Sidx=Sidx, Tidx=Tidx, shape=(m_seq, n, d),
        nodes=nodes, leaf_ids=leaf_ids, leaf_index_map=leaf_index_map,
        edges=edges, S_edge_leaf=S_edge_leaf, centroids=centroids,
        mode="tamle", lam_time=lam_time, lam_idx=0.0, W=0
    )

# =========================
# 2) CTWD — BANDED (theo scaffold bạn dán)
# =========================
def build_ctwd_banded(
    M: np.ndarray,
    W: int = 15,
    lam_idx: float = 5.0,
    lam_tree: float = 10.0,
    leaf_size: int = 32,
    max_depth: int = 20,
    seed: int = 0,
) -> CTWDModel:
    P, Sidx, Tidx, (m_seq, n, d) = _linearize_points(M)
    nodes: List[_Node] = []; leaf_ids: List[int] = []

    def _euclid_radius(X):
        if X.shape[0] <= 1: return 0.0
        if X.shape[0] > 1024:
            I = np.random.default_rng(0).choice(X.shape[0], 1024, replace=False); Y = X[I]
        else: Y = X
        j0 = 0; d0 = np.linalg.norm(Y - Y[j0], axis=1); j1 = int(np.argmax(d0))
        d1 = np.linalg.norm(Y - Y[j1], axis=1); return 0.5 * float(d1.max())

    def _gonz2_banded(idx_local, seed_):
        Xsub = P[idx_local]; Tsub = Tidx[idx_local]
        rng = np.random.default_rng(seed_)
        i0 = int(rng.integers(0, Xsub.shape[0]))
        # farthest from i0
        eu = np.linalg.norm(Xsub - Xsub[i0], axis=1)
        dt = np.abs(Tsub - Tsub[i0]); c = eu + lam_idx * (dt.astype(float)**2) / (n**2)
        c[dt > W] = BIG; i1 = int(np.argmax(c))
        # farthest from i1
        eu = np.linalg.norm(Xsub - Xsub[i1], axis=1)
        dt = np.abs(Tsub - Tsub[i1]); c = eu + lam_idx * (dt.astype(float)**2) / (n**2)
        c[dt > W] = BIG; i2 = int(np.argmax(c))
        return i1, i2  # local indices

    def build(idx: np.ndarray, depth: int, parent: Optional[int], seed_: int) -> int:
        Xsub = P[idx]; h = lam_tree * _euclid_radius(Xsub)
        nid = len(nodes); nodes.append(_Node(idx, h, None, None, parent, False))
        if idx.size <= leaf_size or depth >= max_depth or h == 0.0:
            nodes[nid].is_leaf = True; leaf_ids.append(nid); return nid
        i1_local, i2_local = _gonz2_banded(idx, seed_)
        c1, t1 = P[idx[i1_local]], Tidx[idx[i1_local]]
        c2, t2 = P[idx[i2_local]], Tidx[idx[i2_local]]
        eu1 = np.linalg.norm(P[idx] - c1, axis=1); eu2 = np.linalg.norm(P[idx] - c2, axis=1)
        dt1 = np.abs(Tidx[idx] - t1); dt2 = np.abs(Tidx[idx] - t2)
        cost1 = eu1 + lam_idx * (dt1.astype(float)**2) / (n**2)
        cost2 = eu2 + lam_idx * (dt2.astype(float)**2) / (n**2)
        cost1[dt1 > W] = BIG; cost2[dt2 > W] = BIG
        both_big = (cost1 >= BIG) & (cost2 >= BIG)
        cost1[both_big] = eu1[both_big]; cost2[both_big] = eu2[both_big]  # fallback
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
    S_edge_leaf = np.zeros((E, L))
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

    return CTWDModel(
        P=P, Sidx=Sidx, Tidx=Tidx, shape=(m_seq, n, d),
        nodes=nodes, leaf_ids=leaf_ids, leaf_index_map=leaf_index_map,
        edges=edges, S_edge_leaf=S_edge_leaf, centroids=centroids,
        mode="banded", lam_time=0.0, lam_idx=lam_idx, W=W
    )

# =========================
# 3) RUNTIME: route & distance (dùng chung cho cả 2 biến thể)
# =========================
def _route_to_leaf(model: CTWDModel, x: np.ndarray) -> int:
    nid = 0
    while not model.nodes[nid].is_leaf:
        L = model.nodes[nid].left; R = model.nodes[nid].right
        dl = float(np.linalg.norm(x - model.centroids[L]))
        dr = float(np.linalg.norm(x - model.centroids[R]))
        nid = L if dl <= dr else R
    return model.leaf_index_map[nid]

def _series_hist_on_leaves(model: CTWDModel, s: int) -> np.ndarray:
    m_seq, n, d = model.shape
    mask = model.Sidx == s
    xs = model.P[mask]
    leaf_ids = [_route_to_leaf(model, x) for x in xs]
    L = len(model.leaf_ids)
    h = np.bincount(np.array(leaf_ids, dtype=int), minlength=L).astype(float)
    if h.sum() > 0: h /= h.sum()
    return h

def ctwd_between_series(model: CTWDModel, s_ref: int, s_cmp: int, p: int = 1) -> float:
    assert p == 1, "Hiện hỗ trợ p=1 (W1 trên cây)."
    h1 = _series_hist_on_leaves(model, s_ref)
    h2 = _series_hist_on_leaves(model, s_cmp)
    w = np.array([we for _,_,we in model.edges], dtype=float)
    m1 = model.S_edge_leaf @ h1
    m2 = model.S_edge_leaf @ h2
    return float(np.sum(w * np.abs(m1 - m2)))
