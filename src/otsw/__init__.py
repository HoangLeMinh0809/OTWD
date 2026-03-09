# otsw_api.py  — RAGGED-FRIENDLY (TAMLE ONLY + FARTHest-Point-Clustering SPLIT)
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Sequence, Union

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

    seqs: List[np.ndarray] = []
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

    P = np.vstack(seqs) if m_seq > 0 else np.zeros((0, d), dtype=float)
    Sidx = np.repeat(np.arange(m_seq, dtype=int), lengths)

    # vị trí thời gian chuẩn hoá (không dùng endpoint=1 để tránh trùng 1.0)
    Tpos_list = [(np.arange(n_i, dtype=float) / max(n_i, 1)) for n_i in lengths]
    Tpos = np.concatenate(Tpos_list) if m_seq > 0 else np.zeros((0,), dtype=float)

    return P, Sidx, Tpos, m_seq, lengths, d


def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Khoảng cách 'lai' (d_tau dùng trong clustering):
      Euclid^2 trên các đặc trưng (trừ cột cuối)  +  |orderA - orderB|   (cột cuối)
    Ghi chú: ta dùng dạng "sqdist + penalty" (không sqrt) vì argmin/argmax không đổi nếu sqrt.
    """
    assert A.ndim == 2 and B.ndim == 2, "Expect 2D arrays"
    assert A.shape[1] == B.shape[1], "Dim mismatch"
    D = A.shape[1]
    if D == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=float)
    if D == 1:
        a_ord = A[:, 0]
        b_ord = B[:, 0]
        return np.abs(a_ord[:, None] - b_ord[None, :])

    Af = A[:, :-1]
    Bf = B[:, :-1]
    aa = (Af * Af).sum(1)[:, None]
    bb = (Bf * Bf).sum(1)[None, :]
    Dsq = np.clip(aa + bb - 2.0 * (Af @ Bf.T), 0.0, None)

    a_ord = A[:, -1]
    b_ord = B[:, -1]
    Pen = np.abs(a_ord[:, None] - b_ord[None, :])
    return Dsq + Pen


# =========================
# 0.1) FARTHest-Point-Clustering (as in your LaTeX)
# =========================
def _farthest_point_clustering_centers(
    Z: np.ndarray,
    k: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Chỉ thực hiện bước "select cluster centers" của Farthest Point Clustering:
      y1 <- random in Z
      for c=2..k:
        y_c <- argmax_{z in Z} min_{y in C} d_tau(z,y)

    Trả về: indices của k centers (theo index local trong Z).
    """
    n = Z.shape[0]
    if n == 0:
        raise ValueError("Empty point set.")
    k_eff = int(min(max(1, k), n))
    rng = np.random.default_rng(seed)

    # y1: random element
    i0 = int(rng.integers(0, n))
    centers = [i0]

    # dmin[z] = min distance from z to current centers
    # init with distances to first center
    dmin = _pairwise_sqdist(Z, Z[i0:i0+1]).reshape(-1)

    # iteratively add farthest point
    for _ in range(1, k_eff):
        # y_c = argmax_z dmin[z]
        ic = int(np.argmax(dmin))
        centers.append(ic)

        # update dmin = min(dmin, d(z, new_center))
        dnew = _pairwise_sqdist(Z, Z[ic:ic+1]).reshape(-1)
        dmin = np.minimum(dmin, dnew)

    return np.array(centers, dtype=int)


# =========================
# 0.2) DATA STRUCTS
# =========================
@dataclass
class _Node:
    idx: np.ndarray
    height: float
    children: List[int]          # k children node-ids (empty if leaf)
    parent: Optional[int]
    is_leaf: bool


@dataclass
class OTSWModel:
    # shared runtime fields
    P: np.ndarray                  # (N, d_aug)
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
    centroids: np.ndarray                   # (num_nodes, d_aug)
    # meta
    mode: str = "tamle"
    lam_time: float = 0.0
    # caches
    point_leaf: Optional[np.ndarray] = None  # (N,)
    H: Optional[np.ndarray] = None           # (L, m_seq)
    M: Optional[np.ndarray] = None           # (E, m_seq)
    w: Optional[np.ndarray] = None           # (E,)


# =========================
# 1) ROUTING & PRECOMPUTE
# =========================
def _route_all_points_vectorized(model: OTSWModel) -> np.ndarray:
    """
    Route mọi điểm xuống lá bằng cách so khoảng cách Euclid tới centroid của các con (k-ary).
    (Lưu ý: dùng Euclid cho routing; d_tau đã dùng khi split/cluster centers.)
    """
    N = model.P.shape[0]
    leaf_of_point = np.empty(N, dtype=np.int32)
    stack = [(0, np.arange(N, dtype=np.int32))]

    nodes = model.nodes
    C = model.centroids
    P = model.P

    while stack:
        nid, idxs = stack.pop()
        nd = nodes[nid]
        if nd.is_leaf:
            j = model.leaf_index_map[nid]
            leaf_of_point[idxs] = j
            continue

        # k-ary routing: tìm child có centroid gần nhất
        children = nd.children
        X = P[idxs]
        dists = np.stack([np.linalg.norm(X - C[c], axis=1) for c in children], axis=1)  # (n, k)
        assignments = np.argmin(dists, axis=1)  # (n,)
        for ci, child_nid in enumerate(children):
            mask = (assignments == ci)
            if mask.any():
                stack.append((child_nid, idxs[mask]))

    return leaf_of_point


def _precompute_H_M(model: OTSWModel):
    """
    - point_leaf: mỗi điểm -> leaf id (0..L-1)
    - H: histogram leaf per sequence (L, m_seq), chuẩn hoá theo tổng điểm mỗi chuỗi
    - S_edge_leaf: (E, L) incidence subtree(edge) vs leaves
    - M = S @ H (E, m_seq)
    - w: edge weights
    """
    m_seq = model.m_seq
    L = len(model.leaf_ids)
    E = len(model.edges)

    # 1) route tất cả điểm -> lá
    point_leaf = _route_all_points_vectorized(model)  # (N,)
    model.point_leaf = point_leaf

    # 2) H (L, m_seq)
    H = np.zeros((L, m_seq), dtype=np.float32)
    for s in range(m_seq):
        mask = (model.Sidx == s)
        if not np.any(mask):
            continue
        counts = np.bincount(point_leaf[mask], minlength=L).astype(np.float32)
        tot = float(counts.sum())
        if tot > 0:
            counts /= tot
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
# 2) OTSW — TAM LE (ragged OK) + Farthest Point Clustering split
# =========================
def _augment_points(seq: np.ndarray, lam_time: float) -> np.ndarray:
    n = seq.shape[0]
    t = (np.arange(n, dtype=float) / max(n, 1))[:, None] * np.sqrt(lam_time)
    return np.hstack([seq, t])


def build_otsw_tamle(
    M: Union[np.ndarray, Sequence[np.ndarray]],
    lam_time: float = 5.0,
    leaf_size: int = 16,
    max_depth: int = 20,
    seed: int = 0,
    k_split: int = 2,
) -> OTSWModel:
    """
    Xây 1 cây global theo TamLe (augment theo thời gian chuẩn hoá → ragged friendly),
    và thay thuật toán chọn tâm/split bằng Farthest Point Clustering như pseudo-code LaTeX.

    Split strategy:
      - dùng FPC để lấy k_split centers trên Xsub (theo d_tau = _pairwise_sqdist)
      - gán label theo nearest-center (argmin d_tau)
      - tạo 1 child cho mỗi cluster không rỗng → k-ary tree
    """
    P_raw, Sidx, Tpos, m_seq, lengths, d = _linearize_points_ragged(M)

    # augment từng chuỗi rồi ghép
    P_aug_list = []
    start = 0
    for s in range(m_seq):
        n_i = lengths[s]
        seq = P_raw[start:start + n_i]
        P_aug_list.append(_augment_points(seq, lam_time))
        start += n_i
    P_aug = np.vstack(P_aug_list) if P_aug_list else np.zeros((0, d + 1), dtype=float)

    nodes: List[_Node] = []
    leaf_ids: List[int] = []

    def _euclid_radius(X: np.ndarray) -> float:
        """
        Bán kính xấp xỉ để làm height: 0.5 * max distance tới 1 điểm farthest (heuristic)
        """
        if X.shape[0] <= 1:
            return 0.0
        if X.shape[0] > 1024:
            I = np.random.default_rng(0).choice(X.shape[0], 1024, replace=False)
            Y = X[I]
        else:
            Y = X
        j0 = 0
        d0 = np.linalg.norm(Y - Y[j0], axis=1)
        j1 = int(np.argmax(d0))
        d1 = np.linalg.norm(Y - Y[j1], axis=1)
        return 0.5 * float(d1.max())

    def build(idx: np.ndarray, depth: int, parent: Optional[int], seed_: int) -> int:
        Xsub = P_aug[idx]
        h = _euclid_radius(Xsub)

        nid = len(nodes)
        nodes.append(_Node(idx=idx, height=h, children=[], parent=parent, is_leaf=False))

        # leaf condition
        if idx.size <= leaf_size or depth >= max_depth or h == 0.0:
            nodes[nid].is_leaf = True
            leaf_ids.append(nid)
            return nid

        # Farthest Point Clustering centers (k_effective auto-clip)
        k_effective = int(min(max(2, k_split), idx.size))
        if k_effective < 2:
            nodes[nid].is_leaf = True
            leaf_ids.append(nid)
            return nid

        # centers on local subset Xsub
        C_local = _farthest_point_clustering_centers(Xsub, k=k_effective, seed=seed_)
        centers = Xsub[C_local]  # (k_effective, d_aug)

        # assign points to nearest center via d_tau
        lab = np.argmin(_pairwise_sqdist(Xsub, centers), axis=1)

        # k-ary split: tạo nhóm cho mỗi cluster không rỗng
        child_groups = []
        for c in range(k_effective):
            mask_c = (lab == c)
            if mask_c.any():
                child_groups.append(idx[mask_c])

        # fallback nếu chỉ ra 1 nhóm (tất cả cùng cluster) → chia đều
        if len(child_groups) < 2:
            chunk = max(1, idx.size // k_effective)
            child_groups = []
            for i in range(0, idx.size, chunk):
                child_groups.append(idx[i:i + chunk])
            if len(child_groups) < 2:
                nodes[nid].is_leaf = True
                leaf_ids.append(nid)
                return nid

        # build children
        children_nids = []
        for i, grp in enumerate(child_groups):
            child_nid = build(grp, depth + 1, nid, seed_ + i + 1)
            children_nids.append(child_nid)
        nodes[nid].children = children_nids
        return nid

    if P_aug.shape[0] == 0:
        # model rỗng
        model = OTSWModel(
            P=P_aug, Sidx=Sidx, Tpos=Tpos, lengths=lengths, m_seq=m_seq, d=d,
            nodes=[_Node(idx=np.array([], dtype=int), height=0.0, children=[], parent=None, is_leaf=True)],
            leaf_ids=[0], leaf_index_map={0: 0},
            edges=[], S_edge_leaf=np.zeros((0, 1), dtype=np.float32),
            centroids=np.zeros((1, d + 1), dtype=float),
            mode="tamle", lam_time=lam_time
        )
        _precompute_H_M(model)
        return model

    # build tree
    _ = build(np.arange(P_aug.shape[0], dtype=int), 0, None, seed)

    # edges & weights
    edges: List[Tuple[int, int, float]] = []
    for cid, nd in enumerate(nodes):
        if nd.parent is not None:
            p = nodes[nd.parent]
            w = max(0.0, p.height - nd.height)
            edges.append((nd.parent, cid, w))

    # leaf mapping
    leaf_index_map = {nid: i for i, nid in enumerate(leaf_ids)}
    E, Lcnt = len(edges), len(leaf_ids)

    # build S_edge_leaf (E, L)
    S_edge_leaf = np.zeros((E, Lcnt), dtype=np.float32)

    def collect_leaves(nid_: int, out: List[int]):
        nd_ = nodes[nid_]
        if nd_.is_leaf:
            out.append(nid_)
            return
        for ch in nd_.children:
            collect_leaves(ch, out)

    for e, (pid, cid, _) in enumerate(edges):
        leaves: List[int] = []
        collect_leaves(cid, leaves)
        for ln in leaves:
            j = leaf_index_map[ln]
            S_edge_leaf[e, j] = 1.0

    # centroids for routing
    centroids = np.vstack([P_aug[nd.idx].mean(axis=0) if nd.idx.size else np.zeros((P_aug.shape[1],), dtype=float)
                           for nd in nodes])

    model = OTSWModel(
        P=P_aug, Sidx=Sidx, Tpos=Tpos, lengths=lengths, m_seq=m_seq, d=d,
        nodes=nodes, leaf_ids=leaf_ids, leaf_index_map=leaf_index_map,
        edges=edges, S_edge_leaf=S_edge_leaf, centroids=centroids,
        mode="tamle", lam_time=lam_time
    )
    _precompute_H_M(model)
    return model


# =========================
# 3) DISTANCE APIs
# =========================
def otsw_between_series_fast(model: OTSWModel, s_ref: int, s_cmp: int) -> float:
    """
    OTSW(s_ref, s_cmp) với cache:
      cost = sum_e w_e * |M[e, s_ref] - M[e, s_cmp]|
    """
    if model.M is None or model.w is None:
        raise ValueError("Model caches not computed. Build the model with build_otsw_tamle().")
    w = model.w
    M = model.M
    diff = np.abs(M[:, s_ref] - M[:, s_cmp])
    return float((w * diff).sum())


def otsw_between_series(model: OTSWModel, s_ref: int, s_cmp: int, p: int = 1) -> float:
    """
    Hiện tại hỗ trợ p=1 (W1 trên cây).
    """
    if p != 1:
        raise ValueError("Currently only supports p=1 (tree-W1).")
    return otsw_between_series_fast(model, s_ref, s_cmp)