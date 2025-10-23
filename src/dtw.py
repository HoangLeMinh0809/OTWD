import numpy as np
from math import inf

def _zscore_1d(x, eps=1e-12):
    x = np.asarray(x, float)
    mu = x.mean()
    sd = x.std()
    if sd < eps: sd = 1.0
    return (x - mu) / sd

def _to_2d_series(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        return x[:, None]  # (T,1)
    return x  # (T,C)

def dtw_mean_per_step(
    x1, x2,
    window_ratio=0.1,      # tỉ lệ Sakoe–Chiba (0.05–0.1 thường tốt)
    penalty=0.05,          # phạt cho bước ngang/dọc (0–0.2)
    znormalize=True,       # z-norm từng chuỗi
    use_derivative=False,  # DDTW
    return_mean_per_step=True
):
    """
    DTW đa biến với:
      - ground cost = ||x[i]-y[j]||_2
      - optional: derivative DTW
      - penalty cho bước ngang/dọc để hạn chế over-warping
      - chuẩn hoá theo độ dài đường đi (mean per step) nếu return_mean_per_step=True
    Trả về (dist, path_len).
    """
    X = _to_2d_series(x1)
    Y = _to_2d_series(x2)
    if znormalize:
        X = np.apply_along_axis(_zscore_1d, 0, X)
        Y = np.apply_along_axis(_zscore_1d, 0, Y)

    if use_derivative:
        # central difference; giữ kích thước bằng cách copy biên
        def deriv(A):
            D = np.empty_like(A)
            D[1:-1] = (A[2:] - A[:-2]) / 2.0
            D[0] = A[1] - A[0]
            D[-1] = A[-1] - A[-2]
            return D
        X = deriv(X); Y = deriv(Y)

    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        return inf, 0

    # Sakoe–Chiba window theo tỉ lệ
    W = int(np.ceil(window_ratio * max(n, m))) if window_ratio is not None else max(n, m)
    W = max(W, abs(n - m))  # an toàn

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    # điền DP
    for i in range(1, n + 1):
        j1 = max(1, i - W)
        j2 = min(m, i + W)
        xi = X[i - 1]
        for j in range(j1, j2 + 1):
            c = np.linalg.norm(xi - Y[j - 1])  # L2 theo chiều đặc trưng
            # phạt bước ngang/dọc để hạn chế warp quá mức
            D[i, j] = c + min(
                D[i - 1, j - 1],                 # diag
                D[i - 1, j]     + penalty,       # up
                D[i, j - 1]     + penalty        # left
            )

    total = D[n, m]
    if not np.isfinite(total):
        return inf, 0

    # backtrack để tính path_len (cho chuẩn hoá)
    i, j = n, m
    path_len = 0
    while i > 0 or j > 0:
        path_len += 1
        # chọn bước đã tạo nên D[i,j]
        choices = []
        if i > 0 and j > 0: choices.append((D[i - 1, j - 1], i - 1, j - 1, 0.0))
        if i > 0:           choices.append((D[i - 1, j] + penalty, i - 1, j, penalty))
        if j > 0:           choices.append((D[i, j - 1] + penalty, i, j - 1, penalty))
        # chọn argmin của "giá trước khi cộng c" (không cần c vì cùng c ở ô (i,j))
        choices.sort(key=lambda t: t[0])
        _, ii, jj, _ = choices[0]
        i, j = ii, jj

    if return_mean_per_step:
        return float(total / path_len), int(path_len)
    else:
        # cũng có thể dùng total/(n+m) hoặc total/min(n,m) tuỳ bạn
        return float(total), int(path_len)
