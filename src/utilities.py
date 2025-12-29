import os
from os.path import basename, join
import sys
import time

import joblib
import numpy as np
import ot
import pandas as pd

from sklearn import neighbors
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm

# để import các hàm distance khác nếu bạn đặt trong src/
sys.path.append('src')
# from your_module import dtw_distance_series, taot_distance, gow_sinkhorn_autoscale
import cupy as cp

# =========================
#  Loader UCR từ tslearn
# =========================
def load_ucr_dataset_tsl(data_dir, dataset_name):
    """
    Loads train and test data using tslearn's UCR/UEA loader.
    data_dir hiện không sử dụng nhưng giữ lại cho tương thích.
    """
    from tslearn.datasets import UCR_UEA_datasets
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

    print("Successfully loaded dataset:", dataset_name)
    print("Size of train data:", len(y_train))
    print("Size of test data:", len(y_test))

    return X_train, y_train, X_test, y_test


# =========================
#  Tính MAP đúng kiểu paper
# =========================
def compute_map_knn_precomputed(X_computed, X_test_computed,
                                y_train, y_test, k=1):
    """
    Tính MAP cho k-NN (metric='precomputed') theo mô tả trong paper:

      - Fit k-NN với số láng giềng k.
      - predict_proba trên test -> score cho từng lớp.
      - Với mỗi lớp c:
          AP_c = average_precision_score(1_{y_test==c}, score_c)
      - MAP = trung bình AP_c trên tất cả các lớp.

    Trả về MAP dạng phần trăm [%].
    """
    clf = neighbors.KNeighborsClassifier(
        n_neighbors=k,
        metric="precomputed",
        weights="uniform",
    )
    clf.fit(X_computed, y_train)

    proba = clf.predict_proba(X_test_computed)   # shape (n_test, n_classes)
    classes = clf.classes_

    aps = []
    for c_idx, c in enumerate(classes):
        y_true_c = (y_test == c).astype(int)
        if np.sum(y_true_c) == 0:
            # không có sample lớp c trong test -> bỏ qua
            continue

        y_score_c = proba[:, c_idx]

        # Nếu mọi score y hệt nhau thì PR curve thoái hoá, coi AP = 0
        if np.all(y_score_c == y_score_c[0]):
            aps.append(0.0)
        else:
            ap_c = average_precision_score(y_true_c, y_score_c)
            aps.append(ap_c)

    if len(aps) == 0:
        return 0.0

    return float(np.mean(aps) * 100.0)


# =========================
#  Hàm chính run_knn (nhiều lần)
# =========================
def run_knn(datapath, datatype, alg,
            normalize_cost_matrix=True,
            cost_metric="minkowski",
            num_neighbor_list=[1, 3, 5, 10, 15, 30],
            num_runs=5):
    """
    Run k-NN với precomputed distances, lặp nhiều lần:

      - Mỗi lần:
          + Tính ma trận khoảng cách train–train, test–train.
          + Chạy k-NN với nhiều k, lấy best accuracy (theo k).
          + Tính MAP (dùng k = best_k hoặc k cố định tuỳ chọn).
      - Sau num_runs lần:
          + Lấy mean và variance cho:
                accuracy, map, runtime.

    Ghi vào Excel với các cột:
      dataset, accuracy_mean, accuracy_var, map_mean, map_var,
      runtime_mean, runtime_var
    """
    # 1. Load dữ liệu (chỉ 1 lần)
    if datatype == "UCR_TSL":
        X_train, y_train, X_test, y_test = load_ucr_dataset_tsl("../data/UCR", datapath)
    else:
        raise ValueError(f"Unknown datatype: {datatype}")

    # Downsample CinCECGTorso và MixedShapesSmallTrain xuống 300 mẫu
    if datapath in ["CinCECGTorso", "MixedShapesSmallTrain"]:
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        if len(X_all) > 300:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X_all), size=300, replace=False)
            X_all = X_all[idx]
            y_all = y_all[idx]
            print(f"[DOWNSAMPLE] {datapath}: {len(X_train)+len(X_test)} → 300 samples")
        
        # Chia lại train/test 70/30
        n_train = int(0.7 * len(X_all))
        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_test, y_test = X_all[n_train:], y_all[n_train:]

    train_len = len(y_train)
    test_len = len(y_test)

    # Danh sách để gom kết quả của nhiều lần chạy
    acc_list = []
    map_list = []
    time_list = []

    # ====== Lặp nhiều lần ======
    for run_idx in range(num_runs):
        print(f"\n========== Run {run_idx + 1}/{num_runs} for {alg} on {datapath} ==========")
        t0 = time.time()

        # 2. Khởi tạo ma trận khoảng cách cho lần chạy này
        X_computed = np.zeros((train_len, train_len), dtype=float)      # train–train
        X_test_computed = np.empty((test_len, train_len), dtype=float)  # test–train
        
           
        # 2) Build CTWD model (một lần)
        if alg == "CTWD":
            # 1) Gom chuỗi theo đúng thứ tự: test trước, rồi train (m = test_len + train_len)
            sequences = [np.asarray(X_test[i],  dtype=float) for i in range(test_len)] + \
                        [np.asarray(X_train[j], dtype=float) for j in range(train_len)]
    
            # (tuỳ chọn) sanity check: tất cả chuỗi phải 2D và cùng số kênh d
            d = sequences[0].shape[1]
            assert all(x.ndim == 2 and x.shape[1] == d for x in sequences), "Mỗi chuỗi phải có shape (n_i, d) và cùng d."
    
            # 2) Build CTWD model (một lần)
            # Cấu hình số lượng cây
            n_trees = 5
            
            # Khởi tạo ma trận kết quả tích lũy (Accumulator)
            # X_test_computed cần được khởi tạo bằng 0 để cộng dồn
            # Giả sử shape là (test_len, train_len) như logic cũ
            dist_accumulator = np.zeros((test_len, train_len), dtype=float)

            for t in range(n_trees):
                # QUAN TRỌNG: Thay đổi seed cho mỗi cây để tạo sự đa dạng (diversity)
                # Nếu giữ nguyên seed, 5 cây sẽ y hệt nhau -> trung bình vô nghĩa.
                current_seed = (run_idx * 100) + t 

                # 2) Build CTWD model (cho cây thứ t)
                model_ctwd = build_ctwd_tamle(
                    sequences,
                    lam_time=5.0,       # User parameter
                    leaf_size=16,       # User parameter
                    max_depth=20,       # User parameter
                    seed=current_seed,  # Seed thay đổi theo t
                    k_split=2,          # User parameter
                    box_leaf_size=64,   # User parameter
                    box_max_depth=24    # User parameter
                )

                # 3) Tính distance test-vs-train cho cây t:
                m_total = test_len + train_len
                M_edge_mass = model_ctwd.M          # (E, m_total)
                w = model_ctwd.w.reshape(-1, 1)     # (E, 1)

                for i in range(test_len):
                    # Vector hoá: khoảng cách từ chuỗi i (test) đến tất cả chuỗi
                    dist_all = (w * np.abs(M_edge_mass[:, i:i+1] - M_edge_mass)).sum(axis=0)
                    
                    # Cộng dồn vào kết quả tổng (chỉ lấy phần train)
                    dist_accumulator[i, :] += dist_all[test_len : test_len + train_len]

            # 4) Lấy trung bình
            X_test_computed = dist_accumulator / n_trees
    
        else:
            # 3. Hàm nội bộ tính distance cho một cặp chuỗi
            def _pair_distance(x, y):
                if alg == "DTW":
                    return dtw_distance_series(x, y)
                elif alg == "TAOT":
                    return taot_distance(x, y)
                elif alg == "GOW":
                    C = ot.dist(x, y, metric=cost_metric)
                    if normalize_cost_matrix:
                        maxC = C.max()
                        if maxC > 0:
                            C = C / maxC
                    return gow_sinkhorn_autoscale([], [], C)
                elif alg == "POW":
                    return pow_distance(x, y)
                elif alg == "ASW":
                    return asw_distance(x, y, lam=10.0, auto_weight=True)
                elif alg == "TCOT":
                    x_gpu = (x)
                    y_gpu = (y)
                    return tcot_distance_series(x_gpu, y_gpu)
                elif alg == "OPW":
                    x_gpu = cp.asarray(x, dtype=cp.float64)
                    y_gpu = cp.asarray(y, dtype=cp.float64)
                    return opw_distance_series_gpu(
                        x_gpu, y_gpu,
                        lambda1=1.0,
                        lambda2=0.1,
                        sigma=0.05
                    )

                else:
                    raise ValueError(f"Unknown alg: {alg}")

            # 4. Tính train–train distance (dùng đối xứng để tiết kiệm)
            for i in tqdm(range(train_len), desc=f"Train-train ({alg}) [run {run_idx+1}]"):
                X_computed[i, i] = 0.0
                for j in range(i + 1, train_len):
                    d = _pair_distance(X_train[i], X_train[j])
                    X_computed[i, j] = d
                    X_computed[j, i] = d
    
            # 5. Tính test–train distance
            for i in tqdm(range(test_len), desc=f"Test-train ({alg}) [run {run_idx+1}]"):
                for j in range(train_len):
                    X_test_computed[i, j] = _pair_distance(X_test[i], X_train[j])

        # 6. Chạy kNN với nhiều k, lấy best accuracy
        k_list = sorted(set(num_neighbor_list))
        accuracies = {}
        best_acc = np.nan
        best_k = None

        for k in k_list:
            if k > train_len:
                print(f"Skip k={k} (n_train={train_len} < k)")
                continue

            clf = neighbors.KNeighborsClassifier(n_neighbors=k, metric="precomputed")
            clf.fit(X_computed, y_train)
            y_pred = clf.predict(X_test_computed)
            acc = 100.0 * accuracy_score(y_test, y_pred)
            accuracies[k] = acc
            print(f"[Run {run_idx+1}] Accuracy of {k}NN: {acc:.2f} %")

            if (best_k is None) or (acc > best_acc):
                best_acc = acc
                best_k = k

        if best_k is None:
            best_acc = np.nan

        print(f"[Run {run_idx+1}] Best accuracy: {best_acc:.2f} % (k={best_k})")

        # 7. Tính MAP theo định nghĩa trong paper
        #    Nếu muốn fix k=1 như paper, đổi k_map = 1.
        k_map = best_k if best_k is not None else 1
        map_score = compute_map_knn_precomputed(
            X_computed, X_test_computed, y_train, y_test, k=k_map
        )
        print(f"[Run {run_idx+1}] Mean Average Precision (MAP) with k={k_map}: {map_score:.2f} %")

        # 8. Thời gian chạy lần này
        runtime_s = time.time() - t0
        print(f"[Run {run_idx+1}] Runtime: {runtime_s:.2f} s")

        # Lưu lại
        acc_list.append(best_acc)
        map_list.append(map_score)
        time_list.append(runtime_s)

    # ====== Sau num_runs lần, tính mean và variance ======
    acc_mean = float(np.mean(acc_list))
    acc_var = float(np.var(acc_list))      # nếu muốn sample variance: np.var(..., ddof=1)
    map_mean = float(np.mean(map_list))
    map_var = float(np.var(map_list))
    time_mean = float(np.mean(time_list))
    time_var = float(np.var(time_list))

    print("\n========== Summary over runs ==========")
    print(f"Accuracy: mean={acc_mean:.2f} %, var={acc_var:.4f}")
    print(f"MAP     : mean={map_mean:.2f} %, var={map_var:.4f}")
    print(f"Runtime : mean={time_mean:.2f} s, var={time_var:.4f}")

    # 9. Ghi ra Excel: dataset, accuracy_mean, accuracy_var, map_mean, map_var, runtime_mean, runtime_var
    dataset_key = f"{datapath}_{datatype}"
    out_file = f"{alg}.xlsx"
    cols = [
        "dataset",
        "accuracy_mean", "accuracy_var",
        "map_mean", "map_var",
        "runtime_mean", "runtime_var",
    ]
    
    new_row = {
        "dataset": dataset_key,
        "accuracy_mean": acc_mean,
        "accuracy_var": acc_var,
        "map_mean": map_mean,
        "map_var": map_var,
        "runtime_mean": time_mean,
        "runtime_var": time_var,
    }

    if os.path.exists(out_file):
        try:
            df = pd.read_excel(out_file, engine="openpyxl")
        except Exception:
            df = pd.read_excel(out_file)

        if "dataset" not in df.columns:
            df.insert(0, "dataset", "")

        mask = (df["dataset"] == dataset_key)
        if mask.any():
            for c in cols:
                df.loc[mask, c] = new_row[c]
        else:
            df = pd.concat(
                [df, pd.DataFrame([new_row], columns=cols)],
                ignore_index=True
            )
    else:
        df = pd.DataFrame([new_row], columns=cols)

    df = df[cols]
    df.to_excel(out_file, index=False, engine="openpyxl")

    # Trả về cho code bên ngoài dùng nếu cần
    return {
        "accuracy_mean": acc_mean,
        "accuracy_var": acc_var,
        "map_mean": map_mean,
        "map_var": map_var,
        "runtime_mean": time_mean,
        "runtime_var": time_var,
        "acc_runs": acc_list,
        "map_runs": map_list,
        "time_runs": time_list,
    }
