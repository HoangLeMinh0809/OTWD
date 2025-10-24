import os
from os.path import basename, join
import sys
import joblib

import numpy as np
import ot
from aeon.datasets import load_from_ts_file
from sklearn import neighbors
from sklearn.metrics import accuracy_score


sys.path.append('src')
from gow import gow_sinkhorn_autoscale
from tqdm import tqdm
from bs import *
import time
from dtw import *
from otwd_star import *

import pandas as pd

def load_human_action_dataset(data_dir, dataset_name):
    '''
    Loads train and test data from the folder where the
    Human Actions datasets are stored.
    '''
    X_train = joblib.load(os.path.join(data_dir, dataset_name, "X_train.pkl"))
    y_train = joblib.load(os.path.join(data_dir, dataset_name, "y_train.pkl"))
    X_test = joblib.load(os.path.join(data_dir, dataset_name, "X_test.pkl"))
    y_test = joblib.load(os.path.join(data_dir, dataset_name, "y_test.pkl"))

    print("Successfully loaded dataset:", dataset_name)
    print("Size of train data:", len(y_train))
    print("Size of test data:", len(y_test))

    return X_train, y_train, X_test, y_test

def fix_array(X):
    X_new = np.empty((X.shape[0], X.shape[2], X.shape[1]))

    for i in range (X_new.shape[0]):
        X_new[i] = np.transpose(X[i])

    return X_new

def load_ucr_dataset(data_dir, dataset_name):
    '''
    Loads train and test data from the folder where the
    UCR datasets are stored.
    '''
    X, y_train = load_from_ts_file(os.path.join(data_dir, dataset_name, f'{dataset_name}_TRAIN'))
    X_train = fix_array(X)
    X, y_test = load_from_ts_file(os.path.join(data_dir, dataset_name, f'{dataset_name}_TEST'))
    X_test = fix_array(X)

    print("Successfully loaded dataset:", dataset_name)
    print("Size of train data:", len(y_train))
    print("Size of test data:", len(y_test))

    return X_train, y_train, X_test, y_test

def load_ucr_dataset_tsl(data_dir, dataset_name):
    '''
    Loads train and test data using tslearn's UCR/UEA loader.
    '''
    from tslearn.datasets import UCR_UEA_datasets
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

    print("Successfully loaded dataset:", dataset_name)
    print("Size of train data:", len(y_train))
    print("Size of test data:", len(y_test))

    return X_train, y_train, X_test, y_test

def run_knn(datapath, datatype, alg,
        normalize_cost_matrix=True,
        cost_metric="minkowski",
        num_neighbor_list=[1, 3, 5, 10, 15, 30]):
    """
    Run k-NN with precomputed distances and save results to Excel.
    """
    t0 = time.time()

    # 1) Load data
    if datatype == "UCR":
        X_train, y_train, X_test, y_test = load_ucr_dataset("../data/UCR", datapath)
    elif datatype == "Human_Actions":
        X_train, y_train, X_test, y_test = load_human_action_dataset("../data/Human_Actions", datapath)
    elif datatype == "UCR_TSL":
        X_train, y_train, X_test, y_test = load_ucr_dataset_tsl("../data/UCR", datapath)
    else:
        raise ValueError(f"Unknown datatype: {datatype}")

    train_len = len(y_train)
    test_len  = len(y_test)
    # Assume train_seqs = [X1, X2, ..., XK] where each has shape (n_s, d)
    

    # 2) Initialize distance matrices (train x train unused -> kept as ones for fit)
    X_computed       = np.ones((train_len, train_len), dtype=float)
    X_test_computed  = np.empty((test_len, train_len), dtype=float)

    # 3) Compute test–train distances using the selected algorithm
    if alg == "GOW":
        for i in tqdm(range(test_len), desc="Test batches"):
            for j in range(train_len):
                C = ot.dist(X_test[i], X_train[j], metric=cost_metric)
                if normalize_cost_matrix:
                    maxC = C.max()
                    if maxC > 0:
                        C = C / maxC
                X_test_computed[i, j] = gow_sinkhorn_autoscale([], [], C)
    
    elif alg == "CTWD-TamLe":
    # 1) Prepare M with proper shape (m = test_len + train_len, n, d)
        sequences = [X_test[i] for i in range(test_len)] + [X_train[j] for j in range(train_len)]
        M = np.asarray(sequences, dtype=float)

    # 2) Build CTWD model (one-time)
        model_ctwd = build_ctwd_tamle(
            M,
            lam_time=5.0,     # hệ số lambda cho (i/n - j/m)^2 (ẩn trong augment)
            leaf_size=16, max_depth=20,
            seed=0,
            k_split= M.shape[1],        # split into n clusters, where n is the number of elements per series
            box_leaf_size=64, box_max_depth=24
        )

    # 3) Compute distance between each pair of sequences
        for i in tqdm(range(test_len), desc="CTWD-TamLe | test rows"):
            for j in range(train_len):
                dist_w1 = ctwd_between_series(model_ctwd, s_ref=i, s_cmp=test_len + j, p=1)
                X_test_computed[i, j] = dist_w1


    elif alg == "DTW":
        for i in tqdm(range(test_len), desc="Test batches"):
            for j in range(train_len):
                val = dtw_mean_per_step(X_test[i], X_train[j])  # phòng khi hàm trả tuple
                if isinstance(val, (tuple, list, np.ndarray)):
                    val = val[0]
                X_test_computed[i, j] = float(val)

    elif alg == "BS_MAP":
        for i in tqdm(range(test_len), desc="Test batches"):
            for j in range(train_len):
                X_test_computed[i, j] = 1/2*(bs_distance_between_series(X_test[i], X_train[j], anchor_mode="MAP", num_trees=1) + bs_distance_between_series(X_train[j], X_test[i], anchor_mode="MAP", num_trees=1))

    elif alg == "BS_MC":
        for i in tqdm(range(test_len), desc="Test batches"):
            for j in range(train_len):
                X_test_computed[i, j] = 1/2*(bs_distance_between_series(X_test[i], X_train[j], anchor_mode="MC", num_trees=8) + bs_distance_between_series(X_train[j], X_test[i], anchor_mode="MC", num_trees=8))
    elif alg == "BS_SOFT":
        for i in tqdm(range(test_len), desc="Test batches"):
            for j in range(train_len):
                X_test_computed[i, j] = 1/2*(bs_distance_between_series(X_test[i], X_train[j], anchor_mode="SOFT", num_trees=1) + bs_distance_between_series(X_train[j], X_test[i], anchor_mode="SOFT", num_trees=8))
    else:
        raise ValueError(f"Unknown alg: {alg}")

    # 4) Ensure non-negative
    X_computed      = np.maximum(X_computed, 0.0)
    X_test_computed = np.maximum(X_test_computed, 0.0)

    print("Distance using:", alg)

    # 5) Evaluate KNN for requested k values, skip k > train_len
    desired_k = [1, 3, 5, 10, 15, 30]
    accuracies = {k: np.nan for k in desired_k}

    for k in desired_k:
        if k <= train_len:
            clf = neighbors.KNeighborsClassifier(n_neighbors=k, metric="precomputed")
            clf.fit(X_computed, y_train)
            y_pred = clf.predict(X_test_computed)
            acc = 100.0 * accuracy_score(y_test, y_pred)
            accuracies[k] = acc
            print(f"Accuracy of {k}NN: {acc:.2f} %")
        else:
            print(f"Skip k={k} (n_train={train_len} < k)")

    runtime_s = time.time() - t0
    print(f"Total runtime: {runtime_s:.2f} s")

    # 6) Write Excel: file = {alg}.xlsx, row = datapath_datatype
    dataset_key = f"{datapath}_{datatype}"
    out_file = f"{alg}.xlsx"
    cols = ["dataset", "k=1", "k=3", "k=5", "k=10", "k=15", "k=30", "runtime_s"]
    new_row = {
        "dataset": dataset_key,
        "k=1":  accuracies[1],
        "k=3":  accuracies[3],
        "k=5":  accuracies[5],
        "k=10": accuracies[10],
        "k=15": accuracies[15],
        "k=30": accuracies[30],
        "runtime_s": runtime_s
    }

    if os.path.exists(out_file):
    # Read existing file and update/overwrite the row with the same dataset_key
        try:
            df = pd.read_excel(out_file, engine="openpyxl")
        except Exception:
            df = pd.read_excel(out_file)
        if "dataset" not in df.columns:
            # Trường hợp file cũ không có cột "dataset", cố gắng khôi phục
            df.insert(0, "dataset", "")
        # Cập nhật/append
        mask = (df["dataset"] == dataset_key)
        if mask.any():
            for c in cols:
                df.loc[mask, c] = new_row[c]
        else:
            df = pd.concat([df, pd.DataFrame([new_row], columns=cols)], ignore_index=True)
    else:
        df = pd.DataFrame([new_row], columns=cols)

    # Order columns correctly and save
    df = df[cols]
    df.to_excel(out_file, index=False, engine="openpyxl")

    return accuracies, runtime_s