import os
from os.path import basename, join
import sys
import joblib

import numpy as np
import ot
from aeon.datasets import load_from_ts_file
from sklearn import neighbors
from sklearn.metrics import accuracy_score

from dtw import dtw_mean_per_step
from op_tree.class_ultratwd_gd import UltraTWD_GD

sys.path.append('src')
from gow import gow_sinkhorn_autoscale
import torch

def load_human_action_dataset(data_dir, dataset_name):
    '''
    Loads train and test data from the folder in which
    the Human Actions dataset are stored.
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
    Loads train and test data from the folder in which
    the UCR dataset are stored.
    '''
    X, y_train = load_from_ts_file(os.path.join(data_dir, dataset_name, f'{dataset_name}_TRAIN'))
    X_train = fix_array(X)
    X, y_test = load_from_ts_file(os.path.join(data_dir, dataset_name, f'{dataset_name}_TEST'))
    X_test = fix_array(X)

    print("Successfully loaded dataset:", dataset_name)
    print("Size of train data:", len(y_train))
    print("Size of test data:", len(y_test))

    return X_train, y_train, X_test, y_test

def run_knn(X_train, y_train, X_test, y_test, normalize_cost_matrix=True, cost_metric="minkowski", num_neighbor_list=[1, 3, 5], method = "gow", model=None):
    '''
    Run k-NN classifier with distance between a test and 
    train sequence computed using GOW.
    '''

    train_len = len(y_train)
    test_len = len(y_test)
    X_computed = np.ones((train_len, train_len))
    X_test_computed = np.empty((test_len, train_len))

    if method == "op_tree" and model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        folder = f"UltraTWD_gradient_descent_results"
        os.makedirs(folder, exist_ok=True)

        model_path = f"{folder}/ultratwd_gradient_iter"
        print(f"  Initializing the UltraTWD-GD on {device}...")

        def add_positional_encoding(X):
            """Add positional encoding to the time series data."""
            n_samples = X.shape[0]
            series_length = X.shape[1]
            new_X = []
            for i in range(n_samples):
                median = np.median(X[i, :, 0])
                for t in range(series_length):
                    new_X.append([X[i, t, 0], median* t / series_length])
            return np.array(new_X)
        X = add_positional_encoding(np.concatenate((X_train, X_test), axis=0))
        
        model = UltraTWD_GD(X, model_path, device=device, lambda_cost=1.0)  # lambda_cost là tham số cho compute_new_cost
        opt = torch.optim.Adam([model.param], lr=1)
        log = model.optimize_ultrametric_by_gradient_descent(
            opt, max_iter=80
        )

    for i in range(test_len):
        print("Batch:", str(i+1) + "/" + str(test_len))
        for j in range(train_len):
            C = ot.dist(X_test[i], X_train[j], metric=cost_metric)
            if normalize_cost_matrix:
                C = C / C.max()
            if method == "gow":
                X_test_computed[i][j] = gow_sinkhorn_autoscale([], [], C)
            elif method == "dtw":
                X_test_computed[i][j], _ = dtw_mean_per_step(X_test[i], X_train[j])
            elif method == "op_tree":
                N = X_train.shape[0]        # n_sequences
                L = X_train.shape[1]        # series_length (giả sử cố định)
                idx1, idx2 = [i], [j]  # các cặp chuỗi muốn so sánh

                twd, tsec = UltraTWD_GD.compute_twd_distance_for_sequences(
                    subtree=model.subtree,
                    param=model.param,
                    parents=model.parents,
                    n_sequences=N,
                    series_length=L,
                    idx1=idx1,
                    idx2=idx2
                )
                X_test_computed[i][j] = twd[0]


    for k in num_neighbor_list:
        clf = neighbors.KNeighborsClassifier(n_neighbors = k, metric="precomputed")
        clf.fit(X_computed, y_train)
        y_pred = clf.predict(X_test_computed)
    
        print("Accuracy of " + str(k) + "NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
