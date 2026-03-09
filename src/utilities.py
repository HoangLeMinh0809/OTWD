
# ============================================================
# ABLATION STUDY FOR OTSW PARAMETERS (k-NN Accuracy & MAP)
# ============================================================
"""
Ablation Study for OTSW Hyperparameters
=======================================

This section performs ablation study on OTSW hyperparameters measuring k-NN performance:
- Lambda (lam_time): (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100)
- Max Depth: (5, 10, 15, 20, 25, 30)
- Number of Trees: (1, 3, 5, 7, 9, 11, 13, 15)
- Number of Clusters (k_split): (2, 4, 8, 16, 32)

Default values: lambda=5, depth=30, trees=5, num_cluster=2

When varying one parameter, all others are fixed at default values.
Results include Accuracy, MAP, and execution time for each configuration.
"""

import matplotlib.pyplot as plt

# ---------------------- Configuration ----------------------
# Default parameter values
ABLATION_DEFAULT_LAMBDA = 5
ABLATION_DEFAULT_DEPTH = 30
ABLATION_DEFAULT_TREES = 5
ABLATION_DEFAULT_NUM_CLUSTER = 2  # k_split

# Parameter ranges for ablation
ABLATION_LAMBDA_VALUES = [v**2 for v in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 10000]]
# [1e-06, 2.5e-05, 0.0001, 0.0025, 0.01, 0.25, 1, 25, 100, 2500, 10000, 1000000, 100000000]
ABLATION_DEPTH_VALUES = [5, 10, 15, 20, 25, 30]
ABLATION_TREES_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
ABLATION_NUM_CLUSTER_VALUES = [2, 4, 8, 16, 32, 128]

# Dataset configuration
ABLATION_DATASET = "BasicMotions"
ABLATION_DATATYPE = "UCR_TSL"
ABLATION_LEAF_SIZE = 16
ABLATION_BASE_SEED = 0
ABLATION_K_NN = 1  # k for k-NN


def run_knn_ablation_single(
    X_train, y_train, X_test, y_test,
    lam_time=ABLATION_DEFAULT_LAMBDA,
    max_depth=ABLATION_DEFAULT_DEPTH,
    num_trees=ABLATION_DEFAULT_TREES,
    k_split=ABLATION_DEFAULT_NUM_CLUSTER,
    leaf_size=ABLATION_LEAF_SIZE,
    seed=ABLATION_BASE_SEED,
    k_nn=ABLATION_K_NN,
):
    """
    Run k-NN with OTSW for a single parameter configuration.
    Returns: (Accuracy, MAP, time_total)
    """
    start_time = time.time()
    
    train_len = len(y_train)
    test_len = len(y_test)
    
    # Build sequences: test first, then train
    sequences = [np.asarray(X_test[i], dtype=float) for i in range(test_len)] + \
                [np.asarray(X_train[j], dtype=float) for j in range(train_len)]
    
    # Build OTSW model with multiple trees and average
    dist_accumulator = np.zeros((test_len, train_len), dtype=float)
    train_dist_accumulator = np.zeros((train_len, train_len), dtype=float)
    
    for t in range(num_trees):
        current_seed = seed + t
        
        model_otsw = build_otsw_tamle(
            sequences,
            lam_time=lam_time,
            leaf_size=leaf_size,
            max_depth=max_depth,
            seed=current_seed,
            k_split=k_split,
        )
        
        M_edge_mass = model_otsw.M  # (E, m_total)
        w = model_otsw.w.reshape(-1, 1)  # (E, 1)
        
        # Test-train distances
        for i in range(test_len):
            dist_all = (w * np.abs(M_edge_mass[:, i:i+1] - M_edge_mass)).sum(axis=0)
            dist_accumulator[i, :] += dist_all[test_len : test_len + train_len]
        
        # Train-train distances
        for i in range(train_len):
            idx_i = test_len + i
            dist_all = (w * np.abs(M_edge_mass[:, idx_i:idx_i+1] - M_edge_mass)).sum(axis=0)
            train_dist_accumulator[i, :] += dist_all[test_len : test_len + train_len]
    
    X_test_computed = dist_accumulator / num_trees
    X_computed = train_dist_accumulator / num_trees
    
    # Run k-NN
    clf = neighbors.KNeighborsClassifier(n_neighbors=k_nn, metric="precomputed")
    clf.fit(X_computed, y_train)
    y_pred = clf.predict(X_test_computed)
    acc = 100.0 * accuracy_score(y_test, y_pred)
    
    # Compute MAP
    map_score = compute_map_knn_precomputed(X_computed, X_test_computed, y_train, y_test, k=k_nn)
    
    time_total = time.time() - start_time
    
    return acc, map_score, time_total


def run_knn_ablation_for_param(
    X_train, y_train, X_test, y_test,
    param_name, param_values, num_runs=5, **fixed_params
):
    """
    Run ablation study for a single parameter with multiple runs.
    Returns: DataFrame with columns [param_value, Accuracy_mean, Accuracy_std, MAP_mean, MAP_std, Time_mean, Time_std]
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"Ablation Study (k-NN): {param_name}")
    print(f"Testing {len(param_values)} values: {param_values}")
    print(f"Number of runs per value: {num_runs}")
    print(f"Fixed params: {fixed_params}")
    print(f"{'='*60}")
    
    for val in param_values:
        params = fixed_params.copy()
        params[param_name] = val
        
        print(f"  Testing {param_name}={val}...")
        
        acc_runs = []
        map_runs = []
        time_runs = []
        
        for run_idx in range(num_runs):
            try:
                # Use different seed for each run
                params_with_seed = params.copy()
                params_with_seed['seed'] = ABLATION_BASE_SEED + run_idx * 100
                
                acc, map_score, time_total = run_knn_ablation_single(
                    X_train, y_train, X_test, y_test, **params_with_seed
                )
                acc_runs.append(acc)
                map_runs.append(map_score)
                time_runs.append(time_total)
                print(f"    Run {run_idx+1}/{num_runs}: ACC={acc:.2f}%, MAP={map_score:.2f}%, Time={time_total:.2f}s")
            except Exception as e:
                print(f"    Run {run_idx+1}/{num_runs}: ERROR: {e}")
                acc_runs.append(np.nan)
                map_runs.append(np.nan)
                time_runs.append(np.nan)
        
        # Calculate mean and std
        acc_mean = float(np.nanmean(acc_runs))
        acc_std = float(np.nanstd(acc_runs))
        map_mean = float(np.nanmean(map_runs))
        map_std = float(np.nanstd(map_runs))
        time_mean = float(np.nanmean(time_runs))
        time_std = float(np.nanstd(time_runs))
        
        print(f"    => Mean: ACC={acc_mean:.2f}±{acc_std:.2f}%, MAP={map_mean:.2f}±{map_std:.2f}%, Time={time_mean:.2f}±{time_std:.2f}s")
        
        results.append({
            param_name: val,
            "Accuracy_mean": acc_mean,
            "Accuracy_std": acc_std,
            "MAP_mean": map_mean,
            "MAP_std": map_std,
            "Time_mean": time_mean,
            "Time_std": time_std,
        })
    
    return pd.DataFrame(results)


def plot_knn_ablation_results(df, param_name, save_dir="."):
    """
    Plot ablation results: Accuracy, MAP, and Time vs parameter value with error bars (std).
    Saves directly to the specified directory (default: current directory).
    For lam_time, tick labels show sqrt(value) since the code applies sqrt internally.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # For lam_time: display sqrt(value) as label (code uses sqrt(lam_time) internally)
    if param_name == "lam_time":
        x = [f"{v**0.5:.4g}" for v in df[param_name]]
        display_name = "λ (= √lam_time)"
    else:
        x = df[param_name].astype(str).tolist()
        display_name = param_name
    x_numeric = range(len(x))
    
    # Plot Accuracy with error bars
    axes[0].errorbar(x_numeric, df["Accuracy_mean"], yerr=df["Accuracy_std"], 
                     marker='o', linewidth=2, markersize=8, color='blue', 
                     capsize=4, capthick=1.5, elinewidth=1.5)
    axes[0].set_xlabel(display_name, fontsize=12)
    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    axes[0].set_title(f"Accuracy vs {display_name}", fontsize=14)
    axes[0].set_xticks(x_numeric)
    axes[0].set_xticklabels(x, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAP with error bars
    axes[1].errorbar(x_numeric, df["MAP_mean"], yerr=df["MAP_std"], 
                     marker='s', linewidth=2, markersize=8, color='green',
                     capsize=4, capthick=1.5, elinewidth=1.5)
    axes[1].set_xlabel(display_name, fontsize=12)
    axes[1].set_ylabel("MAP (%)", fontsize=12)
    axes[1].set_title(f"MAP vs {display_name}", fontsize=14)
    axes[1].set_xticks(x_numeric)
    axes[1].set_xticklabels(x, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot Time with error bars
    axes[2].errorbar(x_numeric, df["Time_mean"], yerr=df["Time_std"], 
                     marker='^', linewidth=2, markersize=8, color='red',
                     capsize=4, capthick=1.5, elinewidth=1.5)
    axes[2].set_xlabel(display_name, fontsize=12)
    axes[2].set_ylabel("Time (seconds)", fontsize=12)
    axes[2].set_title(f"Execution Time vs {display_name}", fontsize=14)
    axes[2].set_xticks(x_numeric)
    axes[2].set_xticklabels(x, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure directly in save_dir
    fig_path = os.path.join(save_dir, f"ablation_knn_{param_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Plot saved to {fig_path}")
    return fig_path


# Number of runs for ablation study
ABLATION_NUM_RUNS = 5


def run_full_knn_ablation_study(
    dataset_name=ABLATION_DATASET,
    datatype=ABLATION_DATATYPE,
    save_dir=".",
    num_runs=ABLATION_NUM_RUNS,
):
    """
    Run complete ablation study for all OTSW parameters measuring k-NN performance.
    Each parameter configuration is run num_runs times (default: 5) to compute mean and std.
    Results are saved directly in save_dir (default: current directory).
    """
    print(f"\n{'#'*70}")
    print(f"# OTSW ABLATION STUDY (k-NN) ON DATASET: {dataset_name}")
    print(f"{'#'*70}")
    
    # Load dataset
    if datatype == "UCR_TSL":
        X_train, y_train, X_test, y_test = load_ucr_dataset_tsl("../data/UCR", dataset_name)
    elif datatype == "Human_Actions":
        X_train, y_train, X_test, y_test = load_human_action_dataset("../data/Human_Actions", dataset_name)
    else:
        raise ValueError(f"Unknown datatype: {datatype}")
    
    print(f"\nDataset: {dataset_name}")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print(f"\nDefault parameters:")
    print(f"  - Lambda (lam_time): {ABLATION_DEFAULT_LAMBDA}")
    print(f"  - Max Depth: {ABLATION_DEFAULT_DEPTH}")
    print(f"  - Number of Trees: {ABLATION_DEFAULT_TREES}")
    print(f"  - Number of Clusters (k_split): {ABLATION_DEFAULT_NUM_CLUSTER}")
    print(f"  - Number of runs per config: {num_runs}")
    
    all_results = {}
    
    
    # 1. Ablation on Lambda (lam_time)
    print("\n" + "="*70)
    print("1. ABLATION ON LAMBDA (lam_time)")
    print("="*70)
    df_lambda = run_knn_ablation_for_param(
        X_train, y_train, X_test, y_test,
        param_name="lam_time",
        param_values=ABLATION_LAMBDA_VALUES,
        num_runs=num_runs,
        max_depth=ABLATION_DEFAULT_DEPTH,
        num_trees=ABLATION_DEFAULT_TREES,
        k_split=ABLATION_DEFAULT_NUM_CLUSTER,
    )
    df_lambda.to_csv(os.path.join(save_dir, "ablation_knn_lambda.csv"), index=False)
    plot_knn_ablation_results(df_lambda, "lam_time", save_dir)
    all_results["lambda"] = df_lambda
    '''
    # 2. Ablation on Max Depth
    print("\n" + "="*70)
    print("2. ABLATION ON MAX DEPTH")
    print("="*70)
    df_depth = run_knn_ablation_for_param(
        X_train, y_train, X_test, y_test,
        param_name="max_depth",
        param_values=ABLATION_DEPTH_VALUES,
        num_runs=num_runs,
        lam_time=ABLATION_DEFAULT_LAMBDA,
        num_trees=ABLATION_DEFAULT_TREES,
        k_split=ABLATION_DEFAULT_NUM_CLUSTER,
    )
    df_depth.to_csv(os.path.join(save_dir, "ablation_knn_depth.csv"), index=False)
    plot_knn_ablation_results(df_depth, "max_depth", save_dir)
    all_results["depth"] = df_depth
    
    # 3. Ablation on Number of Trees
    print("\n" + "="*70)
    print("3. ABLATION ON NUMBER OF TREES")
    print("="*70)
    df_trees = run_knn_ablation_for_param(
        X_train, y_train, X_test, y_test,
        param_name="num_trees",
        param_values=ABLATION_TREES_VALUES,
        num_runs=num_runs,
        lam_time=ABLATION_DEFAULT_LAMBDA,
        max_depth=ABLATION_DEFAULT_DEPTH,
        k_split=ABLATION_DEFAULT_NUM_CLUSTER,
    )
    df_trees.to_csv(os.path.join(save_dir, "ablation_knn_trees.csv"), index=False)
    plot_knn_ablation_results(df_trees, "num_trees", save_dir)
    all_results["trees"] = df_trees
    '''
    # 4. Ablation on Number of Clusters (k_split)
    print("\n" + "="*70)
    print("4. ABLATION ON NUMBER OF CLUSTERS (k_split)")
    print("="*70)
    df_cluster = run_knn_ablation_for_param(
        X_train, y_train, X_test, y_test,
        param_name="k_split",
        param_values=ABLATION_NUM_CLUSTER_VALUES,
        num_runs=num_runs,
        lam_time=ABLATION_DEFAULT_LAMBDA,
        max_depth=ABLATION_DEFAULT_DEPTH,
        num_trees=ABLATION_DEFAULT_TREES,
    )
    df_cluster.to_csv(os.path.join(save_dir, "ablation_knn_cluster.csv"), index=False)
    plot_knn_ablation_results(df_cluster, "k_split", save_dir)
    all_results["cluster"] = df_cluster

    # Summary
    print("\n" + "#"*70)
    print("# ABLATION STUDY (k-NN) COMPLETE")
    print("#"*70)
    print(f"\nResults saved to {save_dir}:")
    print("  - ablation_knn_lambda.csv + ablation_knn_lam_time.png")
    print("  - ablation_knn_depth.csv + ablation_knn_max_depth.png")
    print("  - ablation_knn_trees.csv + ablation_knn_num_trees.png")
    print("  - ablation_knn_cluster.csv + ablation_knn_k_split.png")
    
    return all_results


# ---------------------- Usage ----------------------
# Run the full ablation study (5 runs per config by default):
all_results = run_full_knn_ablation_study(dataset_name="ItalyPowerDemand", datatype="UCR_TSL", save_dir=".", num_runs=1)
#
# Or run individual parameter ablations:
#   X_train, y_train, X_test, y_test = load_ucr_dataset_tsl("../data/UCR", "BasicMotions")
#   df_lambda = run_knn_ablation_for_param(X_train, y_train, X_test, y_test, "lam_time", ABLATION_LAMBDA_VALUES, num_runs=5, max_depth=30, num_trees=5, k_split=2)
#
# CSV output columns: param_value, Accuracy_mean, Accuracy_std, MAP_mean, MAP_std, Time_mean, Time_std
