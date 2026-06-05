"""
MOSAIC — Multi-Omics Single-cell Analysis Integration Core
===========================================================

Modes:
  training   : GCN autoencoder for multi-view dimensionality reduction.
  clustering : Affinity-based network fusion + KMeans clustering
               on the low-dimensional embeddings produced by training.

Usage:
  python main.py -t my_dataset -i ./input/my_dataset.list -m training -n 3 --num_views 2
  python main.py -t my_dataset -i ./input/my_dataset.list -m clustering -n 3 --num_views 2
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
from os.path import isfile

from train import train_test
from utils import cluster, testaff, cluster_label, evaluate_clustering
from fusion import network_fusion

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def run_training(args, dataset_name, view_list, lr_e, num_epoch):
    """Training module: learn low-dimensional embeddings via GCN autoencoder."""
    time_start = time.time()

    mean_emb, index = train_test(
        args.file_input, dataset_name, view_list,
        args.cluster_num, lr_e, num_epoch,
    )

    # Save mean embedding
    fea_tmp_file = './fea/' + dataset_name + '.fea'
    fea = pd.DataFrame(
        data=mean_emb.detach().cpu().numpy(),
        index=index,
        columns=['v' + str(x) for x in range(mean_emb.shape[1])],
    )
    fea.to_csv(fea_tmp_file, header=True, index=True, sep='\t')

    elapsed = time.time() - time_start
    print(f"Training finished in {elapsed:.1f}s")
    print(f"Mean embedding saved to {fea_tmp_file}")


def run_clustering(args, dataset_name):
    """
    Clustering module: affinity computation → network fusion → KMeans.

    Reads per-view embedding CSV files produced by the training step,
    computes affinity matrices, fuses them via iterative SNF, and clusters.
    """
    time_start = time.time()

    # ---- 1. Load per-view embeddings ----
    view_files = [
        f'./fea/_view{v}.csv' for v in range(1, 10)
        if isfile(f'./fea/_view{v}.csv')
    ]
    if len(view_files) < 2:
        raise FileNotFoundError(
            "Per-view embedding files not found. "
            "Run training mode first to generate ./fea/_view*.csv"
        )

    print(f"Found {len(view_files)} view embedding files.")

    view_data = []
    for vf in view_files:
        df = pd.read_csv(vf, index_col=0).T
        view_data.append(df.values.astype(np.float64))

    n_samples = view_data[0].shape[0]
    print(f"Number of samples: {n_samples}")

    # ---- 2. Compute distance matrices and affinity ----
    aff_k = args.aff_k

    affinity_list = []
    for i, vd in enumerate(view_data):
        dist_mat = np.linalg.norm(
            vd[:, np.newaxis, :] - vd[np.newaxis, :, :], axis=2
        )
        aff = testaff(dist_mat, aff_k)
        affinity_list.append(aff)
        print(f"  View {i + 1} affinity computed (k={aff_k}).")

    # ---- 3. Network fusion ----
    fusion_k = args.fusion_k
    niter = args.niter
    beta = args.beta
    c_val = args.cluster_num if args.cluster_num > 2 else 3

    print(f"\nRunning network fusion (c={c_val}, k={fusion_k}, niter={niter})...")
    S, F_eig1 = network_fusion(
        affinity_list, c=c_val, k=fusion_k, niter=niter, beta=beta,
    )
    print("Network fusion completed.")

    # ---- 4. Clustering ----
    labels = cluster_label(S, args.cluster_num)
    labels_arr = np.array(labels) + 1  # 1-indexed

    # ---- 5. Save results ----
    out_dir = './analysis/results/'
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir + dataset_name + '.mosiac'
    result_df = pd.DataFrame({'label': labels_arr}, index=range(1, n_samples + 1))
    result_df.to_csv(out_file, header=True, index=True, sep='\t')
    print(f"Clustering result saved to {out_file}")

    # ---- 6. Evaluation (if label file provided) ----
    if args.label_file and isfile(args.label_file):
        metrics = evaluate_clustering(S, args.cluster_num, args.label_file, args.label_col)
        print("\n---- Clustering Evaluation ----")
        for name, val in metrics.items():
            print(f"  {name}: {val:.4f}")

    # Silhouette score
    sil = silhouette_score(S, labels_arr - 1)
    print(f"\n  Silhouette Score: {sil:.4f}")

    elapsed = time.time() - time_start
    print(f"\nClustering finished in {elapsed:.1f}s")


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(
        description='MOSAIC — Multi-Omics Single-cell Analysis Integration Core'
    )
    parser.add_argument(
        "-i", dest='file_input', default="./input/input.list",
        help="Path to input list file (one data file path per line)."
    )
    parser.add_argument(
        "-m", dest='run_mode', default="training",
        help="Run mode: 'training' (dim-reduction) or 'clustering' (fusion)."
    )
    parser.add_argument(
        "-n", dest='cluster_num', type=int, default=-1,
        help="Number of clusters (required)."
    )
    parser.add_argument(
        "-t", dest='type', default="dataset",
        help="Dataset name for output file naming."
    )
    # Clustering-specific
    parser.add_argument(
        "--aff_k", dest='aff_k', type=int, default=18,
        help="Nearest neighbors for affinity matrix."
    )
    parser.add_argument(
        "--fusion_k", dest='fusion_k', type=int, default=42,
        help="Nearest neighbors for network fusion."
    )
    parser.add_argument(
        "--niter", dest='niter', type=int, default=30,
        help="Max iterations for network fusion."
    )
    parser.add_argument(
        "--beta", dest='beta', type=float, default=0.8,
        help="Update rate for network fusion (0 < beta < 1)."
    )
    parser.add_argument(
        "--label_file", dest='label_file', default=None,
        help="Path to true label CSV (for evaluation)."
    )
    parser.add_argument(
        "--label_col", dest='label_col', default='label',
        help="Column name for true labels in label CSV."
    )
    parser.add_argument(
        "--num_views", dest='num_views', type=int, default=2,
        help="Number of omics views (default: 2)."
    )

    # ---- Hyperparameters ----
    num_epoch = 600
    lr_e = 1e-5

    args = parser.parse_args()
    view_list = list(range(1, args.num_views + 1))
    dataset_name = args.type

    # Validate cluster number
    if args.cluster_num == -1:
        print("Please set the number of clusters (-n)!")
        sys.exit(1)

    print(f"MOSAIC | dataset={dataset_name} | mode={args.run_mode} "
          f"| clusters={args.cluster_num}")

    # ---- Dispatch ----
    if args.run_mode == 'training':
        run_training(args, dataset_name, view_list, lr_e, num_epoch)

    elif args.run_mode == 'clustering':
        run_clustering(args, dataset_name)

    else:
        raise ValueError(f"Unknown run mode: {args.run_mode}. "
                         f"Use 'training' or 'clustering'.")


if __name__ == "__main__":
    main()
