# MOSAIC — Multi-Omics Single-cell Analysis Integration Core

MOSAIC is a general-purpose multi-omics dimensionality reduction and clustering analysis framework with two core modules:

- **Training Module**: Multi-view dimensionality reduction based on a GCN autoencoder, learning low-dimensional embeddings for each view via graph convolutional networks with contrastive learning and reconstruction loss.
- **Clustering Module**: Multi-omics clustering via iterative affinity-based network fusion, using local-scale affinity matrices and similarity network fusion to fuse and cluster the reduced embeddings.

---

## Directory Structure

```
MOSAIC/
├── main.py              # Unified entry-point script
├── train.py             # Training module (GCN autoencoder)
├── models.py            # GCN encoder / MLP decoder model definitions
├── utils.py             # Utility functions (graph construction, distance, affinity, clustering evaluation)
├── fusion.py            # Network fusion core algorithm (iterative SNF)
├── README.md            # This document
├── input/               # Input file directory (create manually)
├── fea/                 # Intermediate feature output directory (auto-created)
├── loss_png/            # Loss curve plot output directory (auto-created)
└── analysis/results/    # Clustering result output directory (auto-created)
```

---

## Dependencies

Python 3.8+ is recommended. Required core libraries and suggested versions:

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Numerical computation |
| `pandas` | ≥ 1.3 | Data I/O |
| `torch` (PyTorch) | ≥ 1.10 | GCN model training (CUDA support accelerates) |
| `scikit-learn` | ≥ 1.0 | KMeans / spectral clustering / evaluation metrics |
| `matplotlib` | ≥ 3.5 | Loss curve plotting |
| `scipy` | ≥ 1.7 | Sparse matrix support |

Installation:

```bash
pip install numpy pandas torch scikit-learn matplotlib scipy
```

---

## Input Data Preparation

### 1. Input List File

Create a `.list` file under `./input/` with one **absolute path** per line to each omics view data file. Example `./input/dataset1.list`:

```
/path/to/data/omics1.csv
/path/to/data/omics2.csv
/path/to/data/omics3.csv
```

> CSV file format: **rows = features, columns = samples** (each column is a sample, each row is a feature/gene).

### 2. True Label File (optional, for clustering evaluation only)

CSV format with a sample label column. Example:

```csv
sample,label
sample_1,1
sample_2,2
...
```

---

## Usage Instructions

### Mode 1: Training (Dimensionality Reduction)

GCN autoencoder training to learn low-dimensional embeddings for each view.

```bash
python main.py -t my_dataset -i ./input/my_dataset.list -m training -n 3 --num_views 2
```

**Parameter Description:**

| Parameter | Description | Default |
|------|------|--------|
| `-t` | Dataset name, used for output file naming | `dataset` |
| `-i` | Path to input list file | `./input/input.list` |
| `-m` | Run mode: `training` | `training` |
| `-n` | Number of clusters (required) | `-1` (must specify) |
| `--num_views` | Number of data views | `2` |

**Output Files:**

| Path | Content |
|------|------|
| `./fea/{dataset_name}.fea` | Mean embedding across all views (TSV format) |
| `./fea/_view1.csv`, `_view2.csv`, ... | Per-view independent embeddings |
| `./loss_png/{dataset_name}G1_show.png`, etc. | Loss curve plots |

### Mode 2: Clustering (Fusion & Clustering)

Network fusion and clustering on the reduced embeddings.

```bash
python main.py -t my_dataset -i ./input/my_dataset.list -m clustering -n 3 --num_views 2
```

**Clustering-Specific Parameters:**

| Parameter | Description | Default |
|------|------|--------|
| `--aff_k` | Number of nearest neighbors for affinity matrix | `18` |
| `--fusion_k` | Number of nearest neighbors for network fusion | `42` |
| `--niter` | Maximum iterations for network fusion | `30` |
| `--beta` | Fusion update rate (0~1) | `0.8` |
| `--label_file` | Path to true label CSV (for evaluation) | None |
| `--label_col` | Column name for true labels | `label` |

**Output Files:**

| Path | Content |
|------|------|
| `./analysis/results/{dataset_name}.mosiac` | Clustering label results |

### Complete Workflow Example

```bash
# Step 1: Dimensionality reduction
python main.py -t my_dataset -i ./input/my_dataset.list -m training -n 3 --num_views 2

# Step 2: Fusion & clustering (with evaluation)
python main.py -t my_dataset -i ./input/my_dataset.list -m clustering -n 3 --num_views 2 \
    --label_file ./data/true_labels.csv --label_col label
```

---

---

## Parameter Tuning Suggestions

| Parameter | Suggestion |
|------|------|
| `-n` | Determine based on prior knowledge or the elbow method |
| `--num_views` | Should match the number of data files in the `.list` file |
| `--aff_k` | Typically 5%~15% of sample count; default 18 |
| `--fusion_k` | Generally larger than `aff_k`; default 42 |
| `--niter` | Increase for better fusion quality at the cost of runtime |
| `--beta` | 0.6~0.9; larger values give larger per-iteration updates |
