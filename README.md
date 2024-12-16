# LigandPoseAnalyzer

**LigandPoseAnalyzer** is a Python tool for molecular dynamics trajectory analysis, specifically focusing on protein-ligand interactions. It provides functionalities for:
- **Shared contact matrix computation**
- **Clustering based on shared contact distances**
- **Cluster lifetime analysis**
- **Visualization of shared contact matrices and cluster lifetimes**
- **Saving clustered trajectories and centroid structures**

---

## Installation

This project requires Python and the following dependencies:
- `mdtraj`
- `numpy`
- `matplotlib`
- `scipy`
- `pandas`

### Using Conda
Create a Conda environment and install the required dependencies:
```bash
conda create -n LigandPoseAnalyzer mdtraj matplotlib scipy pandas -y
conda activate LigandPoseAnalyzer
```

## Usage

The tool is contained in a single Python file: `LigandPoseAnalyzer.py`. To use it, provide the input trajectory (`.xtc`) and topology (`.pdb`) files, along with optional arguments for customization.

### Example Command
```bash
python LigandPoseAnalyzer.py trajectory.xtc topology.pdb --stride 10 --contact_cutoff 0.5 --output_dir output
```

### Arguments
- `xtc_path`: Path to the `.xtc` trajectory file.
- `pdb_path`: Path to the `.pdb` topology file.
- `--stride`: Stride value for loading the trajectory (default: 10).
- `--verbose`: Enable verbose output for debugging (default: `False`).
- `--contact_cutoff`: Distance cutoff for defining contacts (default: 0.5 nm).
- `--output_dir`: Directory to save all output files (default: `output`).

## Output

The tool generates the following output in the specified directory (`--output_dir`):

1. **Shared Contact Distance Matrix**
   - File: `shared_contact_distance.png`
   - A heatmap of the shared contact distances between trajectory frames.

2. **Sorted Shared Contact Matrix**
   - File: `sorted_shared_contact_distance.png`
   - A heatmap of the contact matrix sorted by clusters.

3. **Clustered Trajectories**
   - Files: `cluster_<cluster_id>.xtc` for each cluster.
   - XTC files containing frames belonging to individual clusters.

4. **Centroid Structures**
   - File: `centroids.pdb`
   - A PDB file containing centroid structures for all clusters.

5. **Cluster Lifetimes**
   - File: `cluster_lifetimes.csv`
   - A CSV file summarizing the lifetime (in ns) of each cluster.

6. **Lifetime Plots**
   - Files: `autocorrelation_cluster_<cluster_id>.png`
   - Plots of the autocorrelation function and exponential decay fit for each cluster.

## Features

1. **Shared Contact Matrix**
   - Calculates the overlap of protein-ligand contacts across trajectory frames.

2. **Clustering**
   - Groups frames based on shared contacts using hierarchical clustering.

3. **Lifetime Analysis**
   - Computes the lifetime of each cluster using autocorrelation and exponential decay fitting.

4. **Visualization**
   - Generates heatmaps and lifetime plots for visualizing the results.

5. **Trajectory Export**
   - Exports individual cluster trajectories and centroid structures for further analysis.

## Methods


### Shared Contacts
Shared contacts $S(i, j)$ between trajectory frames are computed as:

$$
S(i, j) = \sum_{k=1}^N \left( \mathbb{I}(d_{i,k} < \text{cutoff}) \land \mathbb{I}(d_{j,k} < \text{cutoff}) \right)
$$

where:
- $d_{i,k}$ and $d_{j,k}$ are the distances between atom $ k $ and the respective trajectory frames $i$ and $j$.
- $\mathbb{I}$ is an indicator function that equals 1 if the condition is satisfied (contact within the cutoff distance) and 0 otherwise.
- $N$ is the total number of atom pairs.

---

### Distance Metric
The pairwise distance $D(i, j)$ between trajectory frames $i$ and $j$ is defined as:

$$
D(i, j) = 1 - \frac{1}{1 + e^{-k \cdot (\text{Normalized Similarity}(i,j) - \text{mid})}}
$$

where:
- $k$ is a steepness factor controlling the sigmoid curve.
- **Normalized Similarity** represents normalized $S(i, j)$  between two frames.

---

### Clustering
Hierarchical clustering is used to analyze the shared contact matrix:
1. **Linkage method:** "Single" linkage, which connects clusters based on the minimum distance between frames.
2. **Metric:** "Correlation" metric to evaluate similarity between trajectory frames.

The clustering process serves two purposes:
1. **Sorting** the distance matrix to reveal block structures corresponding to clusters.
2. **Identifying clusters and centroids**, where centroid frames are the most representative structures within each cluster.

