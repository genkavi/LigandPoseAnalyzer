import os
import argparse
from copy import deepcopy

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.signal import correlate
from scipy.optimize import curve_fit
import pandas as pd

import matplotlib.pyplot as plt

import mdtraj as md

def load_trajectory(xtc_path, pdb_path, stride=10, verbose=False):
    """
    Load a molecular dynamics trajectory using MDTraj.

    Parameters:
    xtc_path (str): Path to the XTC file.
    pdb_path (str): Path to the PDB file (topology).
    stride (int): Stride value for loading the trajectory (default is 10).
    verbose (bool): If True, prints details of the loaded trajectory (default is False).

    Returns:
    traj (md.Trajectory): The loaded trajectory object.
    """
    try:
        traj = md.load(xtc_path, top=pdb_path, stride=stride)
        if verbose:
            print(f"Loaded trajectory: {xtc_path} with topology: {pdb_path}")
            print(f"Trajectory details: {traj}")
        return traj
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return None

def compute_shared_contact_matrix(traj, contact_cutoff=0.5):
    """
    Compute a shared contact matrix based on protein-ligand contacts.

    Parameters:
    traj (md.Trajectory): The trajectory object.
    contact_cutoff (float): Distance cutoff for defining contacts (default is 0.5 nm).

    Returns:
    shared_contact_matrix (np.ndarray): The shared contact matrix.
    """
    # Select protein and ligand atoms
    ligand_atoms = traj.topology.select('resname LIG and not element H')  # Adjust 'LIG' to your ligand's residue name
    protein_atoms = traj.topology.select('protein and not element H')

    # Define pairs of protein-ligand atoms to compute distances
    protein_ligand_pairs = traj.topology.select_pairs(protein_atoms, ligand_atoms)

    # Compute distances between protein-ligand atom pairs across all frames
    distances = md.compute_distances(traj, protein_ligand_pairs)

    # Create a matrix to store the shared contact distance between frames
    shared_contact_distance_matrix = np.zeros((traj.n_frames, traj.n_frames))

    for i in range(traj.n_frames):
        for j in range(i):
            dist_i = distances[i]
            dist_j = distances[j]

            # Determine which atom pairs are in contact in both frames
            close_pairs_i = dist_i < contact_cutoff
            close_pairs_j = dist_j < contact_cutoff

            # Find the number of contacts shared between both frames
            shared_contacts = np.sum(close_pairs_i & close_pairs_j)

            # Avoid division by zero by setting shared_contacts to 1 if none are found
            if shared_contacts == 0:
                shared_contacts = 1

            # Store the inverse of shared contacts as the "distance"
            shared_contact_distance_matrix[i, j] = shared_contacts
            shared_contact_distance_matrix[j, i] = shared_contact_distance_matrix[i, j]  # Symmetric matrix

    return shared_contact_distance_matrix


def plot_shared_contact_matrix(matrix, output_file="shared_contact_matrix.png"):
    """
    Plot and save the shared contact matrix.

    Parameters:
    matrix (np.ndarray): The shared contact matrix.
    output_file (str): The file path to save the plot.
    """

    plt.imshow(matrix, cmap="viridis")
    plt.colorbar(label="Shared Contact Distance")
    plt.title("Shared Contact Distance Matrix")
    plt.xlabel("Frame")
    plt.ylabel("Frame")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Plot saved to {output_file}")


def compute_contact_clusters(matrix, min_elements=10, distance_threshold=0.4):
    """
    Process a single shared contact distance matrix and perform clustering.

    Parameters:
    matrix (np.ndarray): The shared contact distance matrix.
    min_elements (int): Minimum number of elements a cluster should have (default is 10).
    distance_threshold (float): Distance threshold for hierarchical clustering (default is 0.4).

    Returns:
    filtered_cluster_labels (np.ndarray): Array of filtered cluster labels.
    sorted_matrix (np.ndarray): The sorted matrix after clustering.
    sorted_indices (np.ndarray): Indices used for sorting the matrix.
    sorted_cluster_labels (np.ndarray): Cluster labels sorted based on indices.
    """

    ### Converting the shared contact matrix to a distance matrix for clustering ###
    # Normalize and apply sigmoid transformation to the matrix
    matrix = deepcopy(matrix)
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    mid = 0.5
    k = 10
    matrix = 1 - 1 / (1 + np.exp(-k * (matrix - mid)))
    np.fill_diagonal(matrix, 0)

    # Ensure the matrix is symmetric
    assert np.allclose(matrix, matrix.T), "Matrix is not symmetric"

    # Sample IDs associated with each row/column
    sample_ids = np.array([f'Sample{i}' for i in range(matrix.shape[0])])

    ### Grouping like lements together using hierarchical clustering ###
    # Perform hierarchical clustering on the matrix
    Z = linkage(squareform(matrix), method='single', metric="correlation")

    # Get the order of the rows/columns based on the hierarchical clustering
    sorted_indices = leaves_list(Z)

    # Apply the same sorting to rows and columns to preserve symmetry
    sorted_matrix = matrix[sorted_indices, :][:, sorted_indices]


    ### Finding cluster boundaries ###
    # Set the threshold at the distance just before the largest jump
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # Count the number of elements in each cluster
    cluster_counts = np.bincount(cluster_labels)

    # Find clusters that have at least min_elements
    valid_clusters = np.where(cluster_counts >= min_elements)[0]

    # Filter out clusters with fewer than min_elements
    filtered_cluster_labels = np.array([label if label in valid_clusters else -1 for label in cluster_labels])

    # Map valid clusters to contiguous indices, keeping -1 unchanged
    unique_clusters = sorted(set(filtered_cluster_labels) - {-1})
    cluster_mapping = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    filtered_cluster_labels = np.array([cluster_mapping[label] if label != -1 else -1 for label in filtered_cluster_labels])

    # Initialize a list to store centroids for the current trajectory
    centroids = []

    # Loop through valid cluster labels (exclude -1)
    for cluster in np.unique(filtered_cluster_labels):
        if cluster == -1:
            continue  # Skip invalid clusters

        # Get the indices of samples belonging to the current cluster
        cluster_indices = np.where(filtered_cluster_labels == cluster)[0]
        
        # Calculate the centroid of the cluster as the mean of the corresponding rows/columns
        cluster_matrix = matrix[cluster_indices, :][:, cluster_indices]
        centroid = np.mean(cluster_matrix, axis=0)
        
        # Store the centroid and the corresponding sample IDs
        centroids.append((cluster, sample_ids[cluster_indices], centroid))
  
    # Reorder cluster labels based on the sorted indices
    sorted_cluster_labels = filtered_cluster_labels[sorted_indices]

    return filtered_cluster_labels, sorted_matrix, sorted_indices, sorted_cluster_labels, centroids


def plot_clustered_matrix(sorted_matrix, sorted_cluster_labels, centroids, output_file="sorted_shared_contact_matrix_with_clusters.png"):
    """
    Plot the sorted shared contact distance matrix with cluster boundaries.

    Parameters:
    sorted_matrix (np.ndarray): The sorted shared contact distance matrix.
    sorted_cluster_labels (np.ndarray): Cluster labels sorted based on indices.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Plot the sorted matrix
    plt.figure()
    plt.imshow(sorted_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.clim(0.5, 1)

    # Identify the boundaries between clusters
    boundaries = np.where(np.diff(sorted_cluster_labels))[0]

    # Plot rectangles (boxes) around each cluster
    for boundary_start, boundary_end in zip([0] + list(boundaries + 1), list(boundaries + 1) + [sorted_matrix.shape[0]]):
        size = boundary_end - boundary_start
        cluster_label = sorted_cluster_labels[boundary_start]  # Get the label of the current cluster
        
        # Skip drawing rectangles for clusters with label -1 (removed clusters)
        if cluster_label == -1:
            continue
        
        # Draw the rectangle around each valid cluster
        plt.gca().add_patch(plt.Rectangle((boundary_start, boundary_start), size, size, 
                                          fill=False, edgecolor='red', lw=2))

    # Mark centroids on the plot
    for cluster, sample_ids_in_cluster, centroid in centroids:
        # Get the indices of the samples in the cluster from the sorted order
        sorted_cluster_indices = np.where(sorted_cluster_labels == cluster)[0]
        
        # Mark the centroid of the cluster (mean of its positions in the sorted order)
        centroid_position = np.mean(sorted_cluster_indices).astype(int)
        
        # Mark the centroid with a white circle
        plt.plot(centroid_position, centroid_position, 'wo', markersize=5, markeredgewidth=2)

    plt.title("Sorted Shared Contact Distance Matrix with Clusters")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Plot saved to {output_file}") 



def save_cluster_trajectories_and_centroids(trajectory, cluster_labels, output_dir='output_trajectories'):
    """
    Write XTC files for each cluster containing only protein and ligand atoms.
    Also writes a single PDB trajectory containing the centroid structures.

    Parameters:
    trajectory (mdtraj.Trajectory): The trajectory object.
    cluster_labels (np.ndarray): Array of cluster labels for each frame.
    output_dir: The directory where the output files will be saved.
    """

    # Superpose trajectory using only the protein backbone (CA atoms)
    protein = trajectory.topology.select("name CA")
    trajectory.superpose(trajectory, frame=0, atom_indices=protein, parallel=True)

    centroid_frames = []  # To collect centroid frames

    # Loop over unique cluster IDs (excluding -1)
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue  # Skip invalid clusters

        # Extract the frames belonging to the current cluster
        cluster_frames = np.where(cluster_labels == cluster_id)[0]

        # Slice the trajectory to include only the frames in the chosen cluster
        cluster_traj = trajectory.slice(cluster_frames)

        # Select only protein and ligand atoms (adjust 'LIG' for your ligand's residue name)
        protein_ligand_atoms = cluster_traj.topology.select('(protein and not element H) or resname LIG')

        # Slice the trajectory again to include only protein and ligand atoms
        protein_ligand_traj = cluster_traj.atom_slice(protein_ligand_atoms)

        # Save the XTC file for the cluster
        xtc_filename = os.path.join(output_dir, f"cluster_{cluster_id}.xtc")
        protein_ligand_traj.save_xtc(xtc_filename)

        # Find the frame closest to the centroid for this cluster
        centroid = np.mean(trajectory.xyz[cluster_frames], axis=0)  # Calculate the centroid of the cluster
        closest_frame_idx = cluster_frames[np.argmin(np.linalg.norm(trajectory.xyz[cluster_frames] - centroid, axis=(1, 2)))]

        # Append the closest centroid frame to the list
        centroid_frames.append(closest_frame_idx)

    # Combine all centroid frames into a single trajectory
    centroid_traj = trajectory.slice(centroid_frames)

    # Select protein and ligand atoms for the centroid frames
    protein_ligand_atoms = centroid_traj.topology.select('(protein and not element H) or resname LIG')
    protein_ligand_centroid_traj = centroid_traj.atom_slice(protein_ligand_atoms)

    # Save the PDB trajectory containing the centroids
    pdb_filename = os.path.join(output_dir, "centroids.pdb")
    protein_ligand_centroid_traj.save_pdb(pdb_filename)
    print(f"Cluster centroids and members saved.") 

def autocorrelation(x):
    """
    Compute the autocorrelation of a 1D array using FFT.
    """
    result = correlate(x, x, mode='full')
    result = result[result.size // 2:]  # Take only the positive lags
    result = result.astype(float)  # Ensure the result is float to avoid type issues
    result /= result[0]  # Normalize the autocorrelation
    return result

def exponential_decay(t, tau):
    """
    Exponential decay function: exp(-t / tau), where tau is the lifetime.
    """
    return np.exp(-t / tau)

def fit_exponential_decay(acf):
    """
    Fit the autocorrelation function to an exponential decay model to estimate the lifetime (tau).
    Returns the fitted tau and the covariance matrix.
    """
    t = np.arange(len(acf))  # Lag times (x-axis)
    
    # Initial guess for tau (decay constant)
    initial_guess = [1.0]
    
    # Fit the exponential decay model
    params, covariance = curve_fit(exponential_decay, t, acf, p0=initial_guess)
    
    return params[0], covariance  # Return the fitted tau (lifetime)


def calculate_cluster_lifetime(filtered_cluster_labels, output_dir):
    """
    Calculate the lifetime of each cluster for a single trajectory by fitting
    an exponential decay model to the autocorrelation function.

    Parameters:
    trajectory (mdtraj.Trajectory): The trajectory object.
    filtered_cluster_labels (np.ndarray): Array of cluster labels for each frame in the trajectory.

    Returns:
    cluster_lifetimes (dict): Dictionary containing lifetimes for each cluster.
    """
    cluster_lifetimes = {}

    # Loop over unique cluster IDs (excluding -1)
    for cluster_id in np.unique(filtered_cluster_labels):
        if cluster_id == -1:
            continue  # Skip invalid clusters

        # Create a binary time series where the cluster is 1 when it's in the current cluster and 0 otherwise
        cluster_time_series = (filtered_cluster_labels == cluster_id).astype(int)

        # Calculate the autocorrelation of the time series
        acf = autocorrelation(cluster_time_series)

        # Fit the exponential decay model to the ACF to get the lifetime (tau)
        tau, _ = fit_exponential_decay(acf)

        # Store the estimated lifetime for the current cluster
        cluster_lifetimes[cluster_id] = tau

        # Plot the autocorrelation function (ACF) and the fitted exponential decay
        t = np.arange(len(acf))  # Lag times
        fitted_acf = exponential_decay(t, tau)

        plt.plot(t, acf, lw=1, label=f'Cluster {cluster_id} ACF')
        plt.plot(t, fitted_acf, linestyle='--', color='red', label=f'Fitted Exp Decay (Lifetime: {tau:.2f} ns)')
        plt.title(f'Lifetime - Cluster {cluster_id}')
        plt.xlabel('Lag (frames)')
        plt.ylabel('Autocorrelation')
        plt.legend()
        plt.savefig(f"{output_dir}/autocorrelation_cluster_{cluster_id}.png", dpi=300)
        plt.close()
        print(f"Life time plot saved to {output_dir}/autocorrelation_cluster_{cluster_id}.png") 

    return cluster_lifetimes


def report_cluster_lifetimes(cluster_lifetimes, output_dir):
    """
    Generate and save a report of cluster lifetimes.

    Parameters:
    cluster_lifetimes (dict): Dictionary containing lifetimes for each cluster.
    output_dir (str): Directory where the report will be saved.

    Returns:
    None
    """
    # Convert the dictionary to a DataFrame
    lifetime_df = pd.DataFrame(list(cluster_lifetimes.items()), columns=["Cluster ID", "Lifetime (ns)"])

    # Save the DataFrame as a CSV file
    csv_filename = os.path.join(output_dir, "cluster_lifetimes.csv")
    lifetime_df.to_csv(csv_filename, index=False)
    print(f"Cluster lifetime report saved to {csv_filename}")

    # Print the DataFrame for quick review
    print("\nCluster Lifetime Report:")
    print(lifetime_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Process a single trajectory and save clusters and centroids.")
    parser.add_argument("xtc_path", type=str, help="Path to the XTC file.")
    parser.add_argument("pdb_path", type=str, help="Path to the PDB file (topology).")
    parser.add_argument("--stride", type=int, default=10, help="Stride value for loading the trajectory (default is 10).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--contact_cutoff", type=float, default=0.5, help="Distance cutoff for defining contacts (default is 0.5 nm).")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save all output files.")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the trajectory
    trajectory = load_trajectory(args.xtc_path, args.pdb_path, stride=args.stride, verbose=args.verbose)
    if trajectory is not None:
        # Compute the shared contact distance matrix
        shared_contact_distance_matrix = compute_shared_contact_matrix(trajectory, contact_cutoff=args.contact_cutoff)
        print(f"Shared contact distance matrix computed with shape: {shared_contact_distance_matrix.shape}")

        # Plot and save the shared contact matrix
        shared_contact_matrix_out = os.path.join(args.output_dir, "shared_contact_distance.png")
        plot_shared_contact_matrix(shared_contact_distance_matrix, output_file=shared_contact_matrix_out)

        # Compute clusters and save the results
        filtered_cluster_labels, sorted_matrix, sorted_indices, sorted_cluster_labels, centroids = compute_contact_clusters(
            shared_contact_distance_matrix, min_elements=10, distance_threshold=0.4
        )
        sorted_shared_contact_distance_out = os.path.join(args.output_dir, "sorted_shared_contact_distance.png")
        plot_clustered_matrix(sorted_matrix, sorted_cluster_labels, centroids, output_file=sorted_shared_contact_distance_out)

        # Save clustered trajectories and centroid structures
        save_cluster_trajectories_and_centroids(trajectory, filtered_cluster_labels, output_dir=args.output_dir)

        # Life time analysis
        cluster_lifetimes = calculate_cluster_lifetime(filtered_cluster_labels, output_dir=args.output_dir)      
        report_cluster_lifetimes(cluster_lifetimes, output_dir=args.output_dir)
        

if __name__ == "__main__":
    main()