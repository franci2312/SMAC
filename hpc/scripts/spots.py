import os
import argparse
import time
import gc
import robust_laplacian
from joblib import Parallel, delayed
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse import diags

def add_isonoise(V, sigma = 0.0001, seed = 231299):
    rng = np.random.RandomState(seed)
    Vn = V + rng.normal(0, sigma, size = V.shape[0]*3).reshape(-1, 3)
    return Vn

# 2. compute_nominal_texture: compute nominal texture (do this once and then modify it in place for each pcd)
def compute_nominal_texture(V, axis_texture = 1, scaling_factor = 10.0, centering = True):
    coords = V[:, axis_texture]
    if centering:
        center = 0.5 * (coords.min() + coords.max())
        half_range = 0.5 * (coords.max() - coords.min())
        nominal_texture = scaling_factor * (1 - np.abs(coords - center) / half_range)
    else:
        nominal_texture = (coords - coords.min()) / (coords.max() - coords.min()) * scaling_factor
    return nominal_texture

# 3. generate_pcd: generate a point cloud with random subsampling and noise + color spots
def generate_pcd_color(V_original, T_nominal, min_size, max_size, sigma_shape, sigma_texture, percentage, color, seed):
    rng = np.random.RandomState(seed)
    V_i = V_original.copy()
    min_s = V_i.shape[0] - max_size
    max_s = V_i.shape[0] - min_size
    howmany = rng.randint(min_s, max_s + 1) 
    indices = rng.choice(V_i.shape[0], size = int(len(V_i) - howmany), replace = False)
    V_i = V_i[indices]
    V_i = add_isonoise(V_i, sigma = sigma_shape, seed = seed)
    T_sub = T_nominal[indices]
    T_i = T_sub + rng.normal(0, sigma_texture, size = T_sub.shape)
    mask = compute_mask_spots(V_i)
    T_i = pores_spacing(V_i, T_i, mask, percentage, color, scaling_factor = 10.0, seed = seed)
    T_i = np.clip(T_i, 0, None)
    return V_i, T_i

# functions for graph laplacian
def compute_graph_laplacian_sparse(adjacency_matrix):
    """Compute graph Laplacian from adjacency matrix"""
    if not isinstance(adjacency_matrix, csr_matrix):
        raise ValueError("adjacency_matrix must be a scipy sparse csr_matrix")
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()  
    D = diags(degrees)  
    L = D - adjacency_matrix
    return L

def compute_adjacency_matrix_sparse(points, n_neighbors=30, gaussian=True):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    rows, cols, data = [], [], []
    for i in range(points.shape[0]):
        for j in range(1, n_neighbors):  
            rows.append(i)
            cols.append(indices[i, j])
            if gaussian:
                data.append(np.exp(-0.5*distances[i, j]**2))
            else:
                data.append(1/(distances[i, j] + 1e-9))
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(points.shape[0], points.shape[0]))
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    return adj_matrix


def pores_spacing(v, t, mask, percentage, color=10.0, scaling_factor = 10.0, seed=123):
    # given v, compute the nearest neighbor for each point to set min_dist
    nbrs = NearestNeighbors(n_neighbors=2).fit(v)
    distances,_ = nbrs.kneighbors(v)
    min_dist = np.median(distances[:, 1]) * 2  # multiply for visible separation
    valid_indices = np.where(mask)[0]
    npoints = int(percentage * v.shape[0])
    
    rng = np.random.RandomState(seed)
    sampled_indices = []
    
    # Start with random point
    first_idx = rng.choice(valid_indices)
    sampled_indices.append(first_idx)

    # Iteratively add points with distance constraint
    max_attempts = npoints * 500
    attempts = 0
    while len(sampled_indices) < npoints and attempts < max_attempts:
        candidate_idx = rng.choice(valid_indices)
        candidate_pos = v[candidate_idx]
        
        # Check distance to already sampled points
        distances = np.linalg.norm(v[sampled_indices] - candidate_pos, axis=1)
        if min_dist <= np.min(distances) <= min_dist * 2:
            sampled_indices.append(candidate_idx)
        attempts += 1

    # If max attempts reached, fill remaining with random sampling
    if len(sampled_indices) < npoints:
        remaining = npoints - len(sampled_indices)
        unsampled = np.setdiff1d(valid_indices, sampled_indices)
        if len(unsampled) >= remaining:
            additional = rng.choice(unsampled, size=remaining, replace=False)
            sampled_indices.extend(additional)
            print(f"Added {remaining} random points to reach {npoints} total.")
        else:
            print(f"Warning: mask is too small to sample the requested number of points.")
    new_texture = t.copy()
    new_texture[sampled_indices] += color 
    new_texture = np.clip(new_texture, 0, None)
    return new_texture

def process_pcd(args):
    V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seed, neig, sigma_eigsh, method, percentage, color = args
    V_i, T_i = generate_pcd_color(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, percentage, color, seed)
    if method == 'lb':
        L, M = robust_laplacian.point_cloud_laplacian(V_i)
        evals, evecs = eigsh(L, neig, M, sigma=sigma_eigsh)
        coeff = evecs.T @ (M @ T_i)
    else:
        raise ValueError(f"Unknown method: {method}")
    return evals, coeff

def process_pcd_gl_color(args):
    V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seed, neig, method, percentage, color = args
    V_i, T_i = generate_pcd_color(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, percentage, color, seed)
    if method == 'gl':
        V_std = (V_i - V_i.mean(axis=0)) / V_i.std(axis=0)
        T_std = (T_i - T_i.mean()) / T_i.std()
        V_aug_std = np.hstack([V_std, T_std.reshape(-1, 1)])
        A = compute_adjacency_matrix_sparse(V_aug_std, n_neighbors=30, gaussian=True)
        gl = compute_graph_laplacian_sparse(A)
        evals = eigsh(gl, neig, which='SM', return_eigenvectors=False)
        evals = evals[::-1]
    else:
        raise ValueError(f"Unknown method: {method}")
    return evals

def process_batch_pcds(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seeds, neig, sigma_eigsh, method, percentage, color, n_jobs):
    args_list = [(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seed, neig, sigma_eigsh, method, percentage, color) for seed in seeds]
    results = Parallel(n_jobs=n_jobs)(delayed(process_pcd)(args) for args in args_list)
    evals_list, coeffs_list = zip(*results)
    return np.array(evals_list), np.array(coeffs_list)

def process_batch_pcds_gl(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seeds, neig, percentage, color, method, n_jobs):
    args_list = [(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seed, neig, method, percentage, color) for seed in seeds]
    results = Parallel(n_jobs=n_jobs)(delayed(process_pcd_gl_color)(args) for args in args_list)
    return np.array(results)

def compute_mask_spots(v):
    return (v[:,1] > 0.14) & (v[:,0] < -0.06) & (v[:,2] < -0.04)

def main():
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Compute Robust Laplacian, Eigenvalues, Eigenvectors and Coefficients for texture.')
        parser.add_argument('--outputFolder', type=str, required=True, help='Output folder')
        parser.add_argument('--inputFolderCAD', type=str, required=True, help='Input folder')
        parser.add_argument('--neig', type=int, required=True, help='N.eigenvalues')
        parser.add_argument('--method', type=str, required=True, help='Method: lb or gl')
        parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
        parser.add_argument('--id', type=int, required=True, help='Index')
        parser.add_argument('--nbatches', type=int, default=10, help='Number of batches (default: 10)')
        parser.add_argument('--size', type=int, default=1000, help='Batch size (default: 1000)')
        parser.add_argument('--date', type=str, required=True, help='Date for encoding')
        parser.add_argument('--no', type=str, required=True, help='Number for encoding')
        return parser.parse_args()
    
    args = parse_arguments()
    print(f"Date: {args.date} and No: {args.no}")

    method = args.method
    input_folder_CAD = args.inputFolderCAD
    print(f"Input folder CAD: {input_folder_CAD}")
    output_folder = args.outputFolder
    print(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    output_folder_evals = os.path.join(output_folder, 'evals')
    os.makedirs(output_folder_evals, exist_ok=True)
    if method == 'lb':
        output_folder_coeffs = os.path.join(output_folder, 'coeff')
        os.makedirs(output_folder_coeffs, exist_ok=True)

    print(f"Method: {method}")
    neig = args.neig
    print(f"N.eigenvalues: {neig}")
    idx = args.id
    main_seed = idx + 200 
    nbatches = args.nbatches
    batch_size = args.size
    print(f"Run Index: {idx}, Number of batches: {nbatches}, Batch size: {batch_size}")
    main_rng = np.random.default_rng(main_seed)
    batch_seeds = main_rng.integers(low=0, high=1e6, size=nbatches)
    print(f"Main seed: {main_seed}")
    print(f"Generated seeds for {nbatches} sets: {batch_seeds}")
    all_batch_seeds = []
    for batch_seed in batch_seeds:
        rng = np.random.default_rng(int(batch_seed))
        seeds = rng.integers(low=0, high=1e6, size=batch_size)
        all_batch_seeds.append(seeds)
    

    sigma_shape = 0.0001
    sigma_texture = 0.01
    sigma_eigsh = 1e-8

    min_size = 8100
    max_size = 8140
    centering = False
    V = np.load(os.path.join(input_folder_CAD, 'bunny.npy'))  
    print(f"Loaded CAD vertices with shape {V.shape}.")
    nominal_texture = compute_nominal_texture(V, axis_texture=1, scaling_factor=10.0, centering=centering)

    colors = [-0.01, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3] 

    perc = 0.01
    assert len(colors) == nbatches, "Length of colors array must match number of batches"
    init = 0
    end = nbatches - 1
    if method == 'lb':  
        for j in range(init, end):
            print(f"\nProcessing batch {j+1}/{nbatches} with seed {batch_seeds[j]}")
            col = colors[j]
            print(f"Using color for pores spacing: {col}")
            start_time = time.time()
            seeds_j = all_batch_seeds[j]
            evals_j, coeffs_j = process_batch_pcds(V, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seeds_j, neig, sigma_eigsh, method, perc, col, n_jobs=args.n_jobs)
            end_time = time.time()
            np.save(os.path.join(output_folder_evals, f'evals_{j+1}.npy'), evals_j)
            np.save(os.path.join(output_folder_coeffs, f'coeff_{j+1}.npy'), coeffs_j)
            print(f"Saved evals and coeff for batch {j+1} to {output_folder}")
            elapsed = end_time - start_time
            print(f"Batch {j+1} completed in {elapsed:.2f}s ({elapsed/batch_size:.3f}s per PCD)")
            gc.collect()
        time.sleep(1)
    elif method == 'gl':
        for j in range(init, end):
            print(f"\nProcessing batch {j+1}/{nbatches} with seed {batch_seeds[j]}")
            col = colors[j]
            print(f"Using color for pores spacing: {col}")
            start_time = time.time()
            seeds_j = all_batch_seeds[j]
            evals_j = process_batch_pcds_gl(V, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seeds_j, neig, perc, col, method, n_jobs=args.n_jobs)
            end_time = time.time()
            np.save(os.path.join(output_folder_evals, f'evals_{j+1}.npy'), evals_j)
            print(f"Saved evals for batch {j+1} to {output_folder}")
            elapsed = end_time - start_time
            print(f"Batch {j+1} completed in {elapsed:.2f}s ({elapsed/batch_size:.3f}s per PCD)")
            gc.collect()
        time.sleep(1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
if __name__ == "__main__":
    main()
    