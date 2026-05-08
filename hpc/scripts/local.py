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

# 1. add_isonoise: add isotropic Gaussian noise to a point cloud
def add_isonoise(V, sigma = 0.0001, seed = 231299):
    rng = np.random.RandomState(seed)
    Vn = V + rng.normal(0, sigma, size = V.shape[0]*3).reshape(-1, 3)
    return Vn

# 2. compute_nominal_texture: compute nominal texture 
def compute_nominal_texture(V, axis_texture = 1, scaling_factor = 10.0, centering = True):
    coords = V[:, axis_texture]
    if centering:
        center = 0.5 * (coords.min() + coords.max())
        half_range = 0.5 * (coords.max() - coords.min())
        nominal_texture = scaling_factor * (1 - np.abs(coords - center) / half_range)
    else:
        nominal_texture = (coords - coords.min()) / (coords.max() - coords.min()) * scaling_factor
    return nominal_texture

def add_isonoise_local(V, mask, sigmaic = 0.0001, sigmaoc = 0.0003, seed = 231299):
    rng = np.random.RandomState(seed)
    size_oc = mask.sum()
    size_ic = V.shape[0] - size_oc
    Vn = V.copy()
    noise_ic = rng.normal(0, sigmaic, size = size_ic*3).reshape(-1, 3)
    noise_oc = rng.normal(0, sigmaoc, size = size_oc*3).reshape(-1, 3)
    Vn[~mask] += noise_ic
    Vn[mask] += noise_oc
    return Vn 

def compute_mask(v):
    return (v[:,1] > 0.14) & (v[:,0] < -0.06) & (v[:,2] < -0.015)

# generate_pcd_local: generate a point cloud with random subsampling and noise + local noise
def generate_pcd_local(V_original, T_nominal, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seed):
    rng = np.random.RandomState(seed)
    V_i = V_original.copy()
    min_s = V_i.shape[0] - max_size
    max_s = V_i.shape[0] - min_size
    howmany = rng.randint(min_s, max_s + 1) 
    indices = rng.choice(V_i.shape[0], size = int(len(V_i) - howmany), replace = False)
    V_i = V_i[indices]
    mask_i = compute_mask(V_i)
    V_i = add_isonoise_local(V_i, mask_i, sigmaic = sigma_shape, sigmaoc = sigma_oc_shape, seed = seed)
    T_sub = T_nominal[indices]
    T_i = T_sub + rng.normal(0, sigma_texture, size = T_sub.shape)
    T_i = np.clip(T_i, 0, None)
    return V_i, T_i

# utils functions for graph laplacian
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

def process_pcd_gl_local(args):
    V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seed, neig, method = args
    V_i, T_i = generate_pcd_local(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seed)
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

# process_pcd: generate and process a single point cloud using lb
def process_pcd_local(args):
    V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seed, neig, sigma_eigsh, method = args
    V_i, T_i = generate_pcd_local(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seed)
    if method == 'lb':
        L, M = robust_laplacian.point_cloud_laplacian(V_i)
        evals, evecs = eigsh(L, neig, M, sigma=sigma_eigsh)
        coeff = evecs.T @ (M @ T_i)
    else:
        raise ValueError(f"Unknown method: {method}")
    return evals, coeff

# process_batch_pcds: generate and process a batch of point clouds
def process_batch_pcds(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seeds, neig, sigma_eigsh, method, n_jobs):
    args_list = [(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seed, neig, sigma_eigsh, method) for seed in seeds]
    results = Parallel(n_jobs=n_jobs)(delayed(process_pcd_local)(args) for args in args_list)
    evals_list, coeffs_list = zip(*results)
    return np.array(evals_list), np.array(coeffs_list)

def process_batch_pcds_gl(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seeds, neig, method, n_jobs):
    args_list = [(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seed, neig, method) for seed in seeds]
    results = Parallel(n_jobs=n_jobs)(delayed(process_pcd_gl_local)(args) for args in args_list)
    return np.array(results)


def main():
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Compute Robust Laplacian, Eigenvalues and Coefficients for texture.')
        parser.add_argument('--outputFolder', type=str, required=True, help='Output folder')
        parser.add_argument('--inputFolderCAD', type=str, required=True, help='Input folder')
        parser.add_argument('--neig', type=int, required=True, help='N.eigenvalues')
        parser.add_argument('--method', type=str, required=True, help='Method: lb or gl')
        parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
        parser.add_argument('--id', type=int, required=True, help='Index')
        parser.add_argument('--nbatches', type=int, default=6, help='Number of batches (default: 6)')
        parser.add_argument('--size', type=int, default=1000, help='Batch size (default: 1000)')
        parser.add_argument('--date', type=str, required=True, help='Date for encoding')
        parser.add_argument('--no', type=str, required=True, help='Number for encoding')
        return parser.parse_args()
    
    args = parse_arguments()
    print(f"Date: {args.date} and No: {args.no}")
    print("Starting processing excess of material")

    method = args.method
    input_folder_CAD = args.inputFolderCAD
    print(f"Input folder CAD excess: {input_folder_CAD}")
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
    main_seed = args.id + 100  # use id as main seed; add 100 to ensure different from other scripts
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


    snr = np.array([1.15, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75])
    assert len(snr) == nbatches, "Length of snr array must match number of batches"
    init = 0
    end = 6
    if method == 'lb':  
        for j in range(init, end):
            print(f"\nProcessing batch {j+1}/{nbatches} with seed {batch_seeds[j]}")
            sigma_oc_shape = sigma_shape * snr[j]
            print(f"Using sigma_oc_shape: {sigma_oc_shape}")
            start_time = time.time()
            seeds_j = all_batch_seeds[j]
            evals_j, coeffs_j = process_batch_pcds(V, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seeds_j, neig, sigma_eigsh, method, n_jobs=args.n_jobs)
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
            sigma_oc_shape = sigma_shape * snr[j]
            print(f"Using sigma_oc_shape: {sigma_oc_shape}")
            start_time = time.time()
            seeds_j = all_batch_seeds[j]
            evals_j = process_batch_pcds_gl(V, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, seeds_j, neig, method, n_jobs=args.n_jobs)
            end_time = time.time()
            np.save(os.path.join(output_folder_evals, f'evals_{j+1}.npy'), evals_j)
            print(f"Saved evals for batch {j+1} to {output_folder}")
            elapsed = end_time - start_time
            print(f"Batch {j+1} completed in {elapsed:.2f}s ({elapsed/batch_size:.3f}s per PCD)")
            gc.collect()
        time.sleep(1)
    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == '__main__':
    main()
