import os
import argparse
import time
import gc
import numpy as np
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed

# 1. add_isonoise: add isotropic Gaussian noise to a point cloud
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

def compute_mask_spots(v):
    return (v[:,1] > 0.14) & (v[:,0] < -0.06) & (v[:,2] < -0.04)

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

# generate_pcd_local_color: generate a point cloud with random subsampling and noise + local noise + color spots
def generate_pcd_local_color(V_original, T_nominal, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, percentage, color, seed):
    rng = np.random.RandomState(seed)
    V_i = V_original.copy()
    min_s = V_i.shape[0] - max_size
    max_s = V_i.shape[0] - min_size
    howmany = rng.randint(min_s, max_s + 1) # howmany points to remove
    indices = rng.choice(V_i.shape[0], size = int(len(V_i) - howmany), replace = False)
    V_i = V_i[indices]
    mask_i = compute_mask(V_i)
    V_i = add_isonoise_local(V_i, mask_i, sigmaic = sigma_shape, sigmaoc = sigma_oc_shape, seed = seed)
    T_sub = T_nominal[indices]
    T_i = T_sub + rng.normal(0, sigma_texture, size = T_sub.shape)
    mask = compute_mask_spots(V_i)
    T_i = pores_spacing(V_i, T_i, mask, percentage, color, scaling_factor = 10.0, seed = seed)
    T_i = np.clip(T_i, 0, None)
    return V_i, T_i


def CMDS(X):
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    D_sq = X_sq + X_sq.T - 2 * np.dot(X, X.T)
    row_means = np.mean(D_sq, axis=1, keepdims=True)
    col_means = np.mean(D_sq, axis=0, keepdims=True)
    grand_mean = np.mean(D_sq)
    B = -0.5 * (D_sq - row_means - col_means + grand_mean)
    return B



def compute_isomap(v, t=None, neig=3, ndim=3, n_neighbors=30, standardize=True):
    if standardize:
        v_std = (v - v.mean(axis=0)) / v.std(axis=0)
        t_std = (t - t.mean()) / t.std() if t is not None else None
        v_input = v_std
        t_input = t_std
    else:
        v_input = v
        t_input = t
        
    if t_input is not None:
        V_aug = np.column_stack((v_input, t_input))
    else:
        V_aug = v_input
        
    S_isomap = Isomap(n_neighbors=n_neighbors, n_components=ndim, n_jobs=1).fit_transform(V_aug)
    B = CMDS(S_isomap)
    eigvals = eigsh(B, k=neig, which='LA', return_eigenvectors=False) 
    return eigvals[::-1]

def compute_lle(v, t=None, neig=3, ndim=3, n_neighbors=30, standardize=True, seed=234):
    if standardize:
        v_std = (v - v.mean(axis=0)) / v.std(axis=0)
        t_std = (t - t.mean()) / t.std() if t is not None else None
        v_input = v_std
        t_input = t_std
    else:
        v_input = v
        t_input = t
        
    if t_input is not None:
        V_aug = np.column_stack((v_input, t_input))
    else:
        V_aug = v_input
    try:
        S_lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=ndim, random_state=seed).fit_transform(V_aug)
    except (ValueError, RuntimeError):
        try:
            S_lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=ndim, random_state=seed, eigen_solver='dense').fit_transform(V_aug)
        except (ValueError, RuntimeError):
            S_lle = LocallyLinearEmbedding(n_neighbors=10, n_components=ndim, random_state=seed).fit_transform(V_aug)
    B = CMDS(S_lle)
    eigvals = eigsh(B, k=neig, which='LA', return_eigenvectors=False) 
    return eigvals[::-1]

def process_pcd_ml(args):
    V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, percentage, color, seed, neig, method = args
    V_i, T_i = generate_pcd_local_color(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, percentage, color, seed)
    if method == 'isomap':
        evals = compute_isomap(v = V_i, t = T_i, neig = neig, ndim = 3, n_neighbors = 30, standardize=True)
    elif method == 'lle':
        evals = compute_lle(v = V_i, t = T_i, neig = neig, ndim = 3, n_neighbors = 30, standardize = True, seed = seed)
    else:
        raise ValueError(f"Unknown method: {method}")
    return evals

def process_batch_pcds_ml(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, percentage, color, seeds, neig, method, n_jobs):
    args_list = [(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, percentage, color, seed, neig, method) for seed in seeds]
    results = Parallel(n_jobs=n_jobs)(delayed(process_pcd_ml)(args) for args in args_list)
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
        parser.add_argument('--initloc', type=int, default=0, help='Initial batch index for local noise (default: 0)')
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

    print(f"Method: {method}")
    neig = args.neig
    print(f"N.eigenvalues: {neig}")
    idx = args.id
    main_seed = args.id + 100 
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

    min_size = 8100
    max_size = 8140
    centering = False
    V = np.load(os.path.join(input_folder_CAD, 'bunny.npy'))  
    print(f"Loaded CAD vertices with shape {V.shape}.")
    nominal_texture = compute_nominal_texture(V, axis_texture=1, scaling_factor=10.0, centering=centering)

    snr = np.array([1.15, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75])
    assert len(snr) == nbatches, "Length of snr array must match number of batches"
    colors = [-0.01, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3] 
    perc = 0.01
    init_col = 0
    end_col = 5
    init = args.initloc
    end = init + 1

    for j in range(init, end):
        print(f"\nProcessing batch {j+1}/{nbatches} with seed {batch_seeds[j]}")
        sigma_oc_shape = sigma_shape * snr[j]
        print(f"Using sigma_oc_shape: {sigma_oc_shape}")
        for k in range(init_col, end_col):
            col = colors[k]
            print(f"  Using color offset: {col}")
            start_time = time.time()
            seeds_j = all_batch_seeds[j]
            evals_j = process_batch_pcds_ml(V, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, sigma_oc_shape, perc, col, seeds_j, neig, method, n_jobs = args.n_jobs)
            end_time = time.time()
            np.save(os.path.join(output_folder_evals, f'evals_{k+1}.npy'), evals_j)
            elapsed = end_time - start_time
            print(f"Batch {j+1} completed in {elapsed:.2f}s ({elapsed/batch_size:.3f}s per PCD)")
            gc.collect()
    time.sleep(1)
    

if __name__ == '__main__':
    main()
