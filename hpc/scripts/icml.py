import os
import argparse
import time
import gc
import numpy as np
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
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

# 3. generate_pcd: generate a point cloud with random subsampling and noise
def generate_pcd(V_original, T_nominal, min_size, max_size, sigma_shape, sigma_texture, seed):
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
    T_i = np.clip(T_i, 0, None)
    return V_i, T_i

def generate_pcd_sub(V_original, T_nominal, min_size, max_size, sigma_shape, sigma_texture, seed):
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
    T_i = np.clip(T_i, 0, None)
    indices_to_keep = rng.choice(len(V_i), size = int(V_i.shape[0] // 3), replace = False)
    return V_i[indices_to_keep], T_i[indices_to_keep]

def CMDS(X):
    """Optimized CMDS using vectorized operations and memory efficiency"""    
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
    V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seed, neig, method, sub = args
    if sub:
        V_i, T_i = generate_pcd_sub(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seed)
    else:
        V_i, T_i = generate_pcd(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seed)
    if method == 'isomap':
        evals = compute_isomap(v = V_i, t = T_i, neig = neig, ndim = 3, n_neighbors = 30, standardize=True)
    elif method == 'lle':
        evals = compute_lle(v = V_i, t = T_i, neig = neig, ndim = 3, n_neighbors = 30, standardize = True, seed = seed)
    else:
        raise ValueError(f"Unknown method: {method}")
    return evals

def process_batch_pcds_ml(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seeds, neig, method, n_jobs, sub):
    args_list = [(V_original, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seed, neig, method, sub) for seed in seeds]
    results = Parallel(n_jobs=n_jobs)(delayed(process_pcd_ml)(args) for args in args_list)
    return np.array(results)

def main():
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Compute ML methods evals.')
        parser.add_argument('--outputFolder', type=str, required=True, help='Output folder')
        parser.add_argument('--inputFolderCAD', type=str, required=True, help='Input folder')
        parser.add_argument('--neig', type=int, required=True, help='N.eigenvalues')
        parser.add_argument('--method', type=str, required=True, help='Method: isomap, lle or tsne')
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
    assert method in ['isomap', 'lle', 'tsne'], "Method must be one of: isomap, lle, tsne"
    input_folder_CAD = args.inputFolderCAD
    print(f"Input folder CAD: {input_folder_CAD}")
    output_folder = args.outputFolder
    print(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    output_folder_evals = os.path.join(output_folder, 'evals')
    os.makedirs(output_folder_evals, exist_ok=True)

    print(f"Method: {method}")
    neig = args.neig
    print(f"N.eigenvalues: {neig}")
    idx = args.id
    main_seed = args.id + 1  # use id as main seed
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
    
    min_size = 8100
    max_size = 8140
    centering = False
    sigma_shape = 0.0001
    sigma_texture = 0.01
    sub = False
    V = np.load(os.path.join(input_folder_CAD, 'bunny.npy'))

    print(f"Loaded CAD vertices with shape {V.shape}.")
    nominal_texture = compute_nominal_texture(V, axis_texture=1, scaling_factor=10.0, centering=centering)
    init = 0
    print(f"Starting from batch index: {init}")
    
    for j in range(init, nbatches):
        print(f"\nProcessing batch {j+1}/{nbatches} with seed {batch_seeds[j]}")
        start_time = time.time()
        seeds_j = all_batch_seeds[j]
        evals_j = process_batch_pcds_ml(V, nominal_texture, min_size, max_size, sigma_shape, sigma_texture, seeds_j, neig, method, n_jobs=args.n_jobs, sub=sub)
        end_time = time.time()
        np.save(os.path.join(output_folder_evals, f'evals_{j+1}.npy'), evals_j)
        print(f"Saved evals for batch {j+1} to {output_folder}")
        elapsed = end_time - start_time
        print(f"Batch {j+1} completed in {elapsed:.2f}s ({elapsed/batch_size:.3f}s per PCD)")
        gc.collect()
    time.sleep(2)

if __name__ == '__main__':
    main()
