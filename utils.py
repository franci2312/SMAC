import os
import sys
import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import eigh
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm


def mean_invs(data, num, rank, remove_first = True):
    if remove_first:
        data = data[:num, 1:rank]
    else:
        data = data[:num, :rank]
    mdata = np.mean(data, axis=0)
    mcov = np.cov(data, rowvar=False)
    sqrtcov = sqrtm(mcov)
    invs = matrix_power(sqrtcov, -1)
    return mdata, invs, data

def matrix_power(A, power):
    eigv, eigvec = eigh(A)
    return eigvec @ np.diag(eigv**power) @ eigvec.T

### Control Charts Utils ###

def apply_cusum(matrix, k = 0.2, init = 0.0, up = True):
    # apply cusum to each row of the matrix
    nrows, ncols = matrix.shape
    out = np.zeros((nrows, ncols))
    for i in range(nrows):
        row = matrix[i, :]
        cusum_row = np.zeros(ncols)
        s_prev = init
        for j in range(ncols):
            if up:
                cusum_row[j] = max(0, s_prev + row[j] - k)
            else:
                cusum_row[j] = min(0, s_prev + row[j] + k)
            s_prev = cusum_row[j]
        out[i, :] = cusum_row
    return out

def apply_ewma(matrix, k = 0.2, init = 0.0):
    # apply ewma to each row of the matrix
    nrows, ncols = matrix.shape
    out = np.zeros((nrows, ncols))
    for i in range(nrows):
        row = matrix[i, :]
        ewma_row = np.zeros(ncols)
        s_prev = init
        for j in range(ncols):
            ewma_row[j] = k * row[j] + (1 - k) * s_prev
            s_prev = ewma_row[j]
        out[i, :] = ewma_row
    return out

def compute_stats_ba(data, stat_type, scalars_mean, scalars_std, k = 0.2, B = 5000, maxrl = 3000, init = 0.0, up = True, seed = 123):
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    scalars = np.sum(data * data, axis = 1)
    rng = np.random.default_rng(seed)
    scalars_bts = np.array([rng.choice(scalars, size=maxrl, replace=True) for _ in range(B)])
    scalars_bts = (scalars_bts - scalars_mean) / scalars_std
    if stat_type == "cusum":
        out = apply_cusum(scalars_bts, init=init, k=k, up=up)
    elif stat_type == "ewma":
        out = apply_ewma(scalars_bts, k=k, init=init)
    else:
        raise ValueError("Invalid stat_type")
    return out

### Single Chart ###

def rl_ba(h, trajectory, maxrl):
    rl = maxrl
    for t in range(maxrl):
        if trajectory[t] > h:
            rl = t + 1
            break
    return rl

def ba_bisection(arl0, # target average run length
                stats, # precomputed stats (already determine maxrl and bootstrap)
                verbose = False, 
                nsims = 1000, a_tol = 1.0, h_tol = sys.float_info.epsilon, maxiter = 30, ss = 0, seed = None):
    if len(stats.shape) != 2:
        raise ValueError("stats must be a 2D array (B x maxrl).")
    if nsims > stats.shape[0]:
        raise ValueError("nsims must be less than or equal to the number of rows in stats.")
    aarl, h_old = 0.0, np.inf
    min_stat, max_stat = np.min(stats.flatten()), np.max(stats.flatten())
    h_low, h_up = min_stat, max_stat
    B, maxrl = stats.shape
    if verbose:
        print("h_low_start: ", min_stat, "h_up_start: ", max_stat)
        print("B:", B, "maxrl:", maxrl)
    base_seed = 231 if seed is None else seed
    rng = np.random.RandomState(base_seed)
    h_path, aarl_path = [], []
    for i in range(maxiter):
        if verbose: print("iteration:", i + 1, "of", maxiter)
        h = (h_up + h_low) / 2.0
        h_path.append(h)
        rl = np.zeros(nsims)
        row_indices = rng.choice(B, size=nsims, replace=False)
        for sim in range(nsims):
            trajectory = stats[row_indices[sim], ss:]
            rl[sim] = rl_ba(h=h, trajectory=trajectory, maxrl=maxrl-ss)
        aarl = np.mean(rl)
        aarl_path.append(aarl)
        sdaarl = np.std(rl)
        if verbose:
            print("limit: ", h, "arl: ", aarl, "sdaarl: ", sdaarl)
            print("quantiles: ", np.quantile(rl, [0.0, 0.25, 0.5, 0.75, 1.0]))
        if (np.abs(h - h_old) < h_tol or np.abs(aarl - arl0) < a_tol):
            break
        if aarl < arl0:
            h_low = h
        else:
            h_up = h
        h_old = h
    results = {
        'h': h,
        'rl': rl,
        'iter': i+1,
        'h_path': h_path,
        'aarl_path': aarl_path
    }
    return results

def _compute_rl(stats_matrix, h, nsims, start_col, maxrl):
    rl = np.zeros(nsims)
    for sim in range(nsims):
        trajectory = stats_matrix[sim, start_col:]
        rl[sim] = rl_ba(h=h, trajectory=trajectory, maxrl=maxrl - start_col)
    return {'h': h, 'aarl': np.mean(rl), 'sdrl': np.std(rl), 'rl': rl}

def ic_sim_arl_ba(h,
                  stats_matrix = None,
                  nsims = 1000,
                  maxrl = 3000,
                  ss=0):
    if stats_matrix is None:
        raise ValueError("Either 'stats_matrix' must be provided.")
    return _compute_rl(stats_matrix, h, nsims, start_col=ss, maxrl=maxrl)

def oc_sim_arl_ba(h, 
                  ic_obs, # in control observations
                  oc_obs, # out of control observations
                  ic_mean, ic_std,
                  nsims = 1000,
                  change_point = 100,
                  stat_type = None, k = 0.2, maxrl = 3000, init = 0.0, up = True, seed = None):
    if stat_type is None:
        raise ValueError("stat_type must be provided.")
    if stat_type not in ["cusum", "ewma"]:
        raise ValueError("Invalid stat_type")
    base_seed = 231 if seed is None else seed
    rng = np.random.default_rng(base_seed)
    if len(ic_obs.shape) == 1:
        ic_obs = ic_obs.reshape(1, -1)
    if len(oc_obs.shape) == 1:
        oc_obs = oc_obs.reshape(1, -1)
    ic_scalars = np.sum(ic_obs * ic_obs, axis = 1) 
    oc_scalars = np.sum(oc_obs * oc_obs, axis = 1)
    ic_scalars = (ic_scalars - ic_mean) / ic_std
    oc_scalars = (oc_scalars - ic_mean) / ic_std
    ic_scalars_bts = np.array([rng.choice(ic_scalars, size=change_point, replace=True) for _ in range(nsims)])
    oc_scalars_bts = np.array([rng.choice(oc_scalars, size=maxrl, replace=True) for _ in range(nsims)])
    data = np.hstack((ic_scalars_bts, oc_scalars_bts))
    if stat_type == "cusum":
        stats_matrix = apply_cusum(data, k=k, init=init, up=up)
    elif stat_type == "ewma":
        stats_matrix = apply_ewma(data, k=k, init=init)
    return _compute_rl(stats_matrix, h, nsims, start_col=change_point, maxrl=maxrl)

### Combined Chart ###

def rl_ba_stats_combined(J, h, trajectories, maxrl):
    rl = maxrl
    found = False
    signal = np.zeros(J)
    for t in range(maxrl):
        for j in range(J):
            if trajectories[j][t] > h[j]:
                signal = [1 if trajectories[k][t] > h[k] else 0 for k in range(J)]
                rl = t + 1
                found = True
                break
        if found:
            break
    return rl, signal

def ba_oc_sim_arl_combined(h,
                        ic_obs, # list
                        oc_obs, # list
                        ic_means, ic_stds,
                        nsims = 1000,
                        change_point = 100,
                        stat_type = None, k = 0.2, maxrl = 3000, init = 0.0, up = True, seed = None):
    assert isinstance(h, list) and isinstance(ic_obs, list) and isinstance(oc_obs, list), "h, ic_obs and oc_obs must be lists"
    assert isinstance(ic_means, list) and isinstance(ic_stds, list), "ic_means and ic_stds must be lists"
    J = len(ic_obs)
    if len(oc_obs) != J or len(h) != J:
        raise ValueError(f"ic_obs, oc_obs, and h must be lists of length {J}")
    if stat_type is None:
        raise ValueError("stat_type must be provided.")
    if stat_type not in ["cusum", "ewma"]:
        raise ValueError("Invalid stat_type")
    base_seed = 231 if seed is None else seed
    rng = np.random.default_rng(base_seed)
    if any(len(ic_obs[j].shape) == 1 for j in range(J)):
        ic_obs = [ic_obs[j].reshape(1, -1) if len(ic_obs[j].shape) == 1 else ic_obs[j] for j in range(J)]
    if any(len(oc_obs[j].shape) == 1 for j in range(J)):
        oc_obs = [oc_obs[j].reshape(1, -1) if len(oc_obs[j].shape) == 1 else oc_obs[j] for j in range(J)]
    ic_scalars_list = [np.sum(ic_obs[j] * ic_obs[j], axis = 1) for j in range(J)]
    oc_scalars_list = [np.sum(oc_obs[j] * oc_obs[j], axis = 1) for j in range(J)]
    ic_scalars_std = [(ic_scalars_list[j] - ic_means[j]) / ic_stds[j] for j in range(J)]
    oc_scalars_std = [(oc_scalars_list[j] - ic_means[j]) / ic_stds[j] for j in range(J)]
    # create bootstrap samples
    ic_scalars_bts_list = [np.array([rng.choice(ic_scalars_std[j], size=change_point, replace=True) for _ in range(nsims)]) for j in range(J)]
    oc_scalars_bts_list = [np.array([rng.choice(oc_scalars_std[j], size=maxrl, replace=True) for _ in range(nsims)]) for j in range(J)]
    # combine ic and oc samples
    data_list = [np.hstack((ic_scalars_bts_list[j], oc_scalars_bts_list[j])) for j in range(J)]
    # apply control chart
    stats_matrices = []
    for j in range(J):
        if stat_type == "cusum":
            stats_matrix = apply_cusum(data_list[j], init=init, k=k, up=up)
        elif stat_type == "ewma":
            stats_matrix = apply_ewma(data_list[j], k=k, init=init)
        stats_matrices.append(stats_matrix)
    rl = np.zeros(nsims)
    signal = np.zeros((nsims, J))
    for sim in range(nsims):
        trajectories = [stats_matrices[j][sim, change_point:] for j in range(J)]
        rl[sim], signal[sim] = rl_ba_stats_combined(J=J, h=h, trajectories=trajectories, maxrl=maxrl-change_point)
    results = {
        'h': h,
        'aarl': np.mean(rl),
        'sdrl': np.std(rl),
        'rl': rl,
        'signal': signal
    }
    return results


def ba_ic_sim_arl_combined(h, # list
                        stats, # list
                        nsims=1000, B=None, maxrl=None, ss=0, seed=None):
    assert isinstance(stats, list), "stats must be a list"
    assert B is not None and maxrl is not None, "B and maxrl must be provided"
    J = len(stats)
    if not isinstance(h, list) or len(h) != J:
        raise ValueError(f"h must be a list of length {J}")
    base_seed = 231 if seed is None else seed
    rng = np.random.default_rng(base_seed)
    row_indices = rng.choice(B, size=nsims, replace=False)
    rl = np.zeros(nsims)
    signal = np.zeros((nsims, J))
    for sim in range(nsims):
        idx = row_indices[sim]
        trajectories = [stats[j][idx, ss:] for j in range(J)]
        rl[sim], signal[sim] = rl_ba_stats_combined(J=J, h=h, trajectories=trajectories, maxrl=maxrl-ss)
    
    results = {
        'h': h,
        'aarl': np.mean(rl),
        'sdrl': np.std(rl),
        'rl': rl,
        'signal': signal
    }
    return results

def ba_bisection_combined(arl0, 
                       stats,
                       verbose = False,
                       k = 0.2, init = 0, up = True,
                       a_tol = 1.0, h_tol = [sys.float_info.epsilon, sys.float_info.epsilon],
                       nsims = 1000, maxiter = 30, ss = 0, seed = None): 

    J = len(stats)
    assert isinstance(h_tol, list) and isinstance(stats, list) and len(h_tol) == J
    aarl_g, h_old, h = 0.0, [np.inf for _ in range(J)], [0.0 for _ in range(J)]
    h_low = [np.min(stat) for stat in stats]
    h_up = [np.max(stat) for stat in stats]
    B, maxrl = stats[0].shape
    if nsims > B:
        raise ValueError("nsims must be less than or equal to the number of rows in stats.")
    for j in range(J):
        if stats[j].shape != (B, maxrl):
            raise ValueError(f"stats[{j}] has shape {stats[j].shape}, expected {(B, maxrl)}")
    if verbose:
        print("h_low_start: ", h_low, "h_up_start: ", h_up)
        print("B:", B, "maxrl:", maxrl)
    base_seed = 231 if seed is None else seed
    rng = np.random.default_rng(base_seed)
    all_seeds = [rng.integers(0, 2**31 - 1) for _ in range(maxiter * (J + 1))]
    # h_path, aarl_path are two lists of J entries (each j entry will be updated)
    h_path, aarl_path = [[] for _ in range(J)], [[] for _ in range(J)]
    aarl_g_path = []
    for i in range(maxiter):
        print("iteration: ", i + 1, "of", maxiter) if verbose else None
        # first control scheme
        h[0] = (h_low[0] + h_up[0]) / 2.0
        h_path[0].append(h[0])
        aarl_0 = ic_sim_arl_ba(h=h[0], stats_matrix=stats[0], nsims=nsims, maxrl=maxrl, ss=ss)['aarl']
        aarl_path[0].append(aarl_0)
        print("h[0]: ", h[0], "aarl_0: ", aarl_0) if verbose else None
        # compute h_j such that each control scheme attains the aarl of the first
        for j in range(1, J):
            print("Inner loop for j: ", j + 1, "of", J) if verbose else None
            res_j = ba_bisection(arl0=aarl_0, stats=stats[j], verbose=verbose, nsims=nsims, a_tol=a_tol, h_tol=h_tol[j], maxiter=maxiter, ss=ss, seed=all_seeds[i * (J + 1) + j])
            h[j] = res_j['h']
            h_path[j].append(h[j])
            aarl_path[j].append(np.mean(res_j['rl']))
        # compute arl whole scheme
        res_g = ba_ic_sim_arl_combined(h=h, stats=stats, nsims=nsims, B=B, maxrl=maxrl, ss=ss, seed=all_seeds[i * (J + 1) + J])
        aarl_g = res_g['aarl']
        aarl_g_path.append(aarl_g)
        if verbose:
            print("GLOBAL arl: ", aarl_g, "sdrl: ", res_g['sdrl'])
            print("quantiles: ", np.quantile(res_g['rl'], [0.0, 0.25, 0.5, 0.75, 1.0]))
            print("h:", h)
            print("h_old:", h_old)
            print("np.abs(h - h_old):", np.abs(np.array(h) - np.array(h_old)))
            print("aarl_g:", aarl_g)
            print("---------------------------------------------")
        if np.abs(aarl_g - arl0) < a_tol or np.all(np.abs(np.array(h) - np.array(h_old)) < np.array(h_tol)):
            break
        if aarl_g < arl0:
            h_low = h.copy()
        else:
            h_up = h.copy()
        h_old = h.copy()

    results = {
        'h': h,
        'iter': i+1,
        'h_path': h_path,
        'aarl_path': aarl_path,
        'aarl_g_path': aarl_g_path
    }
    return results

# BENCHMARKS ############################################################

def estimation_benchmarks(input_folder, n1, n2, n3, n4, rank, method):
    total = n1 + n2 + n3 + n4
    n_per_file = 1000
    rank_tmp = rank - 1 if method == 'gl' else rank
    evals = np.zeros((total, rank_tmp))
    n_files = int(np.ceil(total / n_per_file))
    curr = 0
    init = 1 if method == 'gl' else 0
    start_in = 0
    for i in range(n_files):
        evals_i = np.load(os.path.join(input_folder, "evals", f"evals_{i+1}.npy"))[:, init:rank]
        start_out = curr
        end_out = min(curr + n_per_file, total)
        end_in = min(n_per_file, total - curr)
        evals[start_out:end_out] = evals_i[start_in:end_in]
        curr += end_in
    mevals, invsevals, _ = mean_invs(evals[:n1], num = n1, rank = rank_tmp, remove_first = False)
    evals_s2_std = (evals[n1:n1+n2] - mevals) @ invsevals
    evals_s2_sca = np.sum(evals_s2_std * evals_s2_std, axis = 1)
    mevalssca = np.mean(evals_s2_sca)
    stdevalssca = np.std(evals_s2_sca)
    params_evals = {'meanv': mevals, 'invs': invsevals, 'means': mevalssca, 'stds': stdevalssca}
    return params_evals, evals[n1+n2:]


def compute_control_limits_benchmarks(params_evals, evals, n3, n4, stat_type, k, B, arl0, maxiter, maxrl, verbose, nsims, ss, seed):
    assert evals.shape[0] == n3 + n4, "Number of observations not equal to n3 and n4"
    tuning_evals = evals[:n3]
    valid_evals = evals[n3:]
    tuning_evals_std = (tuning_evals - params_evals['meanv']) @ params_evals['invs']
    stats_evals = compute_stats_ba(tuning_evals_std, stat_type, scalars_mean = params_evals['means'], scalars_std = params_evals['stds'], k = k, B = B, maxrl = maxrl, init = 0.0, up = True, seed = seed)
    result_bisection = ba_bisection(arl0 = arl0, stats = stats_evals, verbose = False, a_tol = 1.0, nsims = nsims, maxiter = maxiter, ss = ss, seed = seed)
    if verbose:
        result_bisection['h']
        result_bisection['aarl_path']
    valid_evals_std = (valid_evals - params_evals['meanv']) @ params_evals['invs']
    valid_stats_evals = compute_stats_ba(valid_evals_std, stat_type, scalars_mean = params_evals['means'], scalars_std = params_evals['stds'], k = k, B = B, maxrl = maxrl, init = 0.0, up = True, seed = seed)
    ic_perf = ic_sim_arl_ba(result_bisection['h'], stats_matrix = valid_stats_evals, nsims=B, maxrl=maxrl, ss=ss)
    return ic_perf, result_bisection

def benchmarks_ic(input_folder = '.', output_folder = '.', method = '.', nrep = 30, rank = 101, stat_type = 'ewma', k = 0.2, n1 = 700, n2 = 300, n3 = 2000, n4 = 5000,
             B = 1000, maxrl = 4000, arl0 = 100.0, maxiter = 100, verbose = False, nsims = 1000, ss=50, seed =1142):
    """Compute control limits and ic performance for benchmarks. Conditional ARL performance."""
    assert method in ['gl', 'isomap', 'lle'], "Method must be 'gl', 'isomap', or 'lle'"
    arls = np.zeros((nrep,1))
    for rep in range(nrep):
        output_path = os.path.join(output_folder, f"run{rep+1}")
        os.makedirs(output_path, exist_ok=True)
        input_folder_rep = os.path.join(input_folder, f"run{rep+1}")
        params_evals, evals_ic = estimation_benchmarks(input_folder_rep, n1, n2, n3, n4, rank, method)
        ic_perf, result_bisection = compute_control_limits_benchmarks(params_evals, evals_ic, n3, n4, stat_type, k, B, arl0, maxiter, maxrl, verbose, nsims, ss, seed)
        arls[rep] = ic_perf['aarl']
        if verbose:
            print(f"ARL for replication {rep+1}: {arls[rep]}")
        np.save(os.path.join(output_path, f"lim.npy"), result_bisection['h'])
        np.savez(os.path.join(output_path, f"params_evals.npz"), meanv = params_evals['meanv'], invs = params_evals['invs'], means = params_evals['means'], stds = params_evals['stds'])
    mean_arl = np.mean(arls)
    std_arl = np.std(arls)
    if verbose:
        print(f"Mean: {mean_arl}, Std: {std_arl}")
    return arls

def benchmarks_oc(input_folder_ic, input_folder_oc, input_folder_est, method, rank, stat_type, k, n3, nrep = 30, B = 1000, cp = 100, maxrl = 4000, levels = 6, seed = 4321):
    if not os.path.exists(input_folder_ic):
        raise ValueError(f"Input folder for IC data does not exist: {input_folder_ic}")
    if not os.path.exists(input_folder_oc):
        raise ValueError(f"Input folder for OC data does not exist: {input_folder_oc}")
    subfolder_est = os.path.join(input_folder_est, f"n3_{n3}_{stat_type}k{k}")
    if not os.path.exists(subfolder_est):
        raise ValueError(f"Input folder for estimation results does not exist: {subfolder_est}")
    assert method in ['isomap', 'lle', 'gl'], "Method must be one of 'isomap', 'lle', or 'gl'"
    start_i = 1 if method == 'gl' else 0
    arls = np.zeros((nrep, levels))
    for rep in range(nrep):
        input_folder_ic_rep = os.path.join(input_folder_ic, f"run{rep+1}")
        input_folder_oc_rep = os.path.join(input_folder_oc, f"run{rep+1}")
        subfolder_est_rep = os.path.join(subfolder_est, f"run{rep+1}")
        evals_ic = np.load(os.path.join(input_folder_ic_rep, "evals", f"evals_10.npy"))[:, start_i:rank]
        limit = np.load(os.path.join(subfolder_est_rep, "lim.npy")).tolist()
        params_evals = np.load(os.path.join(subfolder_est_rep, "params_evals.npz"))
        evals_ic_std = (evals_ic - params_evals['meanv']) @ params_evals['invs']
        for j in range(levels):
            evals_oc = np.load(os.path.join(input_folder_oc_rep, "evals", f"evals_{j+1}.npy"))[:, start_i:rank]
            evals_oc_std = (evals_oc - params_evals['meanv']) @ params_evals['invs']
            e_perf = oc_sim_arl_ba(h = limit, ic_obs = evals_ic_std, oc_obs = evals_oc_std, ic_mean = params_evals['means'], ic_std = params_evals['stds'], nsims = B, change_point = cp, stat_type = stat_type, k = k, maxrl = maxrl, init = 0.0, up = True, seed = seed + rep * levels + j)
            arls[rep, j] = e_perf['aarl']
    return arls

# SMAC ####################################################

def estimation_smac(input_folder, n1, n2, n3, n4, rank):
    total = n1 + n2 + n3 + n4
    n_per_file = 1000
    evals, coeff = np.zeros((total, rank-1)), np.zeros((total, rank))
    n_files = int(np.ceil(total / n_per_file))
    curr = 0
    start_in = 0
    for i in range(n_files):
        evals_i = np.load(os.path.join(input_folder, "evals", f"evals_{i+1}.npy"))[:, 1:rank]
        coeff_i = np.abs(np.load(os.path.join(input_folder, "coeff", f"coeff_{i+1}.npy"))[:, :rank])
        start_out = curr
        end_out = min(curr + n_per_file, total)
        end_in = min(n_per_file, total - curr)
        evals[start_out:end_out] = evals_i[start_in:end_in]
        coeff[start_out:end_out] = coeff_i[start_in:end_in]
        curr += end_in
    mevals, invsevals, _ = mean_invs(evals[:n1], num = n1, rank = rank, remove_first = False)
    mcoeff, invscoeff, _ = mean_invs(coeff[:n1], num = n1, rank = rank, remove_first = False)
    evals_s2_std = (evals[n1:n1+n2] - mevals) @ invsevals
    coeff_s2_std = (coeff[n1:n1+n2] - mcoeff) @ invscoeff
    evals_s2_sca = np.sum(evals_s2_std * evals_s2_std, axis = 1)
    coeff_s2_sca = np.sum(coeff_s2_std * coeff_s2_std, axis = 1)
    mcoeffsca = np.mean(coeff_s2_sca)
    stdcoeffsca = np.std(coeff_s2_sca)
    mevalssca = np.mean(evals_s2_sca)
    stdevalssca = np.std(evals_s2_sca)
    params_evals = {'meanv': mevals, 'invs': invsevals, 'means': mevalssca, 'stds': stdevalssca}
    params_coeff = {'meanv': mcoeff, 'invs': invscoeff, 'means': mcoeffsca, 'stds': stdcoeffsca}
    return params_evals, params_coeff, evals[n1+n2:], coeff[n1+n2:]

def compute_control_limits_smac(params_evals, params_coeff, evals, coeff, n3, n4, stat_type, k, B, arl0, maxiter, maxrl, verbose, nsims, ss, seed):
    assert evals.shape[0] and coeff.shape[0] == n3 + n4, "Number of observations not equal to n3 and n4"
    tuning_evals = evals[:n3]
    tuning_coeff = coeff[:n3]
    valid_evals = evals[n3:]
    valid_coeff = coeff[n3:]
    tuning_evals_std = (tuning_evals - params_evals['meanv']) @ params_evals['invs']
    tuning_coeff_std = (tuning_coeff - params_coeff['meanv']) @ params_coeff['invs']
    stats_evals = compute_stats_ba(tuning_evals_std, stat_type, scalars_mean = params_evals['means'], scalars_std = params_evals['stds'], k = k, B = B, maxrl = maxrl, init = 0.0, up = True, seed = seed)
    stats_coeff = compute_stats_ba(tuning_coeff_std, stat_type, scalars_mean = params_coeff['means'], scalars_std = params_coeff['stds'], k = k, B = B, maxrl = maxrl, init = 0.0, up = True, seed = seed)
    tuning_stats = [stats_evals, stats_coeff]
    result_bisection = ba_bisection_combined(arl0 = arl0, stats = tuning_stats, verbose = False, k = k, init = 0.0, up = True, a_tol = 1.0, nsims = nsims, maxiter = maxiter, ss = ss, seed = seed)
    if verbose:
        result_bisection['h']
        result_bisection['aarl_path'][0][-1], result_bisection['aarl_path'][1][-1] 
        result_bisection['aarl_g_path'][-1]
    valid_evals_std = (valid_evals - params_evals['meanv']) @ params_evals['invs']
    valid_coeff_std = (valid_coeff - params_coeff['meanv']) @ params_coeff['invs']
    valid_stats_evals = compute_stats_ba(valid_evals_std, stat_type, scalars_mean = params_evals['means'], scalars_std = params_evals['stds'], k = k, B = B, maxrl = maxrl, init = 0.0, up = True, seed = seed)
    valid_stats_coeff = compute_stats_ba(valid_coeff_std, stat_type, scalars_mean = params_coeff['means'], scalars_std = params_coeff['stds'], k = k, B = B, maxrl = maxrl, init = 0.0, up = True, seed = seed)
    valid_stats = [valid_stats_evals, valid_stats_coeff]
    ic_perf = ba_ic_sim_arl_combined(result_bisection['h'], stats = valid_stats, nsims=B, B=B, maxrl=maxrl, ss=ss, seed=seed)
    return ic_perf, result_bisection


def smac_ic(input_folder = '.', output_folder = '.', nrep = 30, rank = 126, stat_type = 'ewma', k = 0.2, n1 = 700, n2 = 300, n3 = 2000, n4 = 5000,
             B = 5000, maxrl = 4000, arl0 = 100.0, maxiter = 100, verbose = False, nsims = 5000, ss = 50, seed = 1142):
    """Compute control limits and ic performance for satto. Conditional ARL performance."""
    arls = np.zeros((nrep,1))
    for rep in range(nrep):
        output_path = os.path.join(output_folder, f"run{rep+1}")
        os.makedirs(output_path, exist_ok=True)
        input_folder_rep = os.path.join(input_folder, f"run{rep+1}")
        params_evals, params_coeff, evals_ic, coeff_ic = estimation_smac(input_folder_rep, n1, n2, n3, n4, rank)
        ic_perf, result_bisection = compute_control_limits_smac(params_evals, params_coeff, evals_ic, coeff_ic, n3, n4, stat_type, k, B, arl0, maxiter, maxrl, verbose, nsims, ss, seed)
        arls[rep] = ic_perf['aarl']
        if verbose:
            print(f"ARL for replication {rep+1}: {arls[rep]}")
        np.save(os.path.join(output_path, f"lim.npy"), result_bisection['h'])
        np.savez(os.path.join(output_path, "params_evals.npz"), meanv = params_evals['meanv'], invs = params_evals['invs'], means = params_evals['means'], stds = params_evals['stds'])
        np.savez(os.path.join(output_path, "params_coeff.npz"), meanv = params_coeff['meanv'], invs = params_coeff['invs'], means = params_coeff['means'], stds = params_coeff['stds'])
    mean_arl = np.mean(arls)
    std_arl = np.std(arls)
    if verbose:
        print(f"Mean: {mean_arl}, Std: {std_arl}")
    return arls

def smac_oc(input_folder_ic, input_folder_oc, input_folder_est, stat_type, k, n3, rank = 101, nrep = 30, B = 1000, cp = 100, maxrl = 4000, levels = 6, seed = 4321):
    if not os.path.exists(input_folder_ic):
        raise ValueError(f"Input folder for IC data does not exist: {input_folder_ic}")
    if not os.path.exists(input_folder_oc):
        raise ValueError(f"Input folder for OC data does not exist: {input_folder_oc}")
    subfolder_est = os.path.join(input_folder_est, f"n3_{n3}_{stat_type}k{k}")
    if not os.path.exists(subfolder_est):
        raise ValueError(f"Input folder for estimation results does not exist: {subfolder_est}")
    arls = np.zeros((nrep, levels))
    for rep in range(nrep):
        input_folder_ic_rep = os.path.join(input_folder_ic, f"run{rep+1}")
        input_folder_oc_rep = os.path.join(input_folder_oc, f"run{rep+1}")
        subfolder_est_rep = os.path.join(subfolder_est, f"run{rep+1}")
        evals_ic = np.load(os.path.join(input_folder_ic_rep, "evals", f"evals_10.npy"))[:, 1:rank]
        coeff_ic = np.abs(np.load(os.path.join(input_folder_ic_rep, "coeff", f"coeff_10.npy"))[:, :rank])
        limit = np.load(os.path.join(subfolder_est_rep, "lim.npy")).tolist()
        params_evals = np.load(os.path.join(subfolder_est_rep, "params_evals.npz"))
        params_coeff = np.load(os.path.join(subfolder_est_rep, "params_coeff.npz"))
        evals_ic_std = (evals_ic - params_evals['meanv']) @ params_evals['invs']
        coeff_ic_std = (coeff_ic - params_coeff['meanv']) @ params_coeff['invs']
        samples_ic = [evals_ic_std, coeff_ic_std]
        ic_means = [params_evals['means'], params_coeff['means']]
        ic_stds = [params_evals['stds'], params_coeff['stds']]
        for j in range(levels):
            evals_oc = np.load(os.path.join(input_folder_oc_rep, "evals", f"evals_{j+1}.npy"))[:, 1:rank]
            coeff_oc = np.abs(np.load(os.path.join(input_folder_oc_rep, "coeff", f"coeff_{j+1}.npy"))[:, :rank])
            evals_oc_std = (evals_oc - params_evals['meanv']) @ params_evals['invs']
            coeff_oc_std = (coeff_oc - params_coeff['meanv']) @ params_coeff['invs']
            samples_oc = [evals_oc_std, coeff_oc_std]
            oc_perf = ba_oc_sim_arl_combined(limit, ic_obs=samples_ic, oc_obs=samples_oc, ic_means=ic_means, ic_stds=ic_stds, nsims=B, change_point=cp, stat_type=stat_type, k=k, maxrl=maxrl, seed=seed)
            arls[rep, j] = oc_perf['aarl']
    return arls


def smac_oc_signal(input_folder_ic, input_folder_oc, input_folder_est, stat_type, k, n3, rank = 101, nrep = 30, B = 1000, cp = 100, maxrl = 4000, levels = 6, seed = 4321):
    if not os.path.exists(input_folder_ic):
        raise ValueError(f"Input folder for IC data does not exist: {input_folder_ic}")
    if not os.path.exists(input_folder_oc):
        raise ValueError(f"Input folder for OC data does not exist: {input_folder_oc}")
    subfolder_est = os.path.join(input_folder_est, f"n3_{n3}_{stat_type}k{k}")
    if not os.path.exists(subfolder_est):
        raise ValueError(f"Input folder for estimation results does not exist: {subfolder_est}")
    arls = np.zeros((nrep, levels))
    signals = {}
    for rep in range(nrep):
        input_folder_ic_rep = os.path.join(input_folder_ic, f"run{rep+1}")
        input_folder_oc_rep = os.path.join(input_folder_oc, f"run{rep+1}")
        subfolder_est_rep = os.path.join(subfolder_est, f"run{rep+1}")
        evals_ic = np.load(os.path.join(input_folder_ic_rep, "evals", f"evals_10.npy"))[:, 1:rank]
        coeff_ic = np.abs(np.load(os.path.join(input_folder_ic_rep, "coeff", f"coeff_10.npy"))[:, :rank])
        limit = np.load(os.path.join(subfolder_est_rep, "lim.npy")).tolist()
        params_evals = np.load(os.path.join(subfolder_est_rep, "params_evals.npz"))
        params_coeff = np.load(os.path.join(subfolder_est_rep, "params_coeff.npz"))
        evals_ic_std = (evals_ic - params_evals['meanv']) @ params_evals['invs']
        coeff_ic_std = (coeff_ic - params_coeff['meanv']) @ params_coeff['invs']
        samples_ic = [evals_ic_std, coeff_ic_std]
        ic_means = [params_evals['means'], params_coeff['means']]
        ic_stds = [params_evals['stds'], params_coeff['stds']]
        for j in range(levels):
            evals_oc = np.load(os.path.join(input_folder_oc_rep, "evals", f"evals_{j+1}.npy"))[:, 1:rank]
            coeff_oc = np.abs(np.load(os.path.join(input_folder_oc_rep, "coeff", f"coeff_{j+1}.npy"))[:, :rank])
            evals_oc_std = (evals_oc - params_evals['meanv']) @ params_evals['invs']
            coeff_oc_std = (coeff_oc - params_coeff['meanv']) @ params_coeff['invs']
            samples_oc = [evals_oc_std, coeff_oc_std]
            oc_perf = ba_oc_sim_arl_combined(limit, ic_obs=samples_ic, oc_obs=samples_oc, ic_means=ic_means, ic_stds=ic_stds, nsims=B, change_point=cp, stat_type=stat_type, k=k, maxrl=maxrl, seed=seed)
            arls[rep, j] = oc_perf['aarl']
            signals[(rep, j)] = oc_perf['signal']
            print(f"Level {j+1}, ARL: {oc_perf['aarl']}, SDRL: {oc_perf['sdrl']}")
    return arls, signals

def run_sensitivity_n3(input_folder, methods, ranks, n1, n2, n3_list, n4, 
                       stat_type='cusum', k=0.05, B=5000, maxrl=4000, 
                       arl0=100.0, maxiter=100, nrep=30, nsims=1000, ss=50, verbose=False, seed=1142):
    assert isinstance(methods, list) and isinstance(ranks, list) and isinstance(n3_list, list), "Methods, ranks and n3_list must be lists"
    assert len(methods) == len(ranks), "Methods and ranks must have same length"
    
    for method in methods:
        assert method in ['lb', 'gl', 'isomap', 'lle'], f"Method must be 'lb', 'gl', 'isomap', or 'lle', got {method}"
    
    results = {method: {} for method in methods}
    
    # Run for each method
    for method, rank in zip(methods, ranks):
        print(f"\n{'='*60}")
        if method == 'lb':
            print(f"Running method: SMAC")
        else:
            print(f"Running method: {method.upper()}")
        print(f"{'='*60}")
        
        method_input_folder = os.path.join(input_folder, method, 'ic')
        
        # Run for each n3 value
        for n3_val in n3_list:
            print(f"\n  n3 = {n3_val}...")
            
            # Create output folder
            output_folder = os.path.join(
                input_folder, method, 'ic_res', 
                f'n3_{n3_val}_{stat_type}k{k}'
            )
            os.makedirs(output_folder, exist_ok=True)
            
            # Run appropriate function based on method
            if method == 'lb':
                arls = smac_ic(input_folder=method_input_folder, output_folder=output_folder, nrep=nrep, rank=rank, stat_type=stat_type, k=k, n1=n1, n2=n2, n3=n3_val, n4=n4, 
                               B=B, maxrl=maxrl, arl0=arl0, maxiter=maxiter, verbose=verbose, nsims=nsims, ss=ss, seed=seed) 
            else:  # gl, isomap, lle
                arls = benchmarks_ic(input_folder=method_input_folder, output_folder=output_folder, method=method, nrep=nrep, rank=rank, stat_type=stat_type, k=k, n1=n1, n2=n2, 
                    n3=n3_val, n4=n4, B=B, maxrl=maxrl, arl0=arl0, maxiter=maxiter, verbose=verbose, nsims=nsims, ss=ss, seed=seed)
            
            mean_arl = np.mean(arls)
            std_arl = np.std(arls)/np.sqrt(nrep)  
            results[method][n3_val] = {
                'mean': mean_arl,
                'std': std_arl,
                'formatted': f"{mean_arl:.2f} ({std_arl:.2f})"
            }
            
            print(f"    AARL: {mean_arl:.2f} (SDARL: {std_arl:.2f})")
    
    df_data = {}
    for method in methods:
        df_data[method.upper()] = [
            results[method][n3]['formatted'] for n3 in n3_list
        ]
    
    df = pd.DataFrame(df_data, index=n3_list).T
    df.columns.name = 'n3'
    df.index.name = 'Method'
    
    return df, results



def run_oc(input_folder, methods, ranks, n3, label, stat_type='cusum', k=0.05, B=5000, 
           maxrl=4000, nrep=30, cp=60, levels=6, seed=4321):
    # Validate inputs
    assert isinstance(methods, list) and isinstance(ranks, list), "Methods and ranks must be lists"
    assert len(methods) == len(ranks), "Methods and ranks must have same length"
    
    for method in methods:
        assert method in ['lb', 'gl', 'isomap', 'lle'], f"Method must be 'lb', 'gl', 'isomap', or 'lle', got {method}"
    
    # Store results
    results = {method: {} for method in methods}
    
    # Run for each method
    for method, rank in tqdm(zip(methods, ranks), total=len(methods), desc="Running OC"):        
        method_label_input_folder = os.path.join(input_folder, method, label)
        method_ic_input_folder = os.path.join(input_folder, method, 'ic')
        method_est_input_folder = os.path.join(input_folder, method, 'ic_res')
        if not os.path.exists(method_ic_input_folder):
            raise ValueError(f"Input folder for IC data does not exist: {method_ic_input_folder}")
        if not os.path.exists(method_label_input_folder):
            raise ValueError(f"Input folder for OC data does not exist: {method_label_input_folder}")
        if not os.path.exists(method_est_input_folder):
            raise ValueError(f"Input folder for estimation results does not exist: {method_est_input_folder}")
        

        # Create output folder
        output_folder = os.path.join(input_folder, method, 'oc_res', label, f'n3_{n3}_{stat_type}k{k}')
        os.makedirs(output_folder, exist_ok=True)
            
        # Run appropriate function based on method
        if method == 'lb':
            arls = smac_oc(input_folder_ic=method_ic_input_folder, input_folder_oc=method_label_input_folder, input_folder_est=method_est_input_folder, stat_type=stat_type, k=k, n3=n3, rank=rank, 
                               nrep=nrep, B=B, cp=cp, maxrl=maxrl, levels=levels, seed=seed) 
        else:  # gl, isomap, lle
            arls = benchmarks_oc(input_folder_ic=method_ic_input_folder, input_folder_oc=method_label_input_folder, input_folder_est=method_est_input_folder, method=method, rank=rank, stat_type=stat_type, k=k, 
                    n3=n3, nrep=nrep, B=B, cp=cp, maxrl=maxrl, levels=levels, seed=seed)
            
        mean_arl = np.mean(arls, axis=0)  # Mean across replications for each level
        std_arl = np.std(arls, axis=0)/np.sqrt(nrep)   # Std across replications for each level
        
        results[method] = {
            'mean': mean_arl,
            'std': std_arl,
            'mean_list': mean_arl.tolist(),
            'std_list': std_arl.tolist(),
            'formatted': [f"{m:.2f} ({s:.2f})" for m, s in zip(mean_arl, std_arl)]
        }
    
    df_data = {}
    for method in methods:
        df_data[method.upper()] = results[method]['formatted']
    
    df = pd.DataFrame(df_data, index=range(1, levels+1)).T
    df.columns = [f'Level {i}' for i in range(1, levels+1)]
    df.index.name = 'Method'
    
    return df, results


def parse_mean_std(value_str):
    """Parse 'mean (std)' format from CSV"""
    match = re.match(r'([\d.]+)\s*\(([\d.]+)\)', str(value_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan

def plot_arl_from_csv(csv_oc_path, csv_ic_path, ic_column,
                       shifts=[0, 0.005, 0.01, 0.02, 0.04, 0.08],
                       methods=None,
                       colors=None,
                       path_output=None, 
                       x_label='Shift',
                       y_lim=None,
                       n_std=1,
                       title=None):
    """
    Plot ARL curves from CSV files.
    
    Parameters:
    -----------
    csv_oc_path : str
        Path to CSV with OC results (columns: Method, Level 1, Level 2, ...)
    csv_ic_path : str
        Path to IC results CSV
    ic_column : str
        Column name for IC results (e.g., 'Level 1', '3000', etc.)
    shifts : list
        Shift levels corresponding to Level 1, Level 2, etc.
    methods : list, optional
        List of methods to plot. If None, plots all methods in CSV
    colors : dict, optional
        Color mapping for methods
    path_output : str, optional
        Path to save figure
    n_std : float
        Number of standard errors for error bars
    title : str, optional
        Plot title
    """
    # --- Global style tweaks ---
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.2
    })

    # Read CSV files
    df_oc = pd.read_csv(csv_oc_path)
    df_ic = pd.read_csv(csv_ic_path)
    
    # Replace 'LB' with 'SMAC' in both dataframes
    df_oc['Method'] = df_oc['Method'].replace('LB', 'SMAC')
    df_ic['Method'] = df_ic['Method'].replace('LB', 'SMAC')
    
    # Get methods from CSV if not specified
    if methods is None:
        methods = df_oc['Method'].tolist()
    
    # Default colors if not provided
    if colors is None:
        colors = {
            'GL': "#009E73",
            'SMAC': "#0072B2",
            'ISOMAP': "#E69F00",
            'LLE': "#D55E00"
        }
    
    # Line styles for different methods
    linestyles = {
        'SMAC': '-',      # continuous
        'GL': ':',        # dotted
        'ISOMAP': '--',   # dashed
        'LLE': '-'        # continuous (default)
    }

    # Prepare x-axis
    x = np.arange(len(shifts))
    
    fig, ax = plt.subplots(figsize=(9, 7))

    for method in methods:
        if method not in df_oc['Method'].values:
            print(f"Warning: Method '{method}' not found in OC CSV")
            continue
            
        means = []
        stds = []
        
        # Get IC result from specified column
        ic_row = df_ic[df_ic['Method'] == method]
        if not ic_row.empty:
            ic_mean, ic_std = parse_mean_std(ic_row.iloc[0][ic_column])
            means.append(ic_mean)
            stds.append(ic_std)
        else:
            print(f"Warning: No IC data for method '{method}'")
            continue
        
        # Get OC results (first n-1 Level columns, where n is total number of shifts)
        oc_row = df_oc[df_oc['Method'] == method].iloc[0]
        level_cols = [col for col in df_oc.columns if col.startswith('Level')]
        
        # Only use as many levels as needed: total shifts - 1 (for IC)
        num_oc_levels = len(shifts) - 1
        for level_col in level_cols[:num_oc_levels]:
            oc_mean, oc_std = parse_mean_std(oc_row[level_col])
            means.append(oc_mean)
            stds.append(oc_std)
        
        # Check if we have the right number of points
        if len(means) != len(shifts):
            print(f"Warning: Method '{method}' has {len(means)} points but expected {len(shifts)}")
            continue
        
        # Plot with error bars
        method_color = colors.get(method, 'gray')
        method_linestyle = linestyles.get(method, '-')
        ax.errorbar(
            x, means, yerr=n_std * np.array(stds),
            color=method_color,
            linestyle=method_linestyle,
            marker='o',
            markersize=6,
            linewidth=2.5,
            capsize=4,
            elinewidth=1.2,
            alpha=0.95,
            label=method
        )

    if y_lim is not None:
        ax.set_ylim(y_lim)

    # --- Axis formatting ---
    ax.set_xlabel(x_label)
    ax.set_ylabel('AARL')
    ax.set_xticks(x)
    ax.set_xticklabels(shifts)

    if title is not None:
        ax.set_title(title)

    # --- Grid: subtle and behind the data ---
    ax.set_axisbelow(True)
    ax.grid(
        True, axis='y',
        linestyle='--',
        linewidth=0.8,
        alpha=0.35
    )
    ax.grid(
        True, axis='x',
        linestyle='--',
        linewidth=0.8,
        alpha=0.35
    )

    # --- Spines ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.8)
    ax.spines['bottom'].set_alpha(0.8)

    # --- Legend ---
    ax.legend(frameon=False, loc='best')

    plt.tight_layout()
    if path_output is not None:
        plt.savefig(path_output, dpi=300, bbox_inches='tight')
    plt.show()