from scipy.spatial.distance import cdist
from scipy.stats import f
from scipy.stats import ks_2samp, mannwhitneyu
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import ast
import math
import random
from collections import defaultdict

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_if_list_string(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x

def preprocess_runs(runs):
    parsed = [parse_if_list_string(x) for x in runs]
    if any(isinstance(v, list) for v in parsed):
        max_len = max(len(v) if isinstance(v, list) else 1 for v in parsed)
        processed = np.zeros((len(parsed), max_len), dtype=object)
        for i, v in enumerate(parsed):
            if isinstance(v, list):
                processed[i, :len(v)] = v
            else:
                processed[i, 0] = v
        return processed
    else:
        return np.array(parsed, dtype=object)

def energy_distance(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    n, m = len(X), len(Y)
    dXY = np.sum(cdist(X, Y)) / (n * m)
    dXX = np.sum(cdist(X, X)) / (n * n)
    dYY = np.sum(cdist(Y, Y)) / (m * m)
    return 2 * dXY - dXX - dYY

def energy_pvalue(X, Y, n_permutations=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    X, Y = np.asarray(X), np.asarray(Y)
    n, m = len(X), len(Y)
    obs = energy_distance(X, Y)

    Z = np.vstack([X, Y])
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(Z))
        Xp = Z[perm[:n]]
        Yp = Z[perm[n:]]
        if energy_distance(Xp, Yp) >= obs:
            count += 1
    return count / n_permutations

def align_and_fill_zeros(runs1, runs2):
    runs1 = np.array(runs1, dtype=float)
    runs2 = np.array(runs2, dtype=float)

    max_rows = max(runs1.shape[0], runs2.shape[0])
    max_cols = max(runs1.shape[1], runs2.shape[1])

    def pad_to_shape(arr, target_shape):
        padded = np.zeros(target_shape, dtype=float)
        r, c = arr.shape
        padded[:r, :c] = arr
        return padded

    runs1_padded = pad_to_shape(runs1, (max_rows, max_cols))
    runs2_padded = pad_to_shape(runs2, (max_rows, max_cols))

    return runs1_padded, runs2_padded

def encode_strings_joint(runs1, runs2):

    def parse_if_needed(runs):
        parsed = []
        for r in runs:
            if isinstance(r, str):
                try:
                    val = ast.literal_eval(r)
                    if isinstance(val, (list, tuple)):
                        parsed.append(list(val))
                    else:
                        parsed.append([val])
                except Exception:
                    parsed.append([r])
            elif r is None:
                parsed.append([0.0])
            else:
                parsed.append(list(r) if isinstance(r, (list, tuple, np.ndarray)) else [r])
        return parsed

    runs1_parsed = parse_if_needed(runs1)
    runs2_parsed = parse_if_needed(runs2)

    flat1 = [str(v if v is not None else "0") for row in runs1_parsed for v in row]
    flat2 = [str(v if v is not None else "0") for row in runs2_parsed for v in row]

    all_vals = sorted(set(flat1 + flat2))
    mapping = {val: i for i, val in enumerate(all_vals)}

    def encode(parsed):
        encoded = [[mapping[str(v if v is not None else "0")] for v in row] for row in parsed]
        return np.array(encoded, dtype=float)

    return encode(runs1_parsed), encode(runs2_parsed)

def all_none(runs):
    runs = np.asarray(runs)
    return np.all(runs == None)

def compare_files(file1, file2, repeat_times):
    data1 = load_json(file1)
    data2 = load_json(file2)
    samples1_dict = {s["sample_id"]: s for s in data1}
    results = []

    for s2_sample in data2:
        sample_id = s2_sample["sample_id"]
        func_type = s2_sample["function_type"]
        print(sample_id)
        s1_sample = samples1_dict.get(sample_id)

        for result in s1_sample["results"]:
            runs = result["runs"]
            k = len(runs) // 100
            merged_runs = []
            if k != 1 and k != 0:
                for i in range(0, len(runs), k):
                    merged_runs.append(runs[i:i+k])
            elif k == 1:
                merged_runs = runs
            else:
                continue
            if len(runs) >= repeat_times:
                result["runs"] = random.sample(merged_runs, repeat_times)
            else:
                result["runs"] = random.choices(merged_runs, k = repeat_times)

        for result in s2_sample["results"]:
            runs = result["runs"]
            k = len(runs) // 100
            merged_runs = []
            if k != 1 and k != 0:
                for i in range(0, len(runs), k):
                    merged_runs.append(runs[i:i+k])
            elif k == 1:
                merged_runs = runs
            else:
                continue
            if len(runs) >= repeat_times:
                result["runs"] = random.sample(merged_runs, repeat_times) #random.sample
            else:
                result["runs"] = random.choices(merged_runs, k = repeat_times)

        for t1, t2 in zip(s1_sample["results"], s2_sample["results"]):
            test_case = t1["test_case"]
            ks_stat, ks_p, mw_stat, mw_p, energy_val, energy_pval, T2, F_stat, p_val = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            runs1 = preprocess_runs(t1["runs"])
            runs2 = preprocess_runs(t2["runs"])

            if all_none(runs1) or all_none(runs2):
                if func_type == "original":
                    energy_pval = 1
                    energy_val = np.nan
                    ks_stat = mw_stat = ks_p = mw_p = np.nan
                else:
                    energy_pval = 0.0
                    energy_val = np.nan
                    ks_stat = mw_stat = ks_p = mw_p = np.nan
                results.append({
                    "sample_id": sample_id,
                    "function_type": func_type,
                    "test_case": test_case,
                    "KS_stat": ks_stat, "KS_p": ks_p,
                    "MW_stat": mw_stat, "MW_p": mw_p,
                    "Energy_distance": energy_val, "Energy_p": energy_pval
                })
                continue
            else:
                try:
                    runs1 = np.where(runs1 == None, 0.0, runs1).astype(float)
                    runs2 = np.where(runs2 == None, 0.0, runs2).astype(float)
                except:
                    runs1, runs2 = encode_strings_joint(runs1, runs2)
                    runs1 = np.where(runs1 == None, 0.0, runs1).astype(float)
                    runs2 = np.where(runs2 == None, 0.0, runs2).astype(float)

            if runs1.size == 0 or runs2.size == 0 or runs1.shape != runs2.shape:
                energy_pval = 0.0
                energy_val = np.nan
                ks_stat = mw_stat = ks_p = mw_p = np.nan
            else:
                if runs1.ndim == 2 and runs1.shape[1] == 1:
                    ks_stat, ks_p = ks_2samp(runs1, runs2)
                    mw_stat, mw_p = mannwhitneyu(runs1, runs2, alternative="two-sided")
                else:
                    if runs1.shape != runs2.shape:
                        runs1, runs2 = align_and_fill_zeros(runs1, runs2)
                    energy_val = energy_distance(runs1, runs2)
                    energy_pval = energy_pvalue(runs1, runs2, n_permutations=100, random_state=None)
                    ks_pvals = []
                    mw_pvals = []
                    n_dims = runs1.shape[1]
                    for j in range(n_dims):
                        col1 = runs1[:, j]
                        col2 = runs2[:, j]

                        if np.all(col1 == col1[0]) and np.all(col2 == col2[0]):
                            ks_pvals.append(1.0)
                            mw_pvals.append(1.0)
                        else:
                            ks_stat, ks_p = ks_2samp(col1, col2)
                            mw_stat, mw_p = mannwhitneyu(col1, col2, alternative="two-sided")
                            ks_pvals.append(ks_p)
                            mw_pvals.append(mw_p)
                    ks_p = ks_pvals
                    mw_p = mw_pvals

            results.append({
                "sample_id": sample_id,
                "function_type": func_type,
                "test_case": test_case,
                "KS_stat": ks_stat, "KS_p": ks_p,
                "MW_stat": mw_stat, "MW_p": mw_p,
                "Energy_distance": energy_val, "Energy_p": energy_pval
            })
    return results

file1 = r"your_path\results_bm.json"
file2 = r"your_path\results_acc.json"#
repeat_times = 100
stats_results = compare_files(file1, file2, repeat_times)

res_json_path = r"your_path\statistical_results_acc.json"
with open(res_json_path, 'w', encoding='utf-8') as f:
    json.dump(stats_results, f, indent=4, ensure_ascii=False, default=str)
