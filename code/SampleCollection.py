import json
import numpy as np
import subprocess
import sys
import multiprocessing
import traceback

def _target_func(function_code: str, test_case: str, queue: multiprocessing.Queue):
    try:
        local_env = {}
        exec(function_code, local_env)  
        result = eval(test_case, local_env)  
        queue.put(("ok", result))
    except Exception:
        queue.put(("error", traceback.format_exc()))

def pad_and_stack(arrays):
    flattened = [a.ravel() for a in arrays]
    all_numeric = all(np.issubdtype(arr.dtype, np.number) for arr in flattened)
    max_len = max(arr.shape[0] for arr in flattened)

    padded = []
    if all_numeric:
        for arr in flattened:
            pad_width = max_len - arr.shape[0]
            if pad_width > 0:
                arr = np.pad(arr, (0, pad_width), constant_values=np.nan)
            padded.append(arr.astype(float))
        return np.vstack(padded)
    else:
        for arr in flattened:
            pad_width = max_len - arr.shape[0]
            if pad_width > 0:
                arr = np.concatenate([arr.astype(object), np.array([None] * pad_width, dtype=object)])
            else:
                arr = arr.astype(object)
            padded.append(arr)
        return np.vstack(padded, dtype=object)


def safe_to_array(out):
    try:
        if isinstance(out, (tuple, list, np.ndarray)):
            return np.array(out, dtype=float)
        else:
            return np.array([out], dtype=float)
    except (ValueError, TypeError):
        return np.array([str(out)], dtype=object)


def run_single_function(item, repeat=10):
    func_code = item["function"]
    test_inputs = item["input"].splitlines()  
    exec_env = {}
    try:
        try:
            exec(func_code, exec_env)
        except ImportError as e:
            msg = str(e)
            if "No module named" in msg:
                pkg = msg.split("'")[1]
                print(f"[INFO] Missing package {pkg}, installing...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                    exec(func_code, exec_env)
                except Exception as install_err:
                    print(f"[ERROR] Failed to install {pkg}: {install_err}. Skipping this sample.")
                    return False
            else:
                print(f"[ERROR] ImportError not due to missing package: {e}")
                return False

        sample_results = []

        for test in test_inputs:
            test = test.strip()
            if not test:
                continue

            run_outputs = []
            for _ in range(repeat):
                try:
                    out = eval(test, exec_env)
                    vec = safe_to_array(out)
                except Exception as e:
                    print(f"[WARN] Test '{test}' failed during eval: {e}. Skipping this run.")
                    vec = np.array([np.nan])

                run_outputs.append(vec)

            try:
                matrix = np.vstack(run_outputs)
            except ValueError:
                matrix = pad_and_stack(run_outputs)

            json_runs = []
            for row in matrix:
                json_row = []
                for x in row:
                    if isinstance(x, (np.floating, float, int)):
                        json_row.append(None if (isinstance(x, float) and np.isnan(x)) else x)
                    elif x is None:
                        json_row.append(None)
                    else:
                        json_row.append(str(x))  
                json_runs.append(json_row)

            sample_results.append({
                "test_case": test,
                "runs": json_runs
            })

        return  sample_results

    except Exception as e:
        print(f"[WARN] Sample id={item.get('id', '?')} skipped due to error: {e}")
        return False

def run_function_tests(input_json, output_json, repeat=10):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = []

    for idx, item in enumerate(data):
        print(f"\n[INFO] Running sample {idx + 1}/{len(data)}, id={item.get('id')}")
        mutants = item.get("mutants", [])
        for m_idx, mutant in enumerate(mutants):
            m_type = mutant.get("mutation_type", f"mutant_{m_idx}")
            print(f"    ↳ Running mutant {m_idx + 1}/{len(mutants)}: {m_type}")

            mutant_item = dict(item)
            mutant_item["function"] = mutant.get("mutated_function", "")

            m_results = run_single_function(mutant_item, repeat=repeat)
            if m_results is False:
                continue
            all_results.append({
                "sample_id": item.get("id"),
                "function_type": m_type,
                "results": m_results
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n✅ All samples (including mutants) processed and saved.")


if __name__ == "__main__":

    input_json = r"your_path\mutant_dataset_acc.json"  
    output_json = r"your_path\results_acc.json" 
    multiprocessing.set_start_method("spawn", force=True)
    run_function_tests(input_json, output_json, repeat=100)