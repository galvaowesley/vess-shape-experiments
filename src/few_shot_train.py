import csv
import subprocess
import random
import os
import yaml
import sys

from torchtrainer.util.train_util import dict_to_argv
from pathlib import Path


def read_names_from_csv(csv_path):
    """
    Reads a CSV file and returns a list of names (removes extension if present).

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        list: List of names (strings) from the first column of the CSV, without extension.
    """
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        return [Path(row[0]).stem for row in reader if row]


def save_report(report_path, report_rows):
    """
    Saves experiment information to a single CSV file in the 'report' folder, appending new rows.
    Each row contains: wandb_group, num_samples, run_number, rep_idx, file_name.

    Args:
        report_path (str): Path to the CSV report file.
        report_rows (list of list): List of rows to append, where each row contains experiment information.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    file_exists = os.path.exists(report_path)
    with open(report_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["wandb_group", "num_samples", "run_number", "rep_idx", "file_name"])
        writer.writerows(report_rows)

def run_experiments(
    params,
    csv_path,
    min_samples=1,
    max_samples=20,
    runs=10,
    reps=5,
    with_replacement=False,
    output_dir="experiments",
    step=1,
    weights_id=None,
    test_params=None,
    delete_checkpoint=False,
    batch_inference=False,
    enable_inference=True,
    force_headless=False,
    skip_boxplot=True,
    max_inference_retries=1,
    delete_only_on_success=True,
    aggregate_inference_means=True,
):
    """
    Runs a series of training experiments with different splits of the dataset, supporting both with and without replacement.
    The number of samples in the split can be incremented by a custom step, always starting with 1 (one-shot).

    Args:
        params (dict): Fixed training parameters (mutated in-place for each run/rep).
        csv_path (str): Path to CSV containing available file names.
        min_samples (int): Minimum number of samples to start (inclusive).
        max_samples (int): Maximum number of samples to use (inclusive).
        runs (int): Number of random splits per num_samples.
        reps (int): Number of repetitions (different seeds) per split.
        with_replacement (bool): If True sample with replacement; else without.
        output_dir (str): Base directory for experiments.
        step (int): Increment step for num_samples after initial increments.
        weights_id (str): Identifier appended in run naming.
        test_params (dict|None): Parameters for inference (optional). If None or enable_inference=False, no inference.
        delete_checkpoint (bool): If True delete checkpoint (last/best as chosen) after inference.
        batch_inference (bool): If True, postpone inference until all runs/reps for a num_samples
            value are completed, then infer sequentially for every produced run. If False (default
            legacy behaviour), inference runs immediately after each training repetition.
    enable_inference (bool): Master switch. If False, skip any inference even if test_params provided.
    force_headless (bool): If True, sets headless backend for matplotlib in inference subprocess.
    skip_boxplot (bool): If True, skip boxplot generation in test script.
    max_inference_retries (int): Number of retry attempts for failed inference runs.
        delete_only_on_success (bool): If True, only delete checkpoint if inference succeeded.
        aggregate_inference_means (bool): If True, append mean metrics of each successful inference
            (from metrics_stats.csv) into a consolidated CSV at report level.
    """
    print("[INFO] Reading names from CSV...")
    names = read_names_from_csv(csv_path)
    used_combinations = set() if not with_replacement else None
    print(f"[INFO] Starting experiments: min_samples={min_samples}, max_samples={max_samples}, runs={runs}, reps={reps}, with_replacement={with_replacement}, step={step}")

    num_samples = min_samples
    report_dir = os.path.join(output_dir, params['experiment_name'])
    report_path = os.path.join(report_dir, 'selected_samples_report.csv')
    # Paths for status logging
    os.makedirs(report_dir, exist_ok=True)
    inference_status_path = os.path.join(report_dir, 'inference_status.csv')
    aggregated_metrics_path = os.path.join(report_dir, 'inference_metrics_mean.csv')
    # Armazena metadados para enriquecer métricas agregadas
    run_metadata = {}

    def append_inference_status(rows):
        """Append status rows; inject dataset_class column automatically."""
        header_needed = not os.path.exists(inference_status_path)
        os.makedirs(os.path.dirname(inference_status_path), exist_ok=True)
        dataset_cls = params.get('dataset_class', '')
        # Transform rows to include dataset_class after run_name
        transformed = []
        for r in rows:
            # Expected original format:
            # [run_name, num_samples, attempt, status, checkpoint_type, timestamp, message]
            # New format:
            # [run_name, dataset_class, num_samples, attempt, status, checkpoint_type, timestamp, message]
            if len(r) == 7:
                new_r = [r[0], dataset_cls] + r[1:]
            else:
                # Fallback: keep as-is if unexpected length
                new_r = r
            transformed.append(new_r)
        with open(inference_status_path, 'a', newline='') as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["run_name", "dataset_class", "num_samples", "attempt", "status", "checkpoint_type", "timestamp", "message"])
            w.writerows(transformed)

    def validate_inference_artifacts(run_path):
        inf_dir = os.path.join(run_path, 'inference_results')
        metrics_file = os.path.join(inf_dir, 'metrics.csv')
        stats_file = os.path.join(inf_dir, 'metrics_stats.csv')
        if not (os.path.exists(metrics_file) and os.path.exists(stats_file)):
            return False, f"Missing metrics files (metrics={os.path.exists(metrics_file)}, stats={os.path.exists(stats_file)})"
        # Basic non-empty check
        if os.path.getsize(metrics_file) == 0 or os.path.getsize(stats_file) == 0:
            return False, "Empty metrics file(s)"
        return True, "ok"

    def run_inference_with_retries(run_name, experiment_name, attempt_offset=0):
        """Attempts inference with retries for a single run. Returns final status bool."""
        run_path = os.path.join(output_dir, experiment_name, run_name)
        # Build base test params
        test_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test.py"))
        attempts = 0
        success = False
        checkpoint_type = (test_params.get('checkpoint_type') if test_params else 'last') if test_params else 'last'
        while attempts <= max_inference_retries and not success:
            attempts += 1
            attempt_num = attempt_offset + attempts
            test_params_local = test_params.copy() if test_params else {}
            test_params_local['run_path'] = run_path
            for k in ["dataset_path", "dataset_class", "model_class"]:
                if k in params and k not in test_params_local:
                    test_params_local[k] = params[k]
            # Pass control flags
            if force_headless:
                test_params_local['force_headless'] = ''  # flag style
            if skip_boxplot:
                test_params_local['skip_boxplot'] = ''
            # Remove flags que não pertencem ao parser de test.py (evita erro de argparse)
            allowed_keys = {
                'run_path','dataset_path','dataset_class','resize_size','dataset_params',
                'save_inference_images','inference_dir_name','model_class','model_params',
                'checkpoint_type','seed','device','use_amp','deterministic','benchmark',
                'tta_type','threshold','force_headless','skip_boxplot'
            }
            test_params_local = {k:v for k,v in test_params_local.items() if k in allowed_keys}
            positional_args = ["dataset_path", "dataset_class", "model_class"]
            command_args = ["python", test_py_path] + dict_to_argv(test_params_local, positional_args)
            print(f"[INFO]       Inference attempt {attempt_num} for {run_name}: {' '.join(command_args)}")
            # Run subprocess capturing return code
            proc = subprocess.run(command_args, capture_output=True, text=True)
            if proc.returncode != 0:
                msg = f"returncode={proc.returncode}; stderr_head={(proc.stderr or '').splitlines()[:3]}"
                append_inference_status([[run_name, num_samples, attempt_num, 'failed', checkpoint_type, str(__import__('datetime').datetime.now()), msg]])
                continue
            # Validate artifacts
            valid, vmsg = validate_inference_artifacts(run_path)
            if not valid:
                append_inference_status([[run_name, num_samples, attempt_num, 'failed', checkpoint_type, str(__import__('datetime').datetime.now()), vmsg]])
                continue
            append_inference_status([[run_name, num_samples, attempt_num, 'success', checkpoint_type, str(__import__('datetime').datetime.now()), '']])
            success = True
            # Delete checkpoint if eligible
            if delete_checkpoint and (not delete_only_on_success or (delete_only_on_success and success)):
                ckpt_file = None
                if checkpoint_type == 'last':
                    ckpt_file = os.path.join(run_path, 'checkpoint.pt')
                elif checkpoint_type == 'best':
                    ckpt_file = os.path.join(run_path, 'best_model.pt')
                if ckpt_file and os.path.exists(ckpt_file):
                    print(f"[INFO]       Deleting checkpoint after success: {ckpt_file}")
                    try:
                        os.remove(ckpt_file)
                    except Exception as e:
                        print(f"[WARN]       Failed to delete checkpoint: {e}")
            # Aggregate mean metrics
            if aggregate_inference_means:
                stats_file = os.path.join(run_path, 'inference_results', 'metrics_stats.csv')
                if os.path.exists(stats_file):
                    try:
                        with open(stats_file, 'r', newline='') as sf:
                            reader = csv.reader(sf)
                            rows = list(reader)
                        if rows:
                            header = rows[0]
                            mean_row = None
                            for r in rows[1:]:
                                if r and r[0] == 'mean':
                                    mean_row = r
                                    break
                            if mean_row:
                                # Metadados (se treinamento foi bem sucedido)
                                meta = run_metadata.get(run_name, ("", ""))
                                wandb_group_val, model_class_val = meta
                                # Prepare aggregated row
                                header_out = ['run_name', 'num_samples', 'wandb_group', 'model_class'] + header[1:]
                                data_out = [run_name, num_samples, wandb_group_val, model_class_val] + mean_row[1:]
                                write_header = not os.path.exists(aggregated_metrics_path)
                                with open(aggregated_metrics_path, 'a', newline='') as agf:
                                    w = csv.writer(agf)
                                    if write_header:
                                        w.writerow(header_out)
                                    w.writerow(data_out)
                    except Exception as e:
                        print(f"[WARN]       Failed to aggregate metrics for {run_name}: {e}")
        if not success:
            print(f"[WARN]    Inference failed after {attempts} attempt(s) for {run_name}")
        return success

    while num_samples <= max_samples:
        print(f"\n[INFO] === num_samples: {num_samples} ===")
        # Collector for deferred batch inference (run names) for this num_samples
        pending_inference_runs = []
        for run_number in range(runs):
            print(f"[INFO]  Run number: {run_number}")
            random.seed(run_number + num_samples * 1000)
            if with_replacement:
                selected_samples = random.choices(names, k=num_samples)
            else:
                # Guarantee unique combinations without replacement
                attempts = 0
                while True:
                    selected_samples = tuple(sorted(random.sample(names, num_samples)))
                    if selected_samples not in used_combinations:
                        used_combinations.add(selected_samples)
                        break
                    attempts += 1
                    if attempts > 1000:
                        print(f"[WARN] Could not find unique combination for num_samples={num_samples}, run_number={run_number}")
                        break
            print(f"[INFO]   Selected files: {selected_samples}")
            params["split_strategy"] = ",".join(selected_samples)
            prefix = (
                f"{params['model_class']}_weights_id:{weights_id}"
            )
            report_rows = []
            for rep_idx in range(reps):
                params["run_name"] = f"{prefix}_run:{run_number+1}_rep:{rep_idx + 1}_ns:{num_samples}"
                params["seed"] = rep_idx
                params["wandb_group"] = f"{params['model_class']} | lr:{params['lr']} | weights_id:{weights_id} | n_samples:{num_samples}"
                train_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "train.py"))
                # Execução direta (herda stdout/stderr) para preservar barras de progresso (tqdm usa \r sem newline)
                command_args = ["python", "-u", train_py_path] + dict_to_argv(params, ["dataset_path", "dataset_class", "model_class"])
                print(f"[INFO]    [rep {rep_idx}] Running: {' '.join(command_args)}")
                proc = subprocess.run(command_args)
                if proc.returncode != 0:
                    print(f"[ERROR]    Training failed (run_name={params['run_name']}) returncode={proc.returncode}")
                    fail_log = os.path.join(report_dir, 'train_failures.log')
                    with open(fail_log, 'a') as fl:
                        fl.write(f"{params['run_name']};returncode={proc.returncode};stderr_tail=[]\n")
                    for name in selected_samples:
                        report_rows.append([
                            params["wandb_group"], num_samples, run_number+1, rep_idx+1, name
                        ])
                    continue
                for name in selected_samples:
                    report_rows.append([
                        params["wandb_group"], num_samples, run_number+1, rep_idx+1, name
                    ])
                # Registrar metadados para agregação de métricas
                run_metadata[params["run_name"]] = (params["wandb_group"], params["model_class"])
                # Inference handling
                if enable_inference and test_params is not None:
                    if batch_inference:
                        # Defer inference: store run name for later batch
                        pending_inference_runs.append(params["run_name"])
                    else:
                        # Immediate inference with retry logic
                        experiment_name = params.get('experiment_name', 'experiment')
                        print(f"[INFO]    [rep {rep_idx}] Immediate inference (with retries) for {params['run_name']}")
                        run_inference_with_retries(params['run_name'], experiment_name)

            save_report(report_path, report_rows)
            print(f"[INFO]   Saved global report to {report_path}")
        # Batch inference for this num_samples (if enabled)
        if enable_inference and batch_inference and test_params is not None and pending_inference_runs:
            experiment_name = params.get('experiment_name', 'experiment')
            print(f"[INFO] Starting batch inference for num_samples={num_samples} | total runs: {len(pending_inference_runs)}")
            failed = []
            # First pass
            for idx, run_name in enumerate(pending_inference_runs, 1):
                print(f"[INFO]    [batch pass1 {idx}/{len(pending_inference_runs)}] {run_name}")
                ok = run_inference_with_retries(run_name, experiment_name, attempt_offset=0)
                if not ok:
                    failed.append(run_name)
            # Summary and optional second pass limited to failed (already retried inside function though)
            summary_path = os.path.join(report_dir, f'group_summary_{num_samples}.txt')
            with open(summary_path, 'w') as sf:
                sf.write(f"num_samples={num_samples}\n")
                sf.write(f"total_runs={len(pending_inference_runs)}\n")
                sf.write(f"failed_runs={len(failed)}\n")
                if failed:
                    sf.write("failed_list=" + ",".join(failed) + "\n")
            if failed:
                print(f"[WARN] Batch inference group num_samples={num_samples} finished with {len(failed)} failure(s). See {summary_path}")
            else:
                print(f"[INFO] Finished batch inference for num_samples={num_samples} with all successes.")
        if num_samples == 1:
            num_samples += 1
        else:
            num_samples += step

def run_inference(test_params, test_py_path, delete_checkpoint=False):
    """
    Run inference using test.py and parameters from config.yaml.

    Args:
        test_params (dict): Dictionary with test parameters.
        test_py_path (str): Path to the test.py script.
    """
   
    positional_args = ["dataset_path", "dataset_class", "model_class"]
    command_args = ["python", test_py_path] + dict_to_argv(test_params, positional_args)
    print(f"[INFO] Running inference: {' '.join(command_args)}")
    subprocess.run(command_args)

    # Deletar checkpoint se solicitado
    if delete_checkpoint:
        run_path = test_params.get("run_path", None)
        checkpoint_type = test_params.get("checkpoint_type", "last")
        if run_path:
            if checkpoint_type == "last":
                ckpt_file = os.path.join(run_path, "checkpoint.pt")
            elif checkpoint_type == "best":
                ckpt_file = os.path.join(run_path, "best_model.pt")
            else:
                ckpt_file = None
            if ckpt_file and os.path.exists(ckpt_file):
                print(f"[INFO] Deleting checkpoint: {ckpt_file}")
                os.remove(ckpt_file)
            else:
                print(f"[WARN] Checkpoint not found: {ckpt_file}")

def load_params_from_yaml(yaml_path):
    """
    Loads experiment, training and test parameters from a YAML configuration file with three sections:
    - train_params: training parameters
    - experiment_params: experiment loop parameters
    - test_params: test/inference parameters

    Args:
        yaml_path (str): Path to the YAML config file.

    Returns:
        tuple: (train_params: dict, experiment_params: dict, test_params: dict)
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['train_params'], config['experiment_params'], config.get('test_params', {})
