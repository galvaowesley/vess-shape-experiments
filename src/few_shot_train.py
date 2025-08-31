import csv
import subprocess
import random
import os
import yaml

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
    """
    print("[INFO] Reading names from CSV...")
    names = read_names_from_csv(csv_path)
    used_combinations = set() if not with_replacement else None
    print(f"[INFO] Starting experiments: min_samples={min_samples}, max_samples={max_samples}, runs={runs}, reps={reps}, with_replacement={with_replacement}, step={step}")

    num_samples = min_samples
    report_dir = os.path.join(output_dir, params['experiment_name'])
    report_path = os.path.join(report_dir, 'selected_samples_report.csv')
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
                command_args = ["python",train_py_path] + dict_to_argv(params, ["dataset_path", "dataset_class", "model_class"])
                print(f"[INFO]    [rep {rep_idx}] Running: {' '.join(command_args)}")
                subprocess.run(command_args)
                for name in selected_samples:
                    report_rows.append([
                        params["wandb_group"], num_samples, run_number+1, rep_idx+1, name
                    ])
                # Inference handling
                if enable_inference and test_params is not None:
                    if batch_inference:
                        # Defer inference: store run name for later batch
                        pending_inference_runs.append(params["run_name"])
                    else:
                        # Immediate (legacy) inference
                        test_params_local = test_params.copy()
                        experiment_name = params.get('experiment_name', 'experiment')
                        run_name = params["run_name"]
                        run_path = os.path.join(output_dir, experiment_name, run_name)
                        test_params_local['run_path'] = run_path
                        for k in ["dataset_path", "dataset_class", "model_class"]:
                            if k in params and k not in test_params_local:
                                test_params_local[k] = params[k]
                                print(f"[INFO]    [rep {rep_idx}] Inheriting {k}: {params[k]}")
                        test_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test.py"))
                        print(f"[INFO]    [rep {rep_idx}] Running inference for {run_path} (immediate mode)")
                        run_inference(
                            test_params=test_params_local,
                            test_py_path=test_py_path,
                            delete_checkpoint=delete_checkpoint
                        )

            save_report(report_path, report_rows)
            print(f"[INFO]   Saved global report to {report_path}")
        # Batch inference for this num_samples (if enabled)
        if enable_inference and batch_inference and test_params is not None and pending_inference_runs:
            print(f"[INFO] Starting batch inference for num_samples={num_samples} | total runs: {len(pending_inference_runs)}")
            test_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test.py"))
            experiment_name = params.get('experiment_name', 'experiment')
            for idx, run_name in enumerate(pending_inference_runs, 1):
                try:
                    run_path = os.path.join(output_dir, experiment_name, run_name)
                    test_params_local = test_params.copy()
                    test_params_local['run_path'] = run_path
                    # Preserve user choice of checkpoint_type (last/best) if provided
                    for k in ["dataset_path", "dataset_class", "model_class"]:
                        if k in params and k not in test_params_local:
                            test_params_local[k] = params[k]
                    print(f"[INFO]    [batch {idx}/{len(pending_inference_runs)}] Inference for {run_path}")
                    run_inference(
                        test_params=test_params_local,
                        test_py_path=test_py_path,
                        delete_checkpoint=delete_checkpoint,
                    )
                except Exception as e:
                    print(f"[ERROR]    Batch inference failed for {run_name}: {e}")
            print(f"[INFO] Finished batch inference for num_samples={num_samples}")
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
