import csv
import subprocess
import random
import os
import yaml
from train import VesselTrainer
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

def run_experiments(params, csv_path, min_samples=1, max_samples=20, runs=10, reps=5, with_replacement=False, output_dir="experiments", step=1, weights_id=None):
    """
    Runs a series of training experiments with different splits of the dataset, supporting both with and without replacement.
    The number of samples in the split can be incremented by a custom step, always starting with 1 (one-shot).

    Args:
        params (dict): Dictionary of fixed parameters for training.
        csv_path (str): Path to the CSV file with the list of available files.
        min_samples (int): Minimum number of samples to start with (default: 1).
        max_samples (int): Maximum number of samples to use (exclusive).
        runs (int): Number of different random splits per num_samples.
        reps (int): Number of repetitions for each split.
        with_replacement (bool): If True, samples with replacement; else, without replacement.
        output_dir (str): Directory to save the selected names CSVs.
        step (int): Step size for incrementing num_samples after the first two steps.
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
        for run_number in range(runs):
            print(f"[INFO]  Run number: {run_number}")
            random.seed(run_number + num_samples * 1000)
            if with_replacement:
                selected_samples = random.choices(names, k=num_samples)
            else:
                # Garantee unique combinations without replacement
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
            # report_rows agora é por rodada
            report_rows = []
            for rep_idx in range(reps):
                params["run_name"] = f"{prefix}_run:{run_number+1}_rep:{rep_idx + 1}_ns:{num_samples}"
                params["seed"] = rep_idx
                params["wandb_group"] = f"{params['model_class']} | lr:{params['lr']} | weights_id:{weights_id} | n_samples:{num_samples}"
                command_args = ["python", "train.py"] + dict_to_argv(params, ["dataset_path", "dataset_class", "model_class"])
                print(f"[INFO]    [rep {rep_idx}] Running: {' '.join(command_args)}")
                subprocess.run(command_args)
                for name in selected_samples:
                    report_rows.append([
                        params["wandb_group"], num_samples, run_number+1, rep_idx+1, name
                    ])
            # Salva o relatório global apenas com os registros da rodada
            save_report(report_path, report_rows)
            print(f"[INFO]   Saved global report to {report_path}")
        if num_samples == 1:
            num_samples += 1
        else:
            num_samples += step

def load_params_from_yaml(yaml_path):
    """
    Loads experiment and training parameters from a YAML configuration file with two sections:
    - train_params: training parameters
    - experiment_params: experiment loop parameters

    Args:
        yaml_path (str): Path to the YAML config file.

    Returns:
        tuple: (train_params: dict, experiment_params: dict)
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['train_params'], config['experiment_params']

if __name__ == "__main__":
    config_path = "config.yaml" 
    train_params, experiment_params = load_params_from_yaml(config_path)
    run_experiments(
        train_params,
        experiment_params['csv_path'],
        min_samples=experiment_params.get('min_samples', 1),
        max_samples=experiment_params.get('max_samples', 20),
        runs=experiment_params.get('runs', 10),
        reps=experiment_params.get('reps', 5),
        with_replacement=experiment_params.get('with_replacement', False),
        output_dir=experiment_params.get('output_dir', 'runs'),
        step=experiment_params.get('step', 1),
        weights_id=experiment_params.get('weights_id', None)
    )