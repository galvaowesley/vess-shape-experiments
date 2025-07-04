import csv
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

def save_selected_names(selected, output_dir, run_number, num_samples):
    """
    Saves the selected names for a given run_number and num_samples to a CSV file in the output directory.

    Args:
        selected (list): List of selected names.
        output_dir (str): Directory to save the CSV file.
        run_number (int): The current run number.
        num_samples (int): The number of samples in this run.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"selected_names_run{run_number}_n{num_samples}.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for name in selected:
            writer.writerow([name])

def save_selected_names_report(report_path, report_rows):
    """
    Saves a report of all selected names for all runs and num_samples in a single CSV file.
    Each row contains: run_number, num_samples, file_name
    """
    with open(report_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_number", "num_samples", "file_name"])
        writer.writerows(report_rows)

def run_experiments(params, csv_path, min_samples=1, max_samples=20, runs=10, reps=5, with_replacement=False, output_dir="experiments", step=1):
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

    report_rows = []
    num_samples = min_samples
    while num_samples < max_samples:
        print(f"\n[INFO] === num_samples: {num_samples} ===")
        for run_number in range(runs):
            print(f"[INFO]  Run number: {run_number}")
            random.seed(run_number + num_samples * 1000)
            if with_replacement:
                selected = random.choices(names, k=num_samples)
            else:
                # Garantee unique combinations without replacement
                attempts = 0
                while True:
                    selected = tuple(sorted(random.sample(names, num_samples)))
                    if selected not in used_combinations:
                        used_combinations.add(selected)
                        break
                    attempts += 1
                    if attempts > 1000:
                        print(f"[WARN] Could not find unique combination for num_samples={num_samples}, run_number={run_number}")
                        break
            print(f"[INFO]   Selected files: {selected}")
            params["split_strategy"] = ",".join(selected)
            # Build the prefix for the run name
            prefix = (
                f"{params['model_class']}_bs:{params['bs_train']}_ep:{params['num_epochs']}"
                f"_lr:{params['lr']}_lr-decay:{params['lr_decay']}_wd:{params['weight_decay']}"
                f"_val-metric:{params['validation_metric']}"
            )
            # Define the experiment_dir as output_dir/experiment_name/prefix
            experiment_dir = os.path.join(output_dir, params['experiment_name'], prefix)
            # Save the selected names for this run_number
            save_selected_names(selected, experiment_dir, run_number, num_samples)
            print(f"[INFO]   Saved selected names to {experiment_dir}/selected_names_run{run_number}_n{num_samples}.csv")
            for rep_idx in range(reps):
                params["run_name"] = f"{prefix}_r:{run_number}_s:{rep_idx}_n:{num_samples}"
                params["seed"] = rep_idx
                commandline = ' '.join(dict_to_argv(params, ["dataset_path", "dataset_class", "model_class"]))
                print(f"[INFO]    [rep {rep_idx}] Running: python train.py")
                os.system(f"python train.py {commandline}")
            for name in selected:
                report_rows.append([run_number, num_samples, name])
        # Step logic: starts with 1, then 2, then step in step
        if num_samples == 1:
            num_samples += 1
        else:
            num_samples += step
    # Salva o relatório único ao final
    report_path = os.path.join(output_dir, params['experiment_name'], 'selected_names_report.csv')
    save_selected_names_report(report_path, report_rows)
    print(f"[INFO]   Saved global report to {report_path}")

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
        step=experiment_params.get('step', 1)
    )