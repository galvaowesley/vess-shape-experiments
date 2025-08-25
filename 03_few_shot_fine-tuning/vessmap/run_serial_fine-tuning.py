import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.serial_training import run_experiments, load_params_from_yaml

if __name__ == "__main__":
    config_path = "config.yaml"
    train_params, experiment_params, test_params = load_params_from_yaml(config_path)
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
        weights_id=experiment_params.get('weights_id', None),
        test_params=test_params,
        delete_checkpoint=experiment_params.get('delete_checkpoint', False)
)