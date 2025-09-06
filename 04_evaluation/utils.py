import os
import re
import csv
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
except ImportError:  # Permite uso do restante mesmo sem plotly
    px = None


def get_experiments_raw_stats(multi_finetune_path):
    """
    Get experiments statistics from the multi-finetune path.
    Returns a DataFrame with the results.
    
    Args:
        multi_finetune_path (str): Path to the directory containing multi-finetune experiments
    Returns:
        pd.DataFrame: DataFrame containing the results of the experiments
    """
    
    wandb_logs_path = 'wandb/latest-run/files'    
    models_results = pd.DataFrame()
    multi_finetune_dirs = [d for d in os.listdir(multi_finetune_path) if os.path.isdir(os.path.join(multi_finetune_path, d))]    
    
    
    models_results = pd.DataFrame()
    for dir_name in multi_finetune_dirs:
        dir_path = os.path.join(multi_finetune_path, dir_name)
        # Check if the directory contains a log.csv file
        if 'log.csv' in os.listdir(dir_path):
            # Read the log.csv file and append it to the DataFrame
            df = pd.read_csv(os.path.join(dir_path, 'log.csv'))
            df = df.dropna(subset=['Dice'])
        
            # Open wandb config.yaml file
            
            # Extract run, rep and ns from dir_name substring
            match = re.search(r'run:(\d+)_rep:(\d+)_ns:(\d+)', dir_name)
            
            if match:
                df['run'] = int(match.group(1))
                df['rep'] = int(match.group(2))
                df['samples'] = int(match.group(3))
            else:
                df['run'] = None
                df['rep'] = None
                df['samples'] = None
            
            df['run_name'] = dir_name  # Add a column to identify the experiment    
                    
            config_path = os.path.join(dir_path, wandb_logs_path, 'config.yaml')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    wandb_group = config.get('wandb_group', {}).get('value', None)
                    model_class = config.get('model_class', {}).get('value', None)
                    df['wandb_group'] = wandb_group
                    df['model_class'] = model_class
            else:
                df['wandb_group'] = None
                df['model_class'] = None
            models_results = pd.concat([models_results, df], ignore_index=True)
                        
    return models_results


def get_experiments_test_metrics(multi_finetune_path):
    """Read aggregated test metrics (the 'mean' row of inference_results/metrics_stats.csv)
    for each experiment directory.

    Returns a DataFrame with columns: Accuracy, IoU, Precision, Recall, Dice, run, rep, samples,
    run_name, wandb_group, model_class. (Epoch/step information is not included.)

    If the file or the 'mean' row does not exist the experiment is skipped.
    """
    wandb_logs_path = 'wandb/latest-run/files'
    test_metrics_all = []
    multi_finetune_dirs = [
        d for d in os.listdir(multi_finetune_path)
        if os.path.isdir(os.path.join(multi_finetune_path, d))
    ]

    for dir_name in multi_finetune_dirs:
        dir_path = os.path.join(multi_finetune_path, dir_name)
        metrics_path = os.path.join(dir_path, 'inference_results', 'metrics_stats.csv')
        if not os.path.exists(metrics_path):
            continue

        try:
            df_metrics = pd.read_csv(metrics_path)
        except Exception:
            continue  # pula arquivos corrompidos
        if 'statistic' not in df_metrics.columns:
            continue
        mean_row = df_metrics[df_metrics['statistic'] == 'mean']
        if mean_row.empty:
            continue
        mean_row = mean_row.iloc[0]

        # Extrai identificadores de diretório
        match = re.search(r'run:(\d+)_rep:(\d+)_ns:(\d+)', dir_name)
        if match:
            run = int(match.group(1))
            rep = int(match.group(2))
            samples = int(match.group(3))
        else:
            run = rep = samples = None

        # Lê config para wandb_group e model_class
        config_path = os.path.join(dir_path, wandb_logs_path, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                wandb_group = config.get('wandb_group', {}).get('value', None)
                model_class = config.get('model_class', {}).get('value', None)
        else:
            wandb_group = None
            model_class = None

        # Subconjunto de métricas desejadas (se existirem)
        metric_names = ['Accuracy', 'IoU', 'Precision', 'Recall', 'Dice']
        metrics_dict = {m: mean_row[m] for m in metric_names if m in mean_row.index}
        metrics_dict.update({
            'run': run,
            'rep': rep,
            'samples': samples,
            'run_name': dir_name,
            'wandb_group': wandb_group,
            'model_class': model_class,
        })
        test_metrics_all.append(metrics_dict)

    if not test_metrics_all:
        return pd.DataFrame(columns=['Accuracy', 'IoU', 'Precision', 'Recall', 'Dice', 'run', 'rep', 'samples', 'run_name', 'wandb_group', 'model_class'])
    return pd.DataFrame(test_metrics_all)


# Combined helper removed per user request; use separate functions instead.


def get_experiments_grouped_stats(df, list_columns, metric_column, agg_funcs):
    """Group experimental results and compute aggregations.

    Args:
        df (pd.DataFrame): DataFrame containing experiment results.
        list_columns (list): Columns to group by.
        metric_column (str): Metric column to aggregate.
        agg_funcs (list): Aggregation functions (e.g. ['mean', 'std']).
    Returns:
        pd.DataFrame: Aggregated statistics with cleaned column names.
    """
    agg_stats = df.groupby(list_columns)[metric_column].agg(agg_funcs).reset_index()
    # Avoid redundant prefixes in column names
    def clean_col_name(func, metric_column):
        # If metric_column already begins with the aggregation name, keep it
        if metric_column.lower().startswith(func.lower()):
            return metric_column
        # Remove duplicated 'mean_' prefix when adding another function
        if metric_column.lower().startswith('mean_') and func != 'mean':
            return f"{func}_{metric_column[5:]}"
        return f"{func}_{metric_column}"
    agg_stats = agg_stats.rename(columns={func: clean_col_name(func, metric_column) for func in agg_funcs})
    agg_stats = agg_stats.sort_values(by=list_columns, ascending=False)
    agg_stats = agg_stats.reset_index(drop=True)
    return agg_stats


def plot_mean_dice_score(dataframe, dataset_name='VessMap', hue='model_type', x_data='num_samples', y_data='Dice',
                         line_styles: dict | None = None, markers: bool = True):
    """Plot mean (with sd error bars) of a metric vs number of samples.

    Each hue category is drawn separately so we can safely assign different
    matplotlib line styles (e.g. '-', '--', '-.', ':').

    Args:
        dataframe: Input DataFrame containing metric values.
        dataset_name: String used in the plot title.
        hue: Column with categorical series identifiers.
        x_data: Name of the x-axis column.
        y_data: Name of the metric column to aggregate.
        line_styles: Optional dict mapping hue values to valid matplotlib linestyles.
        markers: Whether to plot point markers.
    """
    if hue not in dataframe.columns:
        raise ValueError(f"Hue column '{hue}' not found. Columns: {list(dataframe.columns)}")
    if x_data not in dataframe.columns:
        raise ValueError(f"x_data column '{x_data}' not found. Columns: {list(dataframe.columns)}")
    if y_data not in dataframe.columns:
        raise ValueError(f"y_data column '{y_data}' not found. Columns: {list(dataframe.columns)}")

    df = dataframe.copy()
    plt.figure(figsize=(12, 6))

    unique_vals = [v for v in df[hue].dropna().unique()]
    if not unique_vals:
        print("[WARN] No hue categories to plot.")
        return
    if line_styles is None:
        base_patterns = ['-', '--', '-.', ':']
        line_styles = {val: base_patterns[i % len(base_patterns)] for i, val in enumerate(unique_vals)}

    palette = sns.color_palette(n_colors=len(unique_vals))
    color_map = {val: palette[i] for i, val in enumerate(unique_vals)}

    # Aggregate: mean and std per (hue, x)
    grouped = df.groupby([hue, x_data])[y_data].agg(['mean', 'std']).reset_index()

    for val in unique_vals:
        sub = grouped[grouped[hue] == val]
        ls = line_styles.get(val, '-')
        color = color_map[val]
        marker_style = 'o' if markers else None
        plt.plot(sub[x_data], sub['mean'], linestyle=ls, marker=marker_style, color=color, label=val)
        # Error band (std)
        if 'std' in sub and not sub['std'].isna().all():
            plt.fill_between(sub[x_data], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                             color=color, alpha=0.15, linewidth=0)

    plt.title(f'{dataset_name} - Mean {y_data} by {x_data} and {hue}')
    plt.xlabel('Number of Samples')
    plt.ylabel(f'Mean {y_data}')
    plt.legend(title=hue)
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    # Adjust x-axis tick spacing using provided x_data. Fallbacks for legacy column names.
    candidate_cols = [x_data, 'samples', 'num_samples']
    col_found = None
    for c in candidate_cols:
        if c in dataframe.columns:
            col_found = c
            break
    if col_found is not None and pd.api.types.is_numeric_dtype(dataframe[col_found]):
        try:
            max_x = int(dataframe[col_found].max())
            step = 2 if max_x >= 8 else 1
            plt.xticks(range(0, max_x + 1, step))
        except ValueError:
            pass  # silently skip tick customization if conversion fails
    else:
        print(f"[WARN] x-axis column not found or non-numeric among: {candidate_cols}. Skipping custom ticks.")

    plt.show()


def aggregate_inference_means(experiments_root: str, output_csv: str | None = None,
                              stats_filename: str = 'metrics_stats.csv',
                              inference_dir_name: str = 'inference_results',
                              wandb_config_rel: str = 'wandb/latest-run/files/config.yaml',
                              dir_pattern: str = r'run:(\d+)_rep:(\d+)_ns:(\d+)') -> pd.DataFrame:
    """Aggregate mean inference metrics from multiple experiment run folders.

    For each subdirectory in ``experiments_root`` this looks for:
        <run_dir>/<inference_dir_name>/<stats_filename>
    If found, it parses the CSV and extracts the row whose 'statistic' value is 'mean'.

    Additionally attempts to read the wandb config (``wandb_config_rel``) to obtain
    the stored ``wandb_group`` and ``model_class`` values. The number of samples
    (``num_samples``) is parsed from the folder name using ``dir_pattern``.

    Output columns (only retained if present in the stats file):
        run_name, num_samples, wandb_group, model_class, Accuracy, IoU,
        Precision, Recall, Dice, AUC

    Args:
        experiments_root: Path containing the run subdirectories.
        output_csv: Optional path to append / write the aggregated CSV. If None, no file is written.
        stats_filename: Base name of the per-run statistics CSV.
        inference_dir_name: Name of the directory with inference results.
        wandb_config_rel: Relative path to the wandb config yaml file inside each run.
        dir_pattern: Regex capturing run, rep, samples (ns) identifiers.

    Returns:
        pd.DataFrame: Aggregated mean metrics for all discovered runs.
    """
    desired_metrics = ["Accuracy", "IoU", "Precision", "Recall", "Dice", "AUC"]
    rows = []

    if not os.path.isdir(experiments_root):
        raise FileNotFoundError(f"Experiments root not found: {experiments_root}")

    run_dirs = [d for d in os.listdir(experiments_root)
                if os.path.isdir(os.path.join(experiments_root, d))]

    for run_name in sorted(run_dirs):
        run_path = os.path.join(experiments_root, run_name)
        stats_path = os.path.join(run_path, inference_dir_name, stats_filename)
        if not os.path.exists(stats_path):
            continue
        try:
            with open(stats_path, 'r', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
            if not data:
                continue
            header = data[0]
            # Map header -> index for safety
            col_index = {col: idx for idx, col in enumerate(header)}
            if 'statistic' not in col_index:
                continue
            mean_row_vals = None
            for r in data[1:]:
                if r and r[col_index['statistic']] == 'mean':
                    mean_row_vals = r
                    break
            if mean_row_vals is None:
                continue

            # Extract num_samples via regex
            match = re.search(dir_pattern, run_name)
            if match:
                num_samples = int(match.group(3))  # third group is ns
            else:
                num_samples = None

            # Read wandb config
            wandb_group_val = None
            model_class_val = None
            config_path = os.path.join(run_path, wandb_config_rel)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as cf:
                        cfg = yaml.safe_load(cf)
                    wandb_group_val = cfg.get('wandb_group', {}).get('value', None)
                    model_class_val = cfg.get('model_class', {}).get('value', None)
                except Exception:
                    pass

            # Build output row
            out = {
                'run_name': run_name,
                'num_samples': num_samples,
                'wandb_group': wandb_group_val,
                'model_class': model_class_val,
            }
            for m in desired_metrics:
                if m in col_index:
                    try:
                        out[m] = float(mean_row_vals[col_index[m]])
                    except (ValueError, TypeError):
                        out[m] = None
                else:
                    out[m] = None
            rows.append(out)
        except Exception as e:
            print(f"[WARN] Failed to process {stats_path}: {e}")

    df = pd.DataFrame(rows, columns=['run_name', 'num_samples', 'wandb_group', 'model_class'] + desired_metrics)

    if output_csv:
        # Write header if file does not exist
        write_header = not os.path.exists(output_csv)
        df.to_csv(output_csv, mode='a' if not write_header else 'w', index=False, header=write_header)

    return df


def plot_mean_dice_score_px(dataframe: pd.DataFrame, dataset_name: str = 'VessMap',
                            hue: str = 'model_type', x_data: str = 'num_samples',
                            y_data: str = 'Dice', line_styles: dict | None = None,
                            markers: bool = True, show: bool = True):
    """Interactive version of plot_mean_dice_score using Plotly Express.

    Aggregates mean/std of y_data grouped by (hue, x_data) and plots lines with
    optional custom dash styles. Returns the Plotly Figure.

    Args:
        dataframe: Input DataFrame containing raw metric values.
        dataset_name: Dataset label used in title.
        hue: Column defining series (color + dash).
        x_data: Name of x-axis column (samples).
        y_data: Metric column name.
        line_styles: Optional mapping {hue_value: dash_style}; valid dashes include
            'solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot'.
        markers: Whether to include markers on lines.
        show: If True, immediately display (fig.show()).

    Returns:
        plotly.graph_objects.Figure
    """
    if px is None:
        raise ImportError("plotly is not installed. Install with `pip install plotly` to use plot_mean_dice_score_px().")
    for col, name in [(hue, 'hue'), (x_data, 'x_data'), (y_data, 'y_data')]:
        if col not in dataframe.columns:
            raise ValueError(f"Column '{col}' (for {name}) not found. Available: {list(dataframe.columns)}")

    df = dataframe.copy()
    grouped = df.groupby([hue, x_data])[y_data].agg(['mean', 'std']).reset_index()
    grouped = grouped.rename(columns={'mean': f'mean_{y_data}', 'std': f'std_{y_data}'})

    # Default line styles cycling
    if line_styles is None:
        dash_cycle = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
        unique_vals = list(grouped[hue].unique())
        line_styles = {val: dash_cycle[i % len(dash_cycle)] for i, val in enumerate(unique_vals)}

    # Map matplotlib-like styles to plotly accepted dash styles
    dash_map = {
        '-': 'solid',
        '--': 'dash',
        ':': 'dot',
        '-.': 'dashdot'
    }
    def _normalize_dash(style: str):
        if style is None:
            return 'solid'
        return dash_map.get(style, style)  # leave custom sequences untouched

    fig = px.line(
        grouped,
        x=x_data,
        y=f'mean_{y_data}',
        color=hue,
        line_dash=hue,
        markers=markers,
        error_y=f'std_{y_data}',
        title=f'{dataset_name} - Mean {y_data} by {x_data} and {hue}'
    )

    # Apply custom dash styles
    if line_styles:
        for trace in fig.data:
            val = trace.name
            if val in line_styles:
                trace.update(line={'dash': _normalize_dash(line_styles[val])})

    fig.update_layout(
        template='plotly_white',
        xaxis_title='Number of Samples',
        yaxis_title=f'Mean {y_data}',
        legend_title=hue,
    )

    # Adjust x ticks if numeric
    if grouped[x_data].dtype.kind in 'iuf':
        max_x = grouped[x_data].max()
        step = 2 if max_x >= 8 else 1
        fig.update_xaxes(dtick=step)

    if show:
        fig.show()
    else:
        return fig

