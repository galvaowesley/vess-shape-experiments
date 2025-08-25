import os
import re
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def get_experiments_grouped_stats(df, list_columns, metric_column, agg_funcs):
    """
    Get experiments grouped statistics from the models_results DataFrame.
    Returns a DataFrame with the aggregated results.
    
    Args:
        df (pd.DataFrame): DataFrame containing the results of the experiments
        list_columns (list): List of columns to group by
        metric_column (str): Column to aggregate
        agg_funcs (list): List of aggregation functions to apply
    Returns:
        pd.DataFrame: DataFrame containing the aggregated results
    """
    agg_stats = df.groupby(list_columns)[metric_column].agg(agg_funcs).reset_index()
    # Evita redundância nos nomes das colunas
    def clean_col_name(func, metric_column):
        # Se metric_column já começa com func, retorna metric_column
        if metric_column.lower().startswith(func.lower()):
            return metric_column
        # Se metric_column já contém func, remove o prefixo duplicado
        if metric_column.lower().startswith('mean_') and func != 'mean':
            return f"{func}_{metric_column[5:]}"  # Remove 'mean_' do início
        return f"{func}_{metric_column}"
    agg_stats = agg_stats.rename(columns={func: clean_col_name(func, metric_column) for func in agg_funcs})
    agg_stats = agg_stats.sort_values(by=list_columns, ascending=False)
    agg_stats = agg_stats.reset_index(drop=True)
    return agg_stats


def plot_mean_dice_score(dataframe, dataset_name='VessMap'):
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=dataframe,
        x='samples',
        y='mean_Dice',
        hue='model_type',
        marker='o',
        errorbar='sd'
    )
    plt.title(f'{dataset_name} - Mean Dice Score by Samples and Model Class')
    plt.xlabel('Number of Samples')
    plt.ylabel('Mean Dice Score')
    plt.legend(title='Model Class')
    plt.grid(True)
    plt.tight_layout()

# Ajuste da escala do eixo x
    max_x = dataframe['samples'].max()
    plt.xticks(range(0, max_x + 1, 2))

    plt.show()

