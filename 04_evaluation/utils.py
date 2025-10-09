import os
import re
import csv
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import seaborn as sns
from typing import List


try:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _mpl_inset_axes
except Exception:
    _mpl_inset_axes = None
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # Permite uso do restante mesmo sem plotly
    px = None
    go = None


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
def get_experiments_grouped_stats(df: pd.DataFrame, group_cols: List[str], metrics: List[str]):
    agg_map = {m: ['mean','std'] for m in metrics}
    grouped = (df
               .groupby(group_cols)[metrics]
               .agg(agg_map)
              )
    # Flatten MultiIndex de colunas para formato mean_<metric>, std_<metric>
    grouped.columns = [f"{stat}_{metric}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    return grouped



def plot_mean_dice_score(dataframe, dataset_name='VessMap', hue='model_type', x_data='num_samples', y_data='Dice',
                         line_styles: dict | None = None, markers: bool = True, log_x: bool = False,
                         marker_map: dict | None = None, annotate_zero_shot: bool = False,
                         share_zero_shot_color_with_scratch: bool = True,
                         share_zero_shot_color_with: str | None = 'finetuned',
                         title: str | None = None,
                         legend_title: str | None = None,
                         xlabel: str | None = None,
                         legend_loc: str | tuple | None = None,
                         save_path: str | None = None,
                         font_sizes: dict | None = None,
                         dpi: int = 300,
                         figsize: tuple | None = (12, 6),
                         y_limits: tuple | None = None,
                         x_limits: tuple | None = None,
                         percentage: bool = False,
                         main_zero_shift_frac: float | None = 0.0,
                         main_zero_shift_mode: str | None = 'zero-only',
                         inset_enabled: bool = False,
                         inset_x_limits: tuple | None = None,
                         inset_y_limits: tuple | None = None,
                         inset_show_axis_labels: bool = False,
                         inset_x_ticks: list | tuple | None = None,
                         inset_x_integer_ticks: bool = False,
                         inset_loc: str | int = 'upper right',
                         inset_size: tuple | None = None,
                         inset_bbox: tuple | None = None,
                         inset_borderpad: float | None = None,
                         inset_zero_shift_frac: float | None = 0.01):
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
        log_x: If True, use logarithmic scale on x axis (base 10).
        marker_map: Optional dict mapping hue values to matplotlib marker symbols.
            Se None, aplica regra: categorias contendo 'zero-shot' (case-insensitive)
            usam '*', demais 'o'.
        annotate_zero_shot: Se True, escreve o rótulo da categoria acima de cada ponto
            das séries zero-shot.
        share_zero_shot_color_with_scratch: [DEPRECATED] Mantido por compatibilidade; use
            'share_zero_shot_color_with'.
        share_zero_shot_color_with: Estratégia de cor herdada para séries zero-shot.
            Opções: 'finetuned' (padrão), 'scratch' ou None (não herda).
        title: Título customizável. Use '' para não exibir título. Se None, usa o padrão
            "{dataset_name} - {y_data} by {x_data} and {hue}".
    legend_title: Título da legenda. Se None, usa o nome da coluna hue. Se '' (string vazia), não exibe título.
    xlabel: Rótulo do eixo X. Se None, usa '# of labeled examples'.
    legend_loc: Posição da legenda (matplotlib). Aceita valores como 'best', 'upper right', 'lower left', etc.
        Especiais: 'outside-right' (fora à direita), 'outside-top', 'outside-bottom'.
        Também aceita tupla (x, y) em coordenadas do eixo via bbox_to_anchor.
        save_path: Caminho para salvar a figura (extensões suportadas: .png, .jpg, .jpeg, .pdf, .svg).
        font_sizes: Dict com tamanhos das fontes. Chaves suportadas: 'xlabel', 'ylabel', 'xaxis', 'yaxis', 'title', 'legend',
            'zero_shot_annotation' (tamanho da fonte da anotação), 'zero_shot_marker' (tamanho do símbolo zero-shot).
    dpi: DPI ao salvar (matplotlib).
    figsize: Tamanho da figura em polegadas (largura, altura). Default (12, 6).
    y_limits: Tupla opcional (ymin, ymax) para fixar os limites do eixo Y.
    x_limits: Tupla opcional (xmin, xmax) para fixar os limites do eixo X.
    main_zero_shift_frac: Fração do span do X principal para deslocar o eixo principal. 0.0 desativa.
    main_zero_shift_mode: 'zero-only' (padrão) desloca apenas pontos x==0 das séries zero-shot e remapeia o tick '0'.
        'global' desloca todas as marcações (dados/ticks/limites) pela mesma quantidade, preservando os rótulos.
    inset_enabled: Se True, adiciona um inset plot com os mesmos dados.
    inset_x_limits: Limites do eixo X do inset (xmin, xmax).
    inset_y_limits: Limites do eixo Y do inset (ymin, ymax).
    inset_show_axis_labels: Se True, exibe os rótulos dos eixos do inset usando os mesmos textos do gráfico principal.
    inset_x_ticks: Lista explícita de posições de ticks no X do inset. Se fornecida, substitui o automático.
    inset_x_integer_ticks: Se True, força apenas ticks inteiros dentro de inset_x_limits.
    inset_loc: Posição do inset (ex.: 'upper right', 'lower left', 1, 2, ...).
    inset_size: Frações (w, h) do tamanho do eixo principal; padrão (0.45, 0.35).
    inset_bbox: Alternativa para posicionamento absoluto no eixo: (x0, y0, w, h) em fração do eixo.
    inset_borderpad: Espaço entre inset e borda (quando usando inset_loc); default matplotlib se None.
    inset_zero_shift_frac: Fração do span de X do inset usada para deslocar pontos com x==0 (apenas zero-shot). Default 0.01.
    """
    if hue not in dataframe.columns:
        raise ValueError(f"Hue column '{hue}' not found. Columns: {list(dataframe.columns)}")
    if x_data not in dataframe.columns:
        raise ValueError(f"x_data column '{x_data}' not found. Columns: {list(dataframe.columns)}")
    if y_data not in dataframe.columns:
        raise ValueError(f"y_data column '{y_data}' not found. Columns: {list(dataframe.columns)}")

    df = dataframe.copy()
    # Tamanho da figura
    if figsize is None:
        figsize = (12, 6)
    plt.figure(figsize=figsize)

    unique_vals = [v for v in df[hue].dropna().unique()]
    if not unique_vals:
        print("[WARN] No hue categories to plot.")
        return
    if line_styles is None:
        base_patterns = ['-', '--', '-.', ':']
        line_styles = {val: base_patterns[i % len(base_patterns)] for i, val in enumerate(unique_vals)}

    palette = sns.color_palette(n_colors=len(unique_vals))
    color_map = {val: palette[i] for i, val in enumerate(unique_vals)}

    # Estratégia de cor para zero-shot
    strategy = share_zero_shot_color_with if share_zero_shot_color_with is not None else (
        'scratch' if share_zero_shot_color_with_scratch else None
    )
    if strategy in {'scratch', 'finetuned'}:
        # Funções auxiliares: considerar VSUNet/UNet além de ResNet
        def _is_scratch(name: str) -> bool:
            s = str(name).lower()
            # 'scratch' explícito OU UNet simples (evita confundir VSUNet)
            return ('scratch' in s) or (('unet' in s) and ('vsunet' not in s))
        def _is_finetuned(name: str) -> bool:
            s = str(name).lower()
            # expressões de fine-tuning OU identificadores VSUNet
            return (('fine-tuned' in s) or ('finetuned' in s) or ('fine tuned' in s)) or ('vsunet' in s)
        def _is_zero_shot(name: str) -> bool:
            return 'zero-shot' in str(name).lower()
        def _model_id(name: str):
            s = str(name)
            # Prioriza VSUNet antes de UNet para evitar capturar 'unet' dentro de 'vsunet'
            for pat in [r'(resnet\d+)', r'(vsunet\d+)', r'\b(unet\d+)\b']:
                m = re.search(pat, s, flags=re.IGNORECASE)
                if m:
                    return m.group(1).lower()
            return None
        # Mapeia modelo -> cor base conforme estratégia
        base_color_by_model = {}
        for val in unique_vals:
            if (strategy == 'scratch' and _is_scratch(val)) or (strategy == 'finetuned' and _is_finetuned(val)):
                mid = _model_id(val)
                if mid and mid not in base_color_by_model:
                    base_color_by_model[mid] = color_map[val]
        # Atribui cor aos zero-shot
        for val in unique_vals:
            if _is_zero_shot(val):
                mid = _model_id(val)
                if mid and mid in base_color_by_model:
                    color_map[val] = base_color_by_model[mid]

    # Override de cores: VSUNet18 azul, VSUNet50 laranja (inclui Zero-Shot)
    try:
        for val in unique_vals:
            s = str(val).lower()
            if 'vsunet18' in s:
                color_map[val] = 'tab:blue'
            elif 'vsunet50' in s:
                color_map[val] = 'tab:orange'
    except Exception:
        pass

    # Aggregate: mean and std per (hue, x)
    grouped = df.groupby([hue, x_data])[y_data].agg(['mean', 'std']).reset_index()

    # Calcular deslocamento no eixo principal (apenas linear)
    main_zero_offset = None
    if (main_zero_shift_frac or 0) > 0 and not log_x:
        try:
            if x_limits is not None and len(x_limits) == 2:
                xmin_m, xmax_m = float(x_limits[0]), float(x_limits[1])
            else:
                xmin_m, xmax_m = float(df[x_data].min()), float(df[x_data].max())
            span_m = max(xmax_m - xmin_m, 1e-9)
            main_zero_offset = 0.0 + float(main_zero_shift_frac) * span_m
        except Exception:
            main_zero_offset = None

    for val in unique_vals:
        sub = grouped[grouped[hue] == val]
        ls = line_styles.get(val, '-')
        color = color_map[val]
        is_zero = 'zero-shot' in str(val).lower()
        if markers:
            if marker_map is not None:
                marker_style = marker_map.get(val, 'o')
            else:
                marker_style = '*' if is_zero else 'o'
        else:
            marker_style = None
        # Ocultar zero-shot na legenda quando anotado
        label_val = val
        if annotate_zero_shot and is_zero:
            label_val = '_nolegend_'
        plt_kwargs = {}
        z_ms = (font_sizes or {}).get('zero_shot_marker')
        if is_zero and z_ms is not None:
            plt_kwargs['markersize'] = z_ms
        # Deslocamento conforme modo selecionado
        x_vals_main = sub[x_data].to_numpy().astype(float).copy()
        if (main_zero_offset is not None):
            if (main_zero_shift_mode or 'zero-only').lower().startswith('global'):
                # deslocar todo eixo
                x_vals_main = x_vals_main + main_zero_offset
            elif is_zero:
                # deslocar apenas zeros das séries zero-shot
                try:
                    x_vals_main[x_vals_main == 0.0] = main_zero_offset
                except Exception:
                    pass
        plt.plot(x_vals_main, sub['mean'], linestyle=ls, marker=marker_style, color=color, label=label_val, **plt_kwargs)
        # Error band (std)
        if 'std' in sub and not sub['std'].isna().all():
            plt.fill_between(x_vals_main, sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                             color=color, alpha=0.15, linewidth=0)
        # Anotações para zero-shot
        if annotate_zero_shot and is_zero:
            # Ajusta label com quebra de linha após 'zero-shot'
            display_val = re.sub(r'(zero-shot)\s*', lambda m: m.group(1) + '\n', str(val), flags=re.IGNORECASE)
            ann_fs = (font_sizes or {}).get('zero_shot_annotation', 9)
            for xi, yi in zip(x_vals_main, sub['mean']):
                plt.text(xi, yi + 0.005, display_val, ha='center', va='bottom', fontsize=ann_fs, color=color,
                         rotation=0, clip_on=True)

    # Rótulos e título
    default_title = f'{dataset_name} - {y_data} by {x_data} and {hue}'
    title_to_set = default_title if title is None else (title if title != '' else None)
    if title_to_set:
        plt.title(title_to_set, fontsize=(font_sizes or {}).get('title'))
    plt.xlabel(xlabel, fontsize=(font_sizes or {}).get('xlabel'))
    y_label_text = f'{y_data} (%)' if percentage else f'{y_data}'
    plt.ylabel(y_label_text, fontsize=(font_sizes or {}).get('ylabel'))

    # Título da legenda
    if legend_title is None:
        legend_title_arg = hue
    elif legend_title == '':
        legend_title_arg = None
    else:
        legend_title_arg = legend_title

    # Construção da legenda com controle de posição
    legend_kwargs = {'title': legend_title_arg}
    if legend_loc is None:
        leg = plt.legend(**legend_kwargs)
    else:
        loc_value = legend_loc
        bbox = None
        if isinstance(legend_loc, str):
            loc_key = legend_loc.strip().lower().replace('_', '-').replace('centre', 'center')
            if loc_key in {'outside-right', 'right-outside'}:
                # Fora à direita, vertical central
                loc_value = 'center left'
                bbox = (1.02, 0.5)
            elif loc_key in {'outside-top', 'top-outside'}:
                loc_value = 'lower center'
                bbox = (0.5, 1.02)
            elif loc_key in {'outside-bottom', 'bottom-outside'}:
                loc_value = 'upper center'
                bbox = (0.5, -0.02)
            else:
                # Usa diretamente valores válidos do matplotlib (best, upper right, etc.)
                bbox = None
        elif isinstance(legend_loc, (tuple, list)) and len(legend_loc) == 2:
            # Coordenadas customizadas com bbox_to_anchor
            loc_value = 'upper left'
            bbox = (float(legend_loc[0]), float(legend_loc[1]))
        if bbox is not None:
            leg = plt.legend(loc=loc_value, bbox_to_anchor=bbox, borderaxespad=0., **legend_kwargs)
        else:
            leg = plt.legend(loc=loc_value, **legend_kwargs)
    if leg and (font_sizes or {}).get('legend'):
        leg.set_title(leg.get_title().get_text())
        for txt in leg.get_texts():
            txt.set_fontsize((font_sizes or {}).get('legend'))
        leg.get_title().set_fontsize((font_sizes or {}).get('legend'))
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    if log_x:
        # Validate positive values
        if (df[x_data] <= 0).any():
            print("[WARN] Non-positive x values found; cannot apply log scale. Keeping linear scale.")
        else:
            plt.xscale('log', base=10)
            # Optional: minor grid for log
            plt.grid(True, which='both', axis='x', linestyle=':', alpha=0.4)
    else:
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
                ticks = list(range(0, max_x + 1, step))
                if (main_zero_offset is not None) and (main_zero_shift_mode or 'zero-only').lower().startswith('global'):
                    mapped = [t + main_zero_offset for t in ticks]
                    labels = [str(t) for t in ticks]
                    plt.xticks(mapped, labels)
                elif main_zero_offset is not None:
                    # mapear somente o zero
                    mapped = [(main_zero_offset if t == 0 else t) for t in ticks]
                    labels = [('0' if t == 0 else str(t)) for t in ticks]
                    plt.xticks(mapped, labels)
                else:
                    plt.xticks(ticks)
            except ValueError:
                pass  # silently skip tick customization if conversion fails
        else:
            print(f"[WARN] x-axis column not found or non-numeric among: {candidate_cols}. Skipping custom ticks.")

    # Formatação do eixo Y
    ax = plt.gca()
    if percentage:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # Limites opcionais do eixo X
    if x_limits is not None and isinstance(x_limits, (tuple, list)) and len(x_limits) == 2:
        try:
            xmin, xmax = float(x_limits[0]), float(x_limits[1])
            if (main_zero_offset is not None) and (main_zero_shift_mode or 'zero-only').lower().startswith('global'):
                ax.set_xlim((xmin + main_zero_offset, xmax + main_zero_offset))
            else:
                ax.set_xlim((xmin, xmax))
        except Exception:
            pass
    # Limites opcionais do eixo Y
    if y_limits is not None and isinstance(y_limits, (tuple, list)) and len(y_limits) == 2:
        try:
            ymin, ymax = float(y_limits[0]), float(y_limits[1])
            ax.set_ylim((ymin, ymax))
        except Exception:
            pass
    # Tamanho dos ticks
    if font_sizes:
        if font_sizes.get('xaxis'):
            ax.tick_params(axis='x', labelsize=font_sizes.get('xaxis'))
        if font_sizes.get('yaxis'):
            ax.tick_params(axis='y', labelsize=font_sizes.get('yaxis'))

    # Inset plot opcional
    if inset_enabled:
        try:
            if _mpl_inset_axes is None:
                raise ImportError('mpl_toolkits.axes_grid1.inset_locator.inset_axes não disponível.')
            if inset_size is None:
                inset_w, inset_h = 0.45, 0.35
            else:
                inset_w, inset_h = float(inset_size[0]), float(inset_size[1])
            # Criar eixo do inset: por bbox absoluto (se fornecido) ou por loc
            if inset_bbox is not None and len(inset_bbox) == 4:
                try:
                    x0, y0, ww, hh = [float(v) for v in inset_bbox]
                    axins = ax.inset_axes([x0, y0, ww, hh])
                except Exception:
                    # fallback para loc
                    if inset_borderpad is not None:
                        axins = _mpl_inset_axes(ax, width=f"{int(inset_w*100)}%", height=f"{int(inset_h*100)}%", loc=inset_loc, borderpad=float(inset_borderpad))
                    else:
                        axins = _mpl_inset_axes(ax, width=f"{int(inset_w*100)}%", height=f"{int(inset_h*100)}%", loc=inset_loc)
            else:
                if inset_borderpad is not None:
                    axins = _mpl_inset_axes(ax, width=f"{int(inset_w*100)}%", height=f"{int(inset_h*100)}%", loc=inset_loc, borderpad=float(inset_borderpad))
                else:
                    axins = _mpl_inset_axes(ax, width=f"{int(inset_w*100)}%", height=f"{int(inset_h*100)}%", loc=inset_loc)

            # Calcular deslocamento global para x==0 no inset (se houver série zero-shot)
            has_zero_shot = any('zero-shot' in str(v).lower() for v in unique_vals)
            zero_offset = None
            if has_zero_shot:
                try:
                    if inset_x_limits is not None and len(inset_x_limits) == 2:
                        xmin_i, xmax_i = float(inset_x_limits[0]), float(inset_x_limits[1])
                    else:
                        xmin_i, xmax_i = float(df[x_data].min()), float(df[x_data].max())
                    span_i = max(xmax_i - xmin_i, 1e-9)
                    frac_i = 0.01 if inset_zero_shift_frac is None else float(inset_zero_shift_frac)
                    zero_offset = 0.0 + frac_i * span_i
                except Exception:
                    zero_offset = None
            for val in unique_vals:
                sub = grouped[grouped[hue] == val]
                ls = line_styles.get(val, '-')
                color = color_map[val]
                is_zero = 'zero-shot' in str(val).lower()
                if markers:
                    if marker_map is not None:
                        marker_style = marker_map.get(val, 'o')
                    else:
                        marker_style = '*' if is_zero else 'o'
                else:
                    marker_style = None
                plt_kwargs = {}
                z_ms = (font_sizes or {}).get('zero_shot_marker')
                if is_zero and z_ms is not None:
                    plt_kwargs['markersize'] = z_ms
                # Deslocar pontos x==0 para direita no inset (apenas zero-shot)
                x_vals_ins = sub[x_data].to_numpy().astype(float).copy()
                if is_zero and (zero_offset is not None):
                    try:
                        x_vals_ins[x_vals_ins == 0.0] = zero_offset
                    except Exception:
                        pass
                axins.plot(x_vals_ins, sub['mean'], linestyle=ls, marker=marker_style, color=color, **plt_kwargs)
                if 'std' in sub and not sub['std'].isna().all():
                    axins.fill_between(x_vals_ins, sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                                       color=color, alpha=0.15, linewidth=0)
                # Anotações para zero-shot também no inset
                if annotate_zero_shot and is_zero:
                    display_val = re.sub(r'(zero-shot)\s*', lambda m: m.group(1) + '\n', str(val), flags=re.IGNORECASE)
                    ann_fs = (font_sizes or {}).get('zero_shot_annotation', 9)
                    for xi, yi in zip(x_vals_ins, sub['mean']):
                        axins.text(xi, yi + 0.005, display_val, ha='center', va='bottom', fontsize=ann_fs, color=color,
                                   rotation=0, clip_on=True)
            # Limites do inset
            if inset_x_limits is not None and len(inset_x_limits) == 2:
                try:
                    axins.set_xlim((float(inset_x_limits[0]), float(inset_x_limits[1])))
                except Exception:
                    pass
            if inset_y_limits is not None and len(inset_y_limits) == 2:
                try:
                    axins.set_ylim((float(inset_y_limits[0]), float(inset_y_limits[1])))
                except Exception:
                    pass
            # Ticks do inset no eixo X: controle explícito/inteiros e ajuste do tick "0" para zero_offset
            try:
                ticks_to_use = None
                labels_to_use = None
                if inset_x_ticks is not None:
                    ticks_to_use = [float(v) for v in inset_x_ticks]
                elif inset_x_integer_ticks and inset_x_limits is not None and len(inset_x_limits) == 2:
                    xmin_i, xmax_i = float(inset_x_limits[0]), float(inset_x_limits[1])
                    import math
                    start = math.ceil(min(xmin_i, xmax_i))
                    end = math.floor(max(xmin_i, xmax_i))
                    ticks_to_use = list(range(start, end + 1))
                    # Garante inclusão dos limites se forem inteiros exatos
                    if abs(xmin_i - round(xmin_i)) < 1e-9 and int(round(xmin_i)) not in ticks_to_use:
                        ticks_to_use = [int(round(xmin_i))] + ticks_to_use
                    if abs(xmax_i - round(xmax_i)) < 1e-9 and int(round(xmax_i)) not in ticks_to_use:
                        ticks_to_use = ticks_to_use + [int(round(xmax_i))]
                if ticks_to_use is not None:
                    mapped_ticks = []
                    labels_to_use = []
                    for t in ticks_to_use:
                        tv = float(t)
                        if zero_offset is not None and abs(tv - 0.0) < 1e-9:
                            mapped_ticks.append(zero_offset)
                            labels_to_use.append('0')
                        else:
                            mapped_ticks.append(tv)
                            labels_to_use.append(str(int(tv)) if abs(tv - round(tv)) < 1e-9 else f"{tv:.2f}")
                    axins.set_xticks(mapped_ticks)
                    axins.set_xticklabels(labels_to_use)
                elif zero_offset is not None:
                    # fallback: substitui apenas o tick 0 mantendo os demais
                    ticks = list(axins.get_xticks())
                    new_ticks = []
                    new_labels = []
                    for t in ticks:
                        if abs(t - 0.0) < 1e-9:
                            new_ticks.append(zero_offset)
                            new_labels.append('0')
                        else:
                            new_ticks.append(t)
                            if abs(t - round(t)) < 1e-9:
                                new_labels.append(str(int(round(t))))
                            else:
                                new_labels.append(f"{t:.2f}")
                    axins.set_xticks(new_ticks)
                    axins.set_xticklabels(new_labels)
            except Exception:
                pass
            if percentage:
                axins.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            else:
                axins.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # Tamanho dos ticks do inset (se fornecido)
            if font_sizes:
                if font_sizes.get('inset_xaxis'):
                    axins.tick_params(axis='x', labelsize=font_sizes.get('inset_xaxis'))
                if font_sizes.get('inset_yaxis'):
                    axins.tick_params(axis='y', labelsize=font_sizes.get('inset_yaxis'))
            # Rótulos dos eixos do inset (habilitar/desabilitar)
            if inset_show_axis_labels:
                axins.set_xlabel(xlabel, fontsize=(font_sizes or {}).get('xlabel'))
                axins.set_ylabel(f'{y_data}', fontsize=(font_sizes or {}).get('ylabel'))
            else:
                # Garante ocultação dos labels, caso algo tenha sido herdado
                axins.set_xlabel('')
                axins.set_ylabel('')
            axins.grid(True, linestyle=':')
        except Exception as e:
            print(f"[WARN] Falha ao criar inset plot: {e}")

    # Salvar figura, se solicitado
    if save_path:
        try:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        except Exception as e:
            print(f"[WARN] Falha ao salvar figura em {save_path}: {e}")
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
                            markers: bool = True, show: bool = True, log_x: bool = False,
                            symbol_map: dict | None = None, annotate_zero_shot: bool = False,
                            share_zero_shot_color_with_scratch: bool = True,
                            share_zero_shot_color_with: str | None = 'finetuned',
                            title: str | None = None,
                            legend_title: str | None = None,
                            legend_loc: str | tuple | None = None,
                            save_path: str | None = None,
                            font_sizes: dict | None = None,
                            figsize: tuple | None = (12, 6),
                            y_limits: tuple | None = None,
                            x_limits: tuple | None = None,
                            main_zero_shift_frac: float | None = 0.0,
                            main_zero_shift_mode: str | None = 'zero-only',
                            inset_domain: tuple | None = None,
                            inset_x_limits: tuple | None = None,
                            inset_y_limits: tuple | None = None,
                            inset_x_ticks: list | tuple | None = None,
                            inset_x_integer_ticks: bool = False,
                            inset_zero_shift_frac: float | None = 0.01):
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
        log_x: If True, use log scale on x axis (base 10).
        symbol_map: Optional mapping hue-> plotly marker symbol. Se None, regras:
            'zero-shot' (case-insensitive) => 'star', senão 'circle'.
    annotate_zero_shot: Se True, adiciona texto sobre os pontos das séries zero-shot.
        share_zero_shot_color_with_scratch: [DEPRECATED] Mantido por compatibilidade; use
            'share_zero_shot_color_with'.
        share_zero_shot_color_with: Estratégia de cor herdada para séries zero-shot.
            Opções: 'finetuned' (padrão), 'scratch' ou None (não herda).
    x_limits: Limites do eixo X principal (xmin, xmax). Para log_x=True, a range é em escala log10.
    main_zero_shift_frac: Fração do span do X principal usada para deslocamento no eixo principal em Plotly. 0.0 desativa.
    main_zero_shift_mode: 'zero-only' (padrão) desloca apenas pontos x==0 das séries zero-shot e remapeia o tick '0'.
        'global' desloca todas as marcações (dados/ticks/limites) pela mesma quantidade, preservando os rótulos.
    inset_domain: Tupla (x0, x1, y0, y1) no espaço [0,1] para posicionar o inset.
    inset_x_limits: Limites X do inset.
    inset_y_limits: Limites Y do inset.
    inset_x_ticks: Lista explícita de ticks para o X do inset.
    inset_x_integer_ticks: Se True, força ticks inteiros no X do inset.
    inset_zero_shift_frac: Fração do span de X do inset usada para deslocar pontos com x==0 no inset (apenas zero-shot). Default 0.01.
        legend_loc: Posição da legenda (plotly). Aceita presets como 'top-right', 'top-left', 'bottom-right', 'bottom-left',
            'top-center', 'bottom-center', 'center-right', 'center-left', 'center', 'outside-right'.
            Também aceita tupla (x, y) no intervalo [0,1] para posicionamento customizado.

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

    # Construir mapa de cores garantindo compartilhamento zero-shot conforme estratégia
    unique_vals = list(grouped[hue].unique())
    if px is None:
        base_palette = []
    else:
        # Usa paleta padrão do Plotly
        base_palette = px.colors.qualitative.Plotly
    color_map = {val: base_palette[i % len(base_palette)] if base_palette else None for i, val in enumerate(unique_vals)}

    strategy = share_zero_shot_color_with if share_zero_shot_color_with is not None else (
        'scratch' if share_zero_shot_color_with_scratch else None
    )
    if strategy in {'scratch', 'finetuned'}:
        def _is_scratch(name: str) -> bool:
            s = str(name).lower()
            return ('scratch' in s) or (('unet' in s) and ('vsunet' not in s))
        def _is_finetuned(name: str) -> bool:
            s = str(name).lower()
            return (('fine-tuned' in s) or ('finetuned' in s) or ('fine tuned' in s)) or ('vsunet' in s)
        def _is_zero_shot(name: str) -> bool:
            return 'zero-shot' in str(name).lower()
        def _model_id(name: str):
            s = str(name)
            for pat in [r'(resnet\d+)', r'(vsunet\d+)', r'\b(unet\d+)\b']:
                m = re.search(pat, s, flags=re.IGNORECASE)
                if m:
                    return m.group(1).lower()
            return None
        base_color_by_model = {}
        for val in unique_vals:
            if (strategy == 'scratch' and _is_scratch(val)) or (strategy == 'finetuned' and _is_finetuned(val)):
                mid = _model_id(val)
                if mid and mid not in base_color_by_model:
                    base_color_by_model[mid] = color_map[val]
        for val in unique_vals:
            if _is_zero_shot(val):
                mid = _model_id(val)
                if mid and mid in base_color_by_model:
                    color_map[val] = base_color_by_model[mid]

    # Override de cores: VSUNet18 azul, VSUNet50 laranja (inclui Zero-Shot)
    try:
        for val in unique_vals:
            s = str(val).lower()
            if 'vsunet18' in s:
                color_map[val] = 'tab:blue'
            elif 'vsunet50' in s:
                color_map[val] = 'tab:orange'
    except Exception:
        pass

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

    # Preparar coluna de símbolo se necessário
    symbol_col = None
    if markers:
        if symbol_map is None:
            # construir automaticamente
            auto_map = {}
            for v in grouped[hue].unique():
                auto_map[v] = 'star' if 'zero-shot' in str(v).lower() else 'circle'
            symbol_map = auto_map
        # Criar coluna auxiliar
        symbol_col = '_plot_symbol'
        grouped[symbol_col] = grouped[hue].map(symbol_map).fillna('circle')
        # Controlar tamanho do símbolo zero-shot, se fornecido
        zs_ms = (font_sizes or {}).get('zero_shot_marker') if font_sizes else None
        if zs_ms is not None:
            # Cria uma coluna de tamanho por ponto: aplica para zero-shot, None/NaN para outros
            size_col = '_plot_size'
            grouped[size_col] = grouped[hue].apply(lambda v: zs_ms if ('zero-shot' in str(v).lower()) else None)
        else:
            size_col = None

    # Preparar eixo X principal com deslocamento opcional (global ou apenas zero-shot)
    grouped['_x_main'] = grouped[x_data].astype(float)
    main_zero_offset = None
    if (main_zero_shift_frac or 0) > 0 and not log_x:
        try:
            if x_limits is not None and len(x_limits) == 2:
                xmin_m, xmax_m = float(x_limits[0]), float(x_limits[1])
            else:
                xmin_m, xmax_m = float(grouped[x_data].min()), float(grouped[x_data].max())
            span_m = max(xmax_m - xmin_m, 1e-9)
            main_zero_offset = 0.0 + float(main_zero_shift_frac) * span_m
            mode = (main_zero_shift_mode or 'zero-only').lower()
            if mode.startswith('global'):
                grouped['_x_main'] = grouped['_x_main'] + main_zero_offset
            else:
                # deslocar apenas as séries zero-shot com x==0
                mask_zero = grouped[hue].astype(str).str.lower().str.contains('zero-shot')
                grouped.loc[mask_zero & (grouped['_x_main'] == 0.0), '_x_main'] = main_zero_offset
        except Exception:
            main_zero_offset = None

    fig = px.line(
        grouped,
        x='_x_main',
        y=f'mean_{y_data}',
        color=hue,
        line_dash=hue,
        markers=markers,
        symbol=symbol_col if symbol_col else None,
    error_y=f'std_{y_data}',
        title=None,
        log_x=log_x,
        color_discrete_map=color_map
    )

    # Apply custom dash styles
    if line_styles:
        for trace in fig.data:
            val = trace.name
            if val in line_styles:
                trace.update(line={'dash': _normalize_dash(line_styles[val])})
            # Anotação zero-shot: adicionar texto (apenas uma vez por ponto)
            if annotate_zero_shot and 'zero-shot' in str(val).lower():
                # Se já não estiver em modo texto, acrescenta
                mode = trace.mode or 'lines'
                if 'text' not in mode:
                    trace.update(mode=mode + '+text')
                # Mostrar label apenas acima dos pontos (mesmo texto para cada ponto)
                n_pts = len(trace.x)
                display_val = re.sub(r'(zero-shot)\s*', lambda m: m.group(1) + '<br>', str(val), flags=re.IGNORECASE)
                ann_fs = (font_sizes or {}).get('zero_shot_annotation', 10)
                trace.update(text=[display_val]*n_pts, textposition='top center', textfont={'size':ann_fs})
                # Não mostrar na legenda
                trace.update(showlegend=False)

            # Ajustar tamanho do símbolo zero-shot para a série
            zs_ms = (font_sizes or {}).get('zero_shot_marker') if font_sizes else None
            if zs_ms is not None and getattr(trace, 'marker', None) is not None:
                if 'zero-shot' in str(val).lower():
                    trace.update(marker=dict(size=zs_ms))

    # Título
    default_title = f'{dataset_name} - {y_data} by {x_data} and {hue}'
    title_to_set = default_title if title is None else (title if title != '' else None)

    # Converter figsize (polegadas) para pixels (~96 px/inch) para Plotly
    width_px = height_px = None
    if figsize:
        try:
            width_px = int(figsize[0] * 96)
            height_px = int(figsize[1] * 96)
        except Exception:
            width_px = height_px = None

    # Título da legenda (None => usa hue; '' => remove)
    legend_title_arg = hue if legend_title is None else (legend_title if legend_title != '' else None)

    fig.update_layout(
        template='plotly_white',
        xaxis_title='Number of Samples',
        yaxis_title=f'{y_data}',
        legend_title=legend_title_arg,
        title=title_to_set,
        width=width_px,
        height=height_px
    )
    # Posição da legenda (plotly)
    if legend_loc is not None:
        x = y = None
        xanchor = 'center'
        yanchor = 'middle'
        if isinstance(legend_loc, (tuple, list)) and len(legend_loc) == 2:
            try:
                x, y = float(legend_loc[0]), float(legend_loc[1])
            except Exception:
                x = y = None
        elif isinstance(legend_loc, str):
            key = legend_loc.strip().lower().replace('_', '-').replace('centre', 'center')
            mapping = {
                'top-right':   (1.0, 1.0, 'right', 'top'),
                'upper right': (1.0, 1.0, 'right', 'top'),
                'top-left':    (0.0, 1.0, 'left',  'top'),
                'upper left':  (0.0, 1.0, 'left',  'top'),
                'bottom-right':(1.0, 0.0, 'right', 'bottom'),
                'lower right': (1.0, 0.0, 'right', 'bottom'),
                'bottom-left': (0.0, 0.0, 'left',  'bottom'),
                'lower left':  (0.0, 0.0, 'left',  'bottom'),
                'top-center':  (0.5, 1.0, 'center','top'),
                'bottom-center':(0.5,0.0,'center','bottom'),
                'center-right':(1.0, 0.5, 'right', 'middle'),
                'center-left': (0.0, 0.5, 'left',  'middle'),
                'center':      (0.5, 0.5, 'center','middle'),
                'outside-right':(1.02, 1.0, 'left', 'top')
            }
            if key in mapping:
                x, y, xanchor, yanchor = mapping[key]
        if x is not None and y is not None:
            fig.update_layout(legend=dict(x=x, y=y, xanchor=xanchor, yanchor=yanchor))
    # Formatar y com 2 casas decimais
    fig.update_yaxes(tickformat='.2f')
    # Limites opcionais do eixo Y
    if y_limits is not None and isinstance(y_limits, (tuple, list)) and len(y_limits) == 2:
        try:
            ymin, ymax = float(y_limits[0]), float(y_limits[1])
            fig.update_yaxes(range=[ymin, ymax])
        except Exception:
            pass
    # Limites opcionais do eixo X
    if x_limits is not None and isinstance(x_limits, (tuple, list)) and len(x_limits) == 2:
        try:
            xmin, xmax = float(x_limits[0]), float(x_limits[1])
            if log_x:
                import math
                if xmin > 0 and xmax > 0:
                    fig.update_xaxes(range=[math.log10(xmin), math.log10(xmax)])
                else:
                    print('[WARN] x_limits inválidos para escala log; ignorando limites no log.')
            else:
                if (main_zero_offset is not None) and (main_zero_shift_mode or 'zero-only').lower().startswith('global'):
                    fig.update_xaxes(range=[xmin + main_zero_offset, xmax + main_zero_offset])
                else:
                    fig.update_xaxes(range=[xmin, xmax])
        except Exception:
            pass

    # Tamanhos de fonte
    if font_sizes:
        if font_sizes.get('xlabel'):
            fig.update_layout(xaxis_title_font=dict(size=font_sizes.get('xlabel')))
        if font_sizes.get('ylabel'):
            fig.update_layout(yaxis_title_font=dict(size=font_sizes.get('ylabel')))
        if font_sizes.get('xaxis'):
            fig.update_xaxes(tickfont=dict(size=font_sizes.get('xaxis')))
        if font_sizes.get('yaxis'):
            fig.update_yaxes(tickfont=dict(size=font_sizes.get('yaxis')))
        if font_sizes.get('title'):
            fig.update_layout(title_font=dict(size=font_sizes.get('title')))
        if font_sizes.get('legend'):
            fig.update_layout(legend_font_size=font_sizes.get('legend'))

    # Adjust x ticks if numeric
    if not log_x and grouped['_x_main'].dtype.kind in 'iuf':
        max_x = grouped[x_data].max()
        step = 2 if max_x >= 8 else 1
        if main_zero_offset is None:
            fig.update_xaxes(dtick=step)
        elif (main_zero_shift_mode or 'zero-only').lower().startswith('global'):
            ticks = list(range(0, int(max_x) + 1, step))
            mapped = [float(t) + main_zero_offset for t in ticks]
            texts = [str(t) for t in ticks]
            fig.update_xaxes(tickmode='array', tickvals=mapped, ticktext=texts)
    if log_x:
        if (grouped[x_data] <= 0).any():
            print('[WARN] Non-positive x values present; log scale may be invalid.')

    # Ajustar ticks do eixo X principal para manter rótulo '0' no deslocamento
    if (main_zero_offset is not None) and not log_x and not ((main_zero_shift_mode or 'zero-only').lower().startswith('global')):
        try:
            # montar ticks inteiros padrão e mapear 0 -> offset
            if x_limits is not None and len(x_limits) == 2:
                xmin_m, xmax_m = float(x_limits[0]), float(x_limits[1])
            else:
                xmin_m, xmax_m = float(grouped[x_data].min()), float(grouped[x_data].max())
            import math
            start = max(0, math.ceil(xmin_m))
            end = math.floor(xmax_m)
            ticks = list(range(start, end + 1))
            mapped = [(main_zero_offset if t == 0 else float(t)) for t in ticks]
            texts = [('0' if t == 0 else str(t)) for t in ticks]
            fig.update_xaxes(tickmode='array', tickvals=mapped, ticktext=texts)
        except Exception:
            pass

    # Inset plot para plotly (eixos x2/y2) se solicitado
    if inset_domain is not None and go is not None:
        try:
            x0, x1, y0, y1 = [float(v) for v in inset_domain]
            fig.update_layout(
                xaxis2=dict(domain=[x0, x1], matches=None),
                yaxis2=dict(domain=[y0, y1], matches=None)
            )
            # Limites do inset
            if inset_y_limits is not None and len(inset_y_limits) == 2:
                fig.update_layout(yaxis2=dict(range=[float(inset_y_limits[0]), float(inset_y_limits[1])]))
            if inset_x_limits is not None and len(inset_x_limits) == 2:
                xmin, xmax = float(inset_x_limits[0]), float(inset_x_limits[1])
                if log_x:
                    import math
                    if xmin > 0 and xmax > 0:
                        fig.update_layout(xaxis2=dict(range=[math.log10(xmin), math.log10(xmax)]))
                else:
                    fig.update_layout(xaxis2=dict(range=[xmin, xmax]))
            # Calcular deslocamento global para x==0 no inset
            zero_offset = None
            try:
                has_zero_shot = any('zero-shot' in str(v).lower() for v in unique_vals)
                if has_zero_shot and (not log_x):
                    if inset_x_limits is not None and len(inset_x_limits) == 2:
                        xmin_i, xmax_i = float(inset_x_limits[0]), float(inset_x_limits[1])
                    else:
                        xmin_i, xmax_i = float(grouped[x_data].min()), float(grouped[x_data].max())
                    span_i = max(xmax_i - xmin_i, 1e-9)
                    frac_i = 0.01 if inset_zero_shift_frac is None else float(inset_zero_shift_frac)
                    zero_offset = 0.0 + frac_i * span_i
            except Exception:
                zero_offset = None
            # Adicionar traces no inset, sem legenda
            dash_map = {'-': 'solid', '--': 'dash', ':': 'dot', '-.': 'dashdot'}
            for val in unique_vals:
                sub = grouped[grouped[hue] == val]
                dash = line_styles.get(val) if line_styles else None
                dash = dash_map.get(dash, dash) if dash is not None else None
                is_zero = 'zero-shot' in str(val).lower()
                symbol = 'star' if is_zero else 'circle'
                if symbol_map is not None:
                    symbol = symbol_map.get(val, symbol)
                marker_size = None
                zs_ms = (font_sizes or {}).get('zero_shot_marker') if font_sizes else None
                if zs_ms is not None and is_zero:
                    marker_size = zs_ms
                # Deslocar x==0 levemente para direita no inset (somente zero-shot)
                xvals = sub[x_data].to_numpy().astype(float).copy()
                if is_zero and (zero_offset is not None):
                    try:
                        xvals[xvals == 0.0] = zero_offset
                    except Exception:
                        pass
                # Texto de anotação no inset quando solicitado
                mode_val = 'lines+markers' if markers else 'lines'
                text_vals = None
                if annotate_zero_shot and is_zero:
                    mode_val += '+text'
                    display_val = re.sub(r'(zero-shot)\s*', lambda m: m.group(1) + '<br>', str(val), flags=re.IGNORECASE)
                    text_vals = [display_val] * len(xvals)
                fig.add_trace(
                    go.Scatter(
                        x=xvals,
                        y=sub[f'mean_{y_data}'],
                        mode=mode_val,
                        name=str(val),
                        line=dict(color=color_map[val], dash=dash),
                        marker=dict(symbol=symbol, size=marker_size) if markers else None,
                        text=text_vals,
                        textposition='top center' if (annotate_zero_shot and is_zero) else None,
                        textfont={'size': (font_sizes or {}).get('zero_shot_annotation', 10)} if (annotate_zero_shot and is_zero) else None,
                        showlegend=False,
                        xaxis='x2',
                        yaxis='y2'
                    )
                )
            # Ajustar ticks do x do inset conforme controle solicitado e para que o rótulo "0" acompanhe o deslocamento
            try:
                if inset_x_limits is not None and len(inset_x_limits) == 2:
                    xmin_i, xmax_i = float(inset_x_limits[0]), float(inset_x_limits[1])
                    span = xmax_i - xmin_i
                    tickvals = None
                    if inset_x_ticks is not None:
                        tickvals = [float(v) for v in inset_x_ticks]
                    elif inset_x_integer_ticks:
                        import math
                        start = math.ceil(min(xmin_i, xmax_i))
                        end = math.floor(max(xmin_i, xmax_i))
                        tickvals = list(range(start, end + 1))
                        # garantir inclusão dos limites se forem inteiros exatos
                        if abs(xmin_i - round(xmin_i)) < 1e-9 and int(round(xmin_i)) not in tickvals:
                            tickvals = [int(round(xmin_i))] + tickvals
                        if abs(xmax_i - round(xmax_i)) < 1e-9 and int(round(xmax_i)) not in tickvals:
                            tickvals = tickvals + [int(round(xmax_i))]
                    if tickvals is None and zero_offset is not None and span > 0:
                        # fallback "agradável" com 4 ticks
                        step = span / 3.0
                        tickvals = [xmin_i, xmin_i + step, xmin_i + 2*step, xmax_i]
                    if tickvals is not None:
                        # mapear 0 -> zero_offset quando aplicável
                        mapped = []
                        texts = []
                        for v in tickvals:
                            vv = float(v)
                            if (zero_offset is not None) and (not log_x) and abs(vv - 0.0) <= max(1e-9, 1e-6*abs(span)):
                                mapped.append(zero_offset)
                                texts.append('0')
                            else:
                                mapped.append(vv)
                                # Inteiros sem casas; outros com até 2
                                texts.append(str(int(round(vv))) if abs(vv - round(vv)) < 1e-9 else f"{vv:.2f}".rstrip('0').rstrip('.'))
                        fig.update_layout(xaxis2=dict(tickmode='array', tickvals=mapped, ticktext=texts))
            except Exception:
                pass

            # Fontes dos ticks do inset
            try:
                if font_sizes:
                    if font_sizes.get('inset_xaxis'):
                        fig.update_layout(xaxis2=dict(tickfont=dict(size=font_sizes.get('inset_xaxis'))))
                    if font_sizes.get('inset_yaxis'):
                        fig.update_layout(yaxis2=dict(tickfont=dict(size=font_sizes.get('inset_yaxis'))))
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] Falha ao criar inset plot (plotly): {e}")

    # Salvar figura interativa como imagem, se solicitado
    if save_path:
        try:
            # Requer kaleido instalado: pip install -U kaleido
            fig.write_image(save_path)
        except Exception as e:
            print(f"[WARN] Falha ao salvar figura plotly em {save_path}: {e}")

    if show:
        fig.show()
    else:
        return fig

