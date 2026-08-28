# Protocolo de Avaliação — Notas de Trabalho

> Arquivo de memória do diretório `04_evaluation/`. Registra as especificações das
> análises, decisões de design e o histórico de modificações para que o trabalho
> possa ser retomado em qualquer sessão. **Atualize a seção "Histórico de mudanças"
> sempre que algo for alterado.**

Última atualização: 2026-05-29 (fase 4 — multi-métrica + paleta YAML)

---

## 1. Objetivo

Avaliar resultados de **zero-shot** e **few-shot learning** de segmentação de vasos,
para um paper. Os experimentos foram expandidos de 2 para **4 datasets** e de poucas
variantes para **7 variantes de modelo** + zero-shot.

## 2. Datasets

| Dataset  | nº max de samples (ns) | Zero-shot real? | Scratch disponível? |
|----------|------------------------|-----------------|---------------------|
| VessMap  | 20                     | ✅ Sim (6 variantes) | ✅ Sim          |
| DRIVE    | 16                     | ✅ Sim (6 variantes) | ❌ **Ausente** (inferência ainda não rodou para `multi_train_scratch_drive_*`) |
| DCA1     | 20                     | ✅ Sim (6 variantes) | ✅ Sim          |
| OCTA2D   | 20                     | ✅ Sim (6 variantes) | ✅ Sim (1 run sem `metrics_stats.csv`) |

> **Zero-shot (fase 3+4):** os 4 datasets têm zero-shot real gerado por `run_zero_shot_inference.ipynb`
> em 6 variantes — VSUNet18/50 (VessShape), IN-UNet18/50 (ImageNet), LiteMedSAM e LiteMedSAM+IN.
> O mock foi aposentado (fica só como fallback se um CSV faltar).

## 3. Layout dos dados (origem)

Dois estágios de experimentos, cada um com subpasta por dataset:

```
02_few_shot_training_from_scratch/<ds>/experiments/<exp>/<run_dir>/inference_results/metrics_stats.csv
03_few_shot_fine-tuning/<ds>/experiments/<exp>/<run_dir>/inference_results/metrics_stats.csv
```

- `<run_dir>` segue o padrão `<arch>_weights_id:<wid>_run:N_rep:N_ns:N`.
- `metrics_stats.csv` tem linhas `statistic ∈ {mean, std, min, 50%, max}` e colunas
  `Accuracy, IoU, Precision, Recall, Dice, AUC`. **Usamos a linha `mean`.**
- `metrics.csv` (por imagem) também existe, com schema `image,Accuracy,IoU,Precision,Recall,Dice,AUC`.
- O `config.yaml` na **raiz** do run-dir é **flat** (`model_class:`, `wandb_group:` como chaves
  diretas). O `wandb/latest-run/files/config.yaml` usa o formato aninhado `{value: ...}`.
  → O loader lê o flat primeiro, com fallback para o wandb.

## 4. Nomenclatura de modelos (`model_type`)

| Label                | Arquitetura       | Pesos iniciais        | Regime              | Stage      |
|----------------------|-------------------|-----------------------|---------------------|------------|
| `UNet18` / `UNet50`  | U-Net ResNet-18/50| Aleatórios            | Treino do zero      | `scratch`  |
| `VSUNet18` / `VSUNet50` | U-Net ResNet-18/50 | **VessShape**       | Fine-tuning         | `finetune` |
| `IN-UNet18` / `IN-UNet50` | U-Net ResNet-18/50 | **ImageNet**      | Fine-tuning         | `finetune` |
| `LiteMedSAM-FT`      | LiteMedSAM        | **MedSAM**            | Fine-tuning         | `finetune` |
| `Zero-Shot VSUNet18` / `...50` | U-Net ResNet-18/50 | **VessShape** | **Sem** fine-tuning | `zero_shot`|
| `Zero-Shot IN-UNet18` / `...50` | U-Net ResNet-18/50 | **ImageNet** | **Sem** fine-tuning | `zero_shot`|
| `Zero-Shot LiteMedSAM` | LiteMedSAM     | **MedSAM**            | **Sem** fine-tuning | `zero_shot`|
| `Zero-Shot LiteMedSAM+IN` | LiteMedSAM  | **MedSAM** + norm. IN entrada | **Sem** fine-tuning | `zero_shot`|
| `... (mock)`         | —                 | —                     | Placeholder `NaN`   | `zero_shot`|

Convenções: `VS`=VessShape, `IN-`=ImageNet, `-FT`=fine-tuned, `+IN`=normalização ImageNet na entrada
(`_PreprocessModel`, não altera pesos), `Zero-Shot`=sem FT (ns=0), `(mock)`=placeholder com `is_mock=True`.

## 5. Decisões fechadas

- **Estrutura**: 1 notebook mestre + 4 por-dataset, todos reusando `utils_eval.py`.
- **DRIVE scratch ausente** → pular silenciosamente (séries não aparecem; sanity cell avisa).
- **Zero-shot** → real para os 4 datasets (5 variantes), gerado por `run_zero_shot_inference.ipynb`.
  Mock (`mock_zero_shot=True`) só como fallback se um CSV faltar.
- **Dirs ignorados** (em `DEFAULT_IGNORE_DIRS`):
  - `multi_finetune_on_drive_resnet50_A`
  - `multi_finetune_on_drive_unet50_imagenet_weights`
  - `multi_finetune_on_drive_unet18_imagenet_weights`
- **Não modificar** `utils.py` nem `models_evalation.ipynb` (legados). Tudo novo vive em
  `utils_eval.py` + notebooks `eval_*.ipynb`.
- **Ambiente**: conda `mestrado_env`.

## 6. Inventário de arquivos (criados)

| Arquivo                          | Papel |
|----------------------------------|-------|
| `utils_eval.py`                  | Loader/labeler/mock + wrappers de resumo e estilos de linha |
| `eval_vessmap.ipynb`             | Avaliação VessMap (zero-shot real) |
| `eval_drive.ipynb`               | Avaliação DRIVE (zero-shot real, sem scratch) |
| `eval_dca1.ipynb`                | Avaliação DCA1 (zero-shot mock) |
| `eval_octa2d.ipynb`              | Avaliação OCTA2D (zero-shot mock) |
| `eval_master_cross_dataset.ipynb`| Comparações cross-dataset + tabelas/figuras agregadas |
| `run_zero_shot_inference.ipynb`  | **Gera** o zero-shot real (5 variantes × 4 datasets) e consolida os CSVs |
| `model_colors.yaml`              | Paleta padronizada `model_type → cor` para consistência cross-figure (paper) |

Arquivos legados (intocados): `utils.py`, `models_evalation.ipynb`, `inference_zero_shot.ipynb`
(este último substituído por `run_zero_shot_inference.ipynb`).

## 6.1. Estrutura de pastas (a partir de 2026-05-22, fase 2)

```
04_evaluation/
├── eval_*.ipynb, utils_eval.py, utils.py, EVALUATION_NOTES.md   (fonte — topo)
├── models_evalation.ipynb, inference_zero_shot.ipynb            (fonte legado — topo)
├── results/    → <ds>_results_summary.csv, <ds>_all_results.csv,
│                 <ds>_all_results_per_image.csv, cross_dataset_*.csv
├── figures/    → <ds>.svg, <ds>_resnet{18,50}.svg, cross_dataset_*.svg   (sem sufixo _v2)
├── zero_shot/  → zero_shot_inference_results_on_<ds>.csv  (interface importada pelos notebooks)
├── zero_shot_inferences/   (raw: imagens + metrics.csv por imagem; usado pelo legado e pelo per-image)
└── _legacy/    → artefatos antigos do models_evalation.ipynb (all_/best_/*_summary/*_resnet_results)
```

Convenções de saída (constantes em `utils_eval.py`): `RESULTS_DIR=results/`, `FIGURES_DIR=figures/`,
`ZERO_SHOT_DIR=zero_shot/`, `ZERO_SHOT_RAW_DIR=zero_shot_inferences/`. Os notebooks chamam
`ensure_output_dirs()` no setup. **O sufixo `v2` foi aposentado — os arquivos novos são o padrão.**

### Matrizes por dataset (geradas pelos próprios notebooks por-dataset)
Três matrizes intermediárias por granularidade, escritas por `utils_eval.save_result_matrices`:
- `results/<ds>_few_shot_results.csv` — só few-shot (`stage ∈ {scratch, finetune}`).
- `results/<ds>_zero_shot_results.csv` — só zero-shot (`stage == zero_shot`).
- `results/<ds>_all_results.csv` — few + zero-shot (matriz-base completa).

Granularidades:
- **Média por run** (sufixo vazio): uma linha por run = model_type × ns × run × rep.
- **Por imagem** (sufixo `_per_image`): uma linha por imagem de teste (de `inference_results/metrics.csv`). Zero-shot real entra (lido do raw por variante).

### Zero-shot — 6 variantes para TODOS os datasets (fase 3+4)

Geradas por `run_zero_shot_inference.ipynb`. Layout raw (1 pasta por variante):
```
zero_shot_inferences/<variant_key>/inference_results_<ds>/{metrics.csv, metrics_stats.csv}
```
`variant_key ∈ {resnet18_vessshape, resnet50_vessshape, resnet18_imagenet, resnet50_imagenet, litemedsam, litemedsam_normalized}`
(dirs legados `resnet18`/`resnet50` = VessShape, ainda aceitos como fallback). Registry em
`utils_eval.ZERO_SHOT_VARIANTS` mapeia cada `variant_key → {model_class, weights, model_type[, imagenet_normalize]}`.

- **Consolidação:** `utils_eval.build_zero_shot_csv(ds)` lê a linha `mean` de cada variante e escreve
  `zero_shot/zero_shot_inference_results_on_<ds>.csv` com colunas
  `[run_name, num_samples=0, wandb_group, model_class, weights, *METRICS, model_type]` (1 linha/variante).
- **Auto-discovery:** `load_dataset_results` lê esse CSV e **confia no `model_type`** dele (model_class
  sozinho já não distingue VessShape × ImageNet). `load_dataset_per_image` itera as pastas por variante.
  Existindo o CSV/raw real, o mock é ignorado — sem editar código.
- **resize por dataset (ResNet):** vessmap 256 · drive 288 · dca1 288 · octa2d 384; **LiteMedSAM = 256** sempre.
  Canais: ResNet usa o default do dataset (gray/all); LiteMedSAM usa `rgb` (3 canais, exigência do encoder).
- **Normalização ImageNet:** opcional (`IMAGENET_NORMALIZE` no notebook → `--imagenet_normalize` no `test.py`), default off.
- `mock_zero_shot=True` só preenche se um CSV real faltar (fallback gracioso, `is_mock=True`).

## 7. API do `utils_eval.py`

- `load_dataset_results(dataset, root='..', label_map=None, ignore_dirs=None, include_zero_shot=True, mock_zero_shot=False, zero_shot_dir=None)`
  → DataFrame tidy: `[run_name, num_samples, run, rep, wandb_group, model_class, Accuracy, IoU, Precision, Recall, Dice, AUC, model_type, stage, experiment, is_mock]`.
- `load_all_datasets(datasets=DATASETS, **kwargs)` → `dict[str, DataFrame]`.
- `build_summary_table(df, metrics=METRICS, group_cols=('model_type','num_samples'), sample_filter=None)` → mean/std (reusa `utils.get_experiments_grouped_stats`).
- `build_zero_shot_csv(dataset, raw_dir=ZERO_SHOT_RAW_DIR, out_dir=ZERO_SHOT_DIR, variants=ZERO_SHOT_VARIANTS, write=True)`
  → consolida as 6 variantes (linha `mean` de cada `metrics_stats.csv`) em `zero_shot/zero_shot_inference_results_on_<ds>.csv`.
- `save_result_matrices(df, dataset, out_dir=RESULTS_DIR, suffix='')` → escreve few/zero/all matrices.
- `generate_mock_zero_shot(dataset, model_classes=('resnet18_unet','resnet50_unet'), placeholder=nan)`.
- `default_line_styles_7way()` → dict de line styles explícito (cobre VSUNet/IN-UNet/LiteMedSAM zero-shot).
- `load_model_palette(yaml_path=None, warn_missing=None)` → `dict[model_type → cor]` carregado
  de `model_colors.yaml` (default `<module_dir>/model_colors.yaml`). Retorna `None` se o arquivo
  não existir. Passado como `color_map_override` em `plot_mean_dice_score`, `plot_crossdataset_bar`
  e (futuras) plot helpers para cores consistentes cross-figure. `warn_missing` emite warning
  listando labels esperados ausentes.
- `report_coverage(df)` → contagem/min/max de `num_samples` por `(model_type, stage)`.

### Cross-dataset bar charts & Wilcoxon helpers (`utils_eval.py`, fase 4)

Movidos para `utils_eval.py` para que o `eval_master_cross_dataset.ipynb` contenha
apenas orquestração (loops sobre métricas + I/O). Todos os `plot_*` fazem **imports
preguiçosos** de matplotlib/scipy para não inflar o cold load de `utils_eval`.

- `load_per_image_dict(datasets=DATASETS_ORDER, results_dir=RESULTS_DIR)` → `dict[ds → DataFrame]`,
  lê `results/<ds>_all_results_per_image.csv`.
- `compute_bar_sig_pvalues(pivot, per_image, n_samples=1, metric='Dice', fine_tuned_to_zeroshot=None, min_common_images=5)`
  → `dict[ft_model → dict[dataset → p_value]]`. Wilcoxon one-sided (fine-tuned@N > zero-shot par)
  por modelo × dataset. Param `metric` permite reuso em qualquer das 6 métricas.
- `compute_pairwise_sig_matrix(model_a, model_b, per_image, metric='Dice', datasets_order=None, fine_tuned_to_zeroshot=None, min_common_images=5)`
  → `DataFrame(rows=datasets, cols=N)`. Wilcoxon H: Metric(model_a) > Metric(model_b);
  N=0 via `FINE_TUNED_TO_ZEROSHOT`, N≥1 via few-shot.
- `plot_crossdataset_bar(pivot, title='', ylabel='', xlabel='Dataset', save_path=None,
  y_limits=None, ymin=None, ymax=None, sig_pvalues=None, alpha=ALPHA, color_map_override=None,
  palette=None, percentage=False, font_sizes=None, figsize=(12,5.5), rotation=0,
  legend_loc='upper left', legend_bbox=(1.01,1), legend_title='Model', datasets_order=None, dpi=150)`
  → barras agrupadas com **datasets no eixo x** e **cores = modelo**. Aceita `color_map_override`
  (YAML), `percentage` (×100 + `%` no eixo) e `font_sizes` dict
  (`{title, xlabel, ylabel, xaxis, yaxis, legend, sig_marker}`). `y_limits` tem prioridade
  sobre `ymin`/`ymax`.
- `plot_sig_heatmap(p_matrix, title='', save_path=None, alpha=ALPHA, xlabel='# Samples',
  ylabel='Dataset', font_sizes=None, figsize=None, dpi=150)` → heatmap binário: cinza+X
  vermelho = NaN, azul claro = não-sig, azul escuro = sig. `font_sizes` dict consolidado
  (`{title, xlabel, ylabel, xaxis, yaxis}`). Sem anotações numéricas.

Constantes exportadas: `ALPHA=0.05`, `DATASETS_ORDER=['dca1','drive','octa2d','vessmap']`,
`FINE_TUNED_TO_ZEROSHOT` (mapa FT→ZS, 5 entradas), `VSUPAIR_COMPARISONS` (4 pares VSUNet vs.
alternativo para o loop §8).

Constantes/objetos locais ao master notebook (cell de imports/loader):
`PALETTE = load_model_palette()`, `PRETTY` (dataset → label legível para títulos),
`per_image = load_per_image_dict()` (lazy load das matrizes por imagem).
Todas as constantes globais (`ALPHA`, `DATASETS_ORDER`, `FINE_TUNED_TO_ZEROSHOT`,
`VSUPAIR_COMPARISONS`) e as funções de plot moram em `utils_eval.py` — ver seção
"Cross-dataset bar charts & Wilcoxon helpers" mais abaixo.

### Paleta padronizada — `model_colors.yaml`

Mapa autoritativo `model_type → cor matplotlib` (hex ou named). É carregado em **todos** os
notebooks `eval_*.ipynb` e no master via `PALETTE = load_model_palette()` e passado como
`color_map_override` em cada chamada de plot. Garante que VSUNet18, IN-UNet50, etc. têm a
**mesma cor em todas as figuras** do paper.

Editar este arquivo recolore todas as figuras na próxima execução — não há cor hardcoded
no código (o antigo override VSUNet18=azul / VSUNet50=laranja em `utils.plot_mean_dice_score`
foi removido na fase 4 e migrou para o YAML).

Constantes: `DATASETS`, `STAGES`, `DEFAULT_IGNORE_DIRS`, `METRICS`, `DEFAULT_LABEL_MAP`, `ZERO_SHOT_LABEL`,
`ZERO_SHOT_VARIANTS` (registry variant_key → {model_class, weights, model_type}).

## 8. Como rodar

```bash
conda activate mestrado_env
cd 04_evaluation
# Abrir os notebooks no JupyterLab com kernel mestrado_env, ou:
jupyter nbconvert --to notebook --execute eval_vessmap.ipynb --output eval_vessmap.executed.ipynb
# repetir para drive, dca1, octa2d, eval_master_cross_dataset
```

## 9. Checks de sanidade esperados (zero-shot real — fase 3+4, 2026-05-22)

Cada `zero_shot/zero_shot_inference_results_on_<ds>.csv` deve ter **6 linhas** (VSUNet18/50,
IN-UNet18/50, LiteMedSAM, LiteMedSAM+IN) e `load_dataset_results(<ds>)` deve mostrar `is_mock=False`.
Dice zero-shot observado (referência — fase 3):

| Dataset  | VSUNet18 | VSUNet50 | IN-UNet18 | IN-UNet50 | LiteMedSAM | LiteMedSAM+IN |
|----------|----------|----------|-----------|-----------|------------|---------------|
| vessmap  | 0.766    | 0.638    | 0.103     | 0.366     | ~0.00      | (a rodar)     |
| drive    | 0.553    | 0.175    | 0.172     | 0.226     | ~0.00      | (a rodar)     |
| dca1     | 0.536    | 0.480    | 0.055     | 0.092     | ~0.00      | (a rodar)     |
| octa2d   | 0.272    | 0.470    | 0.138     | 0.151     | ~0.00      | (a rodar)     |

> **LiteMedSAM zero-shot ≈ 0 (H1)**: sem normalização ImageNet na entrada, o encoder TinyViT
> recebe inputs fora de distribuição → embeddings degradados → Dice ≈ 0. `LiteMedSAM+IN` valida
> esta hipótese: se Dice > 0, H1 é confirmada. O prompt (caixa da imagem inteira) é o mesmo em
> ambas as variantes e não é a causa primária (o `prompt_encoder` é congelado em treino e teste).

## 10. Pendências / TODO

- [x] **2026-05-29** — Multi-métrica + paleta YAML + controle estético uniforme:
      `model_colors.yaml` criado; `load_model_palette` em `utils_eval.py`; `plot_mean_dice_score`
      ganhou `color_map_override` (override hardcoded VSUNet removido); `plot_crossdataset_bar`
      e `plot_sig_heatmap` reescritas com assinaturas completas (font_sizes, percentage, dpi,
      legend_loc, color_map_override, metric param); todos os notebooks (`eval_*`, master) com
      loop sobre as 6 métricas em §4/§5 (eval_*) e §4/§6/§7/§8 (master). SVGs nomeados por métrica.
- [ ] Rodar `litemedsam_normalized` (fase 4) nos 4 datasets e re-executar `eval_*.ipynb`;
      confirmar que LiteMedSAM+IN Dice > 0 (valida H1) ou documentar como limitação.
- [ ] Rodar inferência **from-scratch** do DRIVE (`multi_train_scratch_drive_*`) para
      preencher `UNet18`/`UNet50` no DRIVE.
- [ ] (Opcional) Coordenar cores das novas variantes `IN-UNet*` e `LiteMedSAM-FT` no
      `plot_mean_dice_score` (hoje usam a paleta padrão; `share_zero_shot_color_with`
      só conhece resnet/unet/vsunet).
- [ ] (Opcional) Suporte a `ax=` no `plot_mean_dice_score` para painéis multi-dataset
      numa única figura (hoje gera 1 figura por dataset).
- [x] **2026-05-29** — Gráficos de barra cross-dataset redesenhados: datasets no eixo x,
      cores = modelo, paleta `muted`, significância Wilcoxon sobre barras few-shot, parâmetros
      `ymin`/`ymax`. Wilcoxon §8 reformulado: hipótese VSUNet > IN-UNet / LiteMedSAM-FT,
      inclui N=0, sem anotações numéricas, X vermelho para NaN, figuras separadas por par.

## 10.1. Riscos / pontos abertos

- `plot_mean_dice_score` cicla 4 line styles para 7+ hues — mitigado passando `default_line_styles_7way()` explícito.
- `share_zero_shot_color_with` só conhece resnet/unet/vsunet; `IN-UNet*` e `LiteMedSAM-FT` ficam com cor da paleta padrão (suficiente para o paper inicial).
- OCTA2D finetune: 1 run sem `metrics_stats.csv` (164/165) — skip silencioso é aceitável.
- DRIVE scratch ainda vazio — sanity cell sinaliza; re-rodar a avaliação quando a inferência from-scratch for feita.
- **LiteMedSAM (torchtrainer)** usa encoders singletons a nível de módulo: `test()` in-process move-os
  p/ GPU e a 2a chamada quebra com device-mismatch. Por isso `run_zero_shot_inference.ipynb` roda
  cada inferência como **subprocesso** de `src/test.py` (igual ao orquestrador few-shot).
- **Caminho do projeto tem espaços** (Google Drive). `dict_to_argv` dá `.split()` nos valores, então
  a saída é escrita por um **symlink sem espaços** (`/tmp/vss_zero_shot_raw`) que aponta p/ `zero_shot_inferences/`.
- LiteMedSAM zero-shot Dice ≈ 0 (H1: encoder TinyViT sem normalização ImageNet recebe inputs
  fora de dist.). `litemedsam_normalized` (+IN) valida a hipótese — se Dice > 0, normalização era necessária.
  H2 (full-image bbox) descartada como causa primária: `prompt_encoder` é congelado, logo o prompt
  embedding é idêntico em zero-shot e few-shot; a diferença vem do encoder/decoder adaptados.

## 11. Histórico de mudanças

- **2026-05-29 (fase 4 — multi-métrica, paleta YAML e modularização)**
  Refatoração ampla do pipeline de plots para o paper:
  - **Multi-métrica**: todos os notebooks (`eval_*`, master) agora plotam as 6 métricas
    (`Accuracy, IoU, Precision, Recall, Dice, AUC`) via loop sobre `METRICS`. SVGs nomeados
    `<ds>_<metric>.svg`, `<ds>_<family>_<metric>.svg` e `sig_wilcoxon_<a>_vs_<b>_<metric>.svg`.
  - **Paleta YAML**: criado `model_colors.yaml`; `load_model_palette()` em `utils_eval.py`
    carrega `dict[model_type → cor]` passado como `color_map_override` em todos os plots.
    Override hardcoded VSUNet18=azul / VSUNet50=laranja em `utils.plot_mean_dice_score`
    removido — cores agora 100% no YAML.
  - **Controle estético uniforme**: `plot_crossdataset_bar` e `plot_sig_heatmap` reescritos
    com assinaturas completas: `font_sizes` dict, `percentage`, `dpi`, `legend_loc`,
    `legend_bbox`, `legend_title`, `xlabel`, `y_limits` etc. — paridade com
    `plot_mean_dice_score`.
  - **Modularização**: `ALPHA`, `DATASETS_ORDER`, `FINE_TUNED_TO_ZEROSHOT`, `VSUPAIR_COMPARISONS`,
    `_sig_label`, `compute_bar_sig_pvalues`, `compute_pairwise_sig_matrix`, `plot_crossdataset_bar`,
    `plot_sig_heatmap` e `load_per_image_dict` movidos para `utils_eval.py`. Cell 14 do master
    passou de ~200 linhas (constantes + 3 funções) para 3 linhas (só `per_image = load_per_image_dict()`).
    Imports de matplotlib/scipy são lazy.

- **2026-05-29 (fase 4 — redesign de visualizações no master notebook)**
  `eval_master_cross_dataset.ipynb` — §6/7 gráficos de barra: eixo x = datasets, cores = modelos,
  paleta `sns.color_palette("muted")` (publicável), marcadores `*`/`**`/`***` via Wilcoxon
  one-sided (few-shot@N=1 vs zero-shot par) acima de cada barra, parâmetros `ymin`/`ymax`.
  §8 Wilcoxon redesenhado: hipótese "Dice(VSUNet) > Dice(IN-UNet / LiteMedSAM-FT)", inclui
  N=0 (zero-shot via `FINE_TUNED_TO_ZEROSHOT`) e N≥1 (few-shot); heatmap sem anotações
  numéricas — apenas cor (cinza/azul-claro/azul-escuro) + X vermelho para NaN; 4 figuras
  SVG separadas (1 por par: VSUNet18>IN-UNet18, VSUNet18>LiteMedSAM-FT, VSUNet50>IN-UNet50,
  VSUNet50>LiteMedSAM-FT); parâmetros de fonte individuais em `plot_sig_heatmap`; eixo
  padronizado para "# Samples".

- **2026-05-22 (fase 4 — investigação LiteMedSAM zero-shot)** — Adicionada 6ª variante
  `litemedsam_normalized` (`Zero-Shot LiteMedSAM+IN`): mesmos pesos `lite_medsam.pth` com
  normalização ImageNet na entrada (`_PreprocessModel`). Hipóteses: H1 (normalização, dominante)
  e H2 (full-image bbox, descartada — `prompt_encoder` é congelado em ambos os regimes).
  `utils_eval.py`: `litemedsam_normalized` em `ZERO_SHOT_VARIANTS`, estilo `--`, flag
  `imagenet_normalize` no registry. `run_zero_shot_inference.ipynb`: Patches A+B no helper,
  nova seção §4, tabela atualizada (6 variantes). `eval_*.ipynb`: legenda + família LiteMedSAM
  (FT + ZS + ZS+IN). `EVALUATION_NOTES.md`: atualizado (seções 2, 4, 6.1, 7, 9, 10, 10.1, 11).
- **2026-05-22 (fase 3 — zero-shot real p/ todos os datasets)** — Novo `run_zero_shot_inference.ipynb`
  gera zero-shot real em **5 variantes × 4 datasets** (VSUNet18/50, IN-UNet18/50, LiteMedSAM), via
  subprocesso de `src/test.py`. `src/test.py` ganhou `--channels` (default preserva comportamento) e
  `--imagenet_normalize` (opcional, default off; wrapper `_PreprocessModel` portado de `test___.py`).
  `utils_eval.py`: `ZERO_SHOT_VARIANTS` + `build_zero_shot_csv`, `_load_real_zero_shot` confia no
  `model_type` do CSV, `_load_real_zero_shot_per_image` itera as pastas por variante, line styles
  para os novos labels. Notebooks `eval_*` com legenda atualizada e `MOCK_ZS=False` em dca1/octa2d.
  20/20 inferências OK; LiteMedSAM zero-shot ≈ 0 (esperado). Legado `inference_zero_shot.ipynb` intocado.
- **2026-05-22 (fase 2 — reorganização)** — Subdiretórios `results/`, `figures/`, `zero_shot/`,
  `_legacy/`. Sufixo `v2` aposentado (arquivos novos são o padrão). Artefatos legados do
  `models_evalation.ipynb` movidos para `_legacy/` (fontes legados mantidos no topo).
  `utils_eval.py`: constantes de saída + `ensure_output_dirs()`, auto-discovery de zero-shot
  para qualquer dataset (`zero_shot/` com fallback `zero_shot_inferences/`), `load_dataset_per_image`
  e `save_result_matrices`. Notebooks por-dataset geram matrizes few-shot / zero-shot / all
  (média por run e por imagem). Master roteado para `results/`+`figures/`.
- **2026-05-22** — Criação do pipeline: `utils_eval.py`, 4 notebooks por-dataset, notebook
  mestre. Smoke-test OK sob `mestrado_env`. Legenda de nomenclaturas adicionada ao topo
  de todos os notebooks. Criado este arquivo de notas.
