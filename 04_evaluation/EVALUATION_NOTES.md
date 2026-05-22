# Protocolo de Avaliação — Notas de Trabalho

> Arquivo de memória do diretório `04_evaluation/`. Registra as especificações das
> análises, decisões de design e o histórico de modificações para que o trabalho
> possa ser retomado em qualquer sessão. **Atualize a seção "Histórico de mudanças"
> sempre que algo for alterado.**

Última atualização: 2026-05-22

---

## 1. Objetivo

Avaliar resultados de **zero-shot** e **few-shot learning** de segmentação de vasos,
para um paper. Os experimentos foram expandidos de 2 para **4 datasets** e de poucas
variantes para **7 variantes de modelo** + zero-shot.

## 2. Datasets

| Dataset  | nº max de samples (ns) | Zero-shot real? | Scratch disponível? |
|----------|------------------------|-----------------|---------------------|
| VessMap  | 20                     | ✅ Sim (CSV)    | ✅ Sim              |
| DRIVE    | 16                     | ✅ Sim (CSV)    | ❌ **Ausente** (inferência ainda não rodou para `multi_train_scratch_drive_*`) |
| DCA1     | 20                     | ❌ **Mock**     | ✅ Sim              |
| OCTA2D   | 20                     | ❌ **Mock**     | ✅ Sim (1 run sem `metrics_stats.csv`) |

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
| `... (mock)`         | —                 | —                     | Placeholder `NaN`   | `zero_shot`|

Convenções: `VS`=VessShape, `IN-`=ImageNet, `-FT`=fine-tuned, `Zero-Shot`=sem FT (ns=0),
` (mock)`=placeholder com `is_mock=True`.

## 5. Decisões fechadas

- **Estrutura**: 1 notebook mestre + 4 por-dataset, todos reusando `utils_eval.py`.
- **DRIVE scratch ausente** → pular silenciosamente (séries não aparecem; sanity cell avisa).
- **Zero-shot DCA1/OCTA2D** → mockado (NaN) até a inferência real existir.
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

Arquivos legados (intocados): `utils.py`, `models_evalation.ipynb`, `inference_zero_shot.ipynb`.

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
- **Por imagem** (sufixo `_per_image`): uma linha por imagem de teste (de `inference_results/metrics.csv`). Zero-shot real entra (lido do raw); zero-shot mock **não** tem per-image, então `<ds>_zero_shot_results_per_image.csv` fica vazio para dca1/octa2d até a inferência real.

### Zero-shot para TODOS os datasets (auto-discovery)
- `load_dataset_results` / `load_dataset_per_image` procuram `zero_shot/zero_shot_inference_results_on_<ds>.csv` (fallback `zero_shot_inferences/`). Existindo o CSV real, ele é usado e o mock é ignorado — **sem editar código**.
- Para habilitar dca1/octa2d: dropar o CSV em `zero_shot/` (mesma convenção de nome) e re-rodar o notebook. Per-image zero-shot virá de `zero_shot_inferences/resnet{18,50}/inference_results_<ds>/metrics.csv`.
- `mock_zero_shot=True` só preenche enquanto o CSV real não existe (fallback gracioso, `is_mock=True`).

## 7. API do `utils_eval.py`

- `load_dataset_results(dataset, root='..', label_map=None, ignore_dirs=None, include_zero_shot=True, mock_zero_shot=False, zero_shot_dir=None)`
  → DataFrame tidy: `[run_name, num_samples, run, rep, wandb_group, model_class, Accuracy, IoU, Precision, Recall, Dice, AUC, model_type, stage, experiment, is_mock]`.
- `load_all_datasets(datasets=DATASETS, **kwargs)` → `dict[str, DataFrame]`.
- `build_summary_table(df, metrics=METRICS, group_cols=('model_type','num_samples'), sample_filter=None)` → mean/std (reusa `utils.get_experiments_grouped_stats`).
- `generate_mock_zero_shot(dataset, model_classes=('resnet18_unet','resnet50_unet'), placeholder=nan)`.
- `default_line_styles_7way()` → dict de line styles explícito (evita ciclo de 4 patterns).
- `report_coverage(df)` → contagem/min/max de `num_samples` por `(model_type, stage)`.

Constantes: `DATASETS`, `STAGES`, `DEFAULT_IGNORE_DIRS`, `METRICS`, `DEFAULT_LABEL_MAP`, `ZERO_SHOT_LABEL`.

## 8. Como rodar

```bash
conda activate mestrado_env
cd 04_evaluation
# Abrir os notebooks no JupyterLab com kernel mestrado_env, ou:
jupyter nbconvert --to notebook --execute eval_vessmap.ipynb --output eval_vessmap.executed.ipynb
# repetir para drive, dca1, octa2d, eval_master_cross_dataset
```

## 9. Checks de sanidade esperados (último smoke-test em 2026-05-22)

| Dataset  | shape   | model_types (nº de runs) |
|----------|---------|--------------------------|
| vessmap  | 1104×16 | 7 variantes (135–165 cada) + 2 zero-shot reais |
| drive    | 677×16  | 5 finetune (135 cada) + 2 zero-shot reais; **sem scratch** |
| dca1     | 999×16  | 7 variantes + 2 zero-shot mock |
| octa2d   | 1155×16 | 7 variantes + 2 zero-shot mock |

## 10. Pendências / TODO

- [ ] Rodar inferência **zero-shot real** para DCA1 e OCTA2D e substituir os mocks
      (gerar `zero_shot_inference_results_on_{dca1,octa2d}.csv` em `zero_shot_inferences/`).
- [ ] Rodar inferência **from-scratch** do DRIVE (`multi_train_scratch_drive_*`) para
      preencher `UNet18`/`UNet50` no DRIVE.
- [ ] (Opcional) Coordenar cores das novas variantes `IN-UNet*` e `LiteMedSAM-FT` no
      `plot_mean_dice_score` (hoje usam a paleta padrão; `share_zero_shot_color_with`
      só conhece resnet/unet/vsunet).
- [ ] (Opcional) Suporte a `ax=` no `plot_mean_dice_score` para painéis multi-dataset
      numa única figura (hoje gera 1 figura por dataset).

## 10.1. Riscos / pontos abertos

- `plot_mean_dice_score` cicla 4 line styles para 7+ hues — mitigado passando `default_line_styles_7way()` explícito.
- `share_zero_shot_color_with` só conhece resnet/unet/vsunet; `IN-UNet*` e `LiteMedSAM-FT` ficam com cor da paleta padrão (suficiente para o paper inicial).
- OCTA2D finetune: 1 run sem `metrics_stats.csv` (164/165) — skip silencioso é aceitável.
- DRIVE scratch ainda vazio — sanity cell sinaliza; re-rodar a avaliação quando a inferência from-scratch for feita.

## 11. Histórico de mudanças

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
