/**
 * Reference table explaining the `model_type` nomenclature used across every
 * chart, so scientists can read the legends without digging into the notebooks.
 * Mirrors the legend in `04_evaluation/eval_*.ipynb`.
 */
const ROWS: [string, string, string, string, string][] = [
  ["UNet18", "U-Net (encoder ResNet-18)", "Aleatórios (Kaiming)", "Treino do zero", "scratch"],
  ["UNet50", "U-Net (encoder ResNet-50)", "Aleatórios (Kaiming)", "Treino do zero", "scratch"],
  ["VSUNet18", "U-Net (encoder ResNet-18)", "Pré-treinados em VessShape", "Fine-tuning no dataset alvo", "finetune"],
  ["VSUNet50", "U-Net (encoder ResNet-50)", "Pré-treinados em VessShape", "Fine-tuning no dataset alvo", "finetune"],
  ["IN-UNet18", "U-Net (encoder ResNet-18)", "Pré-treinados em ImageNet", "Fine-tuning no dataset alvo", "finetune"],
  ["IN-UNet50", "U-Net (encoder ResNet-50)", "Pré-treinados em ImageNet", "Fine-tuning no dataset alvo", "finetune"],
  ["LiteMedSAM-FT", "LiteMedSAM (ViT leve do MedSAM)", "Pré-treinados em MedSAM", "Fine-tuning no dataset alvo", "finetune"],
  ["Zero-Shot VSUNet18", "U-Net (encoder ResNet-18)", "Pré-treinados em VessShape", "Sem fine-tuning — inferência direta", "zero_shot"],
  ["Zero-Shot VSUNet50", "U-Net (encoder ResNet-50)", "Pré-treinados em VessShape", "Sem fine-tuning — inferência direta", "zero_shot"],
  ["Zero-Shot IN-UNet18", "U-Net (encoder ResNet-18)", "Pré-treinados em ImageNet", "Sem fine-tuning — inferência direta", "zero_shot"],
  ["Zero-Shot IN-UNet50", "U-Net (encoder ResNet-50)", "Pré-treinados em ImageNet", "Sem fine-tuning — inferência direta", "zero_shot"],
  ["Zero-Shot LiteMedSAM", "LiteMedSAM (ViT leve do MedSAM)", "Pré-treinados em MedSAM", "Sem fine-tuning — inferência direta", "zero_shot"],
  ["Zero-Shot LiteMedSAM+IN", "LiteMedSAM (ViT leve do MedSAM)", "MedSAM + norm. ImageNet na entrada", "Sem fine-tuning — inferência direta", "zero_shot"],
  ["… (mock)", "—", "—", "Placeholder com NaN", "zero_shot"],
];

const CONVENTIONS: [string, string][] = [
  ["VS", "pré-treino em VessShape (dataset sintético do projeto)."],
  ["IN-", "pré-treino em ImageNet."],
  ["-FT", "fine-tuning a partir de checkpoint pré-treinado."],
  ["+IN", "normalização ImageNet aplicada na entrada (não altera pesos, só pré-processa)."],
  ["Zero-Shot", "nenhum fine-tuning no dataset alvo (num_samples = 0)."],
  ["(mock)", "placeholder com NaN, fallback enquanto o CSV zero-shot real não existir."],
];

export function ModelLegend() {
  return (
    <div className="space-y-4 text-xs text-fg">
      <p className="text-muted-fg">
        Cada <code>model_type</code> codifica três informações: <b>arquitetura</b>, <b>origem dos
        pesos iniciais</b> e <b>regime de treino</b>.
      </p>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="bg-surface-2 text-[11px] uppercase tracking-wide text-muted-fg">
              <th className="px-3 py-2 font-semibold">Label</th>
              <th className="px-3 py-2 font-semibold">Arquitetura</th>
              <th className="px-3 py-2 font-semibold">Pesos iniciais</th>
              <th className="px-3 py-2 font-semibold">Regime</th>
              <th className="px-3 py-2 font-semibold">Stage</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r[0]} className="border-t border-border align-top">
                <td className="px-3 py-1.5">
                  <code className="rounded bg-muted px-1 py-0.5 text-[11px] text-fg">{r[0]}</code>
                </td>
                <td className="px-3 py-1.5 text-muted-fg">{r[1]}</td>
                <td className="px-3 py-1.5 text-muted-fg">{r[2]}</td>
                <td className="px-3 py-1.5 text-muted-fg">{r[3]}</td>
                <td className="px-3 py-1.5">
                  <code className="text-[11px] text-muted-fg">{r[4]}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <div className="mb-1 font-semibold text-fg">Convenções</div>
        <ul className="space-y-0.5">
          {CONVENTIONS.map(([k, v]) => (
            <li key={k} className="text-muted-fg">
              <code className="rounded bg-muted px-1 py-0.5 text-[11px] text-fg">{k}</code> = {v}
            </li>
          ))}
        </ul>
      </div>

      <div>
        <div className="mb-1 font-semibold text-fg">Eixo de comparação do paper</div>
        <ul className="list-disc space-y-0.5 pl-4 text-muted-fg">
          <li>
            <code>UNet*</code> (scratch) vs <code>IN-UNet*</code> (ImageNet) vs <code>VSUNet*</code>{" "}
            (VessShape) → ganho do pré-treino em VessShape sobre as alternativas.
          </li>
          <li>
            <code>Zero-Shot *</code> em <code>num_samples = 0</code> → transferência sem fine-tuning
            de cada origem de pesos.
          </li>
          <li>
            <code>LiteMedSAM-FT</code> → baseline de segmentação médica de larga escala (com
            fine-tuning).
          </li>
          <li>
            <code>Zero-Shot LiteMedSAM</code> vs <code>Zero-Shot LiteMedSAM+IN</code> → valida se a
            normalização ImageNet na entrada era necessária (Dice &gt; 0).
          </li>
        </ul>
      </div>
    </div>
  );
}
