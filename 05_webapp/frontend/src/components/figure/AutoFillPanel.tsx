import { useState } from "react";
import { api } from "../../lib/api";
import type { AutoFillRequest, FigureOptions, GridFigureSpec, RunPolicy } from "../../lib/types";
import { Button, Checkbox, ChipMultiSelect, Field, NumberInput, Segmented, Select } from "../ui";
import { InlineNote } from "../common";

const nsLabel = (n: number) => (n === 0 ? "Zero-shot" : `${n}-shot`);

// Honest, plain-language description of what each policy actually means for a paper figure.
const POLICY_NOTES: Record<RunPolicy, string> = {
  median: "Median run — the representative result, the defensible default for a paper figure.",
  best: "Best run — cherry-picks the best of ~15 runs per cell. Say so explicitly if you use this.",
  worst: "Worst run — a failure-case view, rarely appropriate for a headline comparison.",
  fixed: "Fixed run/rep — the same train split & seed across every model, the fairest apples-to-apples comparison.",
};

export function AutoFillPanel({
  options,
  dataset,
  image,
  onResult,
}: {
  options: FigureOptions | null;
  dataset: string;
  image: string;
  onResult: (spec: GridFigureSpec) => void;
}) {
  const dsOpts = options?.by_dataset?.[dataset];
  const [modelTypes, setModelTypes] = useState<string[]>([]);
  const [numSamples, setNumSamples] = useState<number[]>([]);
  const [metric, setMetric] = useState("Dice");
  const [policy, setPolicy] = useState<RunPolicy>("median");
  const [fixedRun, setFixedRun] = useState<number | null>(null);
  const [fixedRep, setFixedRep] = useState<number | null>(null);
  const [orientation, setOrientation] = useState<AutoFillRequest["orientation"]>("models_as_cols");
  const [includeInput, setIncludeInput] = useState(true);
  const [includeGt, setIncludeGt] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!dataset || !image) {
      setError("Pick a dataset and an image first, in the browser on the right.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const spec = await api.autoFill({
        dataset,
        image,
        model_types: modelTypes,
        num_samples: numSamples,
        metric,
        policy,
        fixed_run: policy === "fixed" ? fixedRun : null,
        fixed_rep: policy === "fixed" ? fixedRep : null,
        orientation,
        include_input: includeInput,
        include_gt: includeGt,
      });
      onResult(spec);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-3">
      <InlineNote>
        Fills the grid for{" "}
        <span className="font-medium text-fg">{image || "(no image selected yet)"}</span> in{" "}
        <span className="font-medium text-fg">{dataset || "(no dataset)"}</span>.
      </InlineNote>

      <Field label="Models (empty = all)">
        <ChipMultiSelect options={dsOpts?.model_types ?? []} selected={modelTypes} onChange={setModelTypes} />
      </Field>

      <Field label="Num. samples (empty = all)">
        <ChipMultiSelect
          options={(dsOpts?.num_samples ?? []).map(nsLabel)}
          selected={numSamples.map(nsLabel)}
          onChange={(labels) => {
            const set = new Set(labels);
            setNumSamples((dsOpts?.num_samples ?? []).filter((n) => set.has(nsLabel(n))));
          }}
        />
      </Field>

      <Field label="Metric">
        <Select
          value={metric}
          onChange={setMetric}
          options={(options?.metrics ?? []).map((m) => ({ value: m, label: m }))}
        />
      </Field>

      <Field label="Run selection policy">
        <Segmented
          value={policy}
          onChange={setPolicy}
          options={[
            { value: "median", label: "Median" },
            { value: "best", label: "Best" },
            { value: "worst", label: "Worst" },
            { value: "fixed", label: "Fixed" },
          ]}
        />
      </Field>
      <InlineNote>{POLICY_NOTES[policy]}</InlineNote>

      {policy === "fixed" && (
        <div className="grid grid-cols-2 gap-2">
          <Field label="Run #">
            <NumberInput value={fixedRun} min={0} onChange={setFixedRun} />
          </Field>
          <Field label="Rep #">
            <NumberInput value={fixedRep} min={0} onChange={setFixedRep} />
          </Field>
        </div>
      )}

      <Field label="Orientation">
        <Segmented
          value={orientation}
          onChange={setOrientation}
          options={[
            { value: "models_as_cols", label: "Models → columns" },
            { value: "models_as_rows", label: "Models → rows" },
          ]}
        />
      </Field>

      <div className="flex items-center gap-4">
        <Checkbox checked={includeInput} onChange={setIncludeInput} label="Include input" />
        <Checkbox checked={includeGt} onChange={setIncludeGt} label="Include ground truth" />
      </div>

      <Button variant="primary" className="w-full" onClick={run} disabled={loading}>
        {loading ? "Filling…" : "Auto-fill grid"}
      </Button>
      {error && <InlineNote tone="danger">{error}</InlineNote>}
    </div>
  );
}
