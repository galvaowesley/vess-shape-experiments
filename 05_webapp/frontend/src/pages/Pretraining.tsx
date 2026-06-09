import { Rocket, Clock } from "lucide-react";
import { Badge } from "../components/ui";
import { PageHeader } from "../components/common";

export default function Pretraining() {
  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Pretraining" subtitle="Stage 01 — pretraining on synthetic VessShape data." />
      <div className="flex flex-1 items-center justify-center p-6">
        <div className="max-w-md rounded-lg border border-border bg-surface p-8 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 text-primary">
            <Rocket className="h-7 w-7" />
          </div>
          <div className="mb-2 flex items-center justify-center gap-2">
            <h2 className="text-lg font-bold text-fg">Coming soon</h2>
            <Badge tone="accent">
              <span className="inline-flex items-center gap-1">
                <Clock className="h-3 w-3" /> planned
              </span>
            </Badge>
          </div>
          <p className="text-sm text-muted-fg">
            Stage 01 (pretraining on synthetic VessShape data) is still being validated, so it is not
            wired into the app yet. Training stages 02 (from scratch) and 03 (fine-tuning) are
            available now under <span className="font-medium text-fg">Training</span>.
          </p>
        </div>
      </div>
    </div>
  );
}
