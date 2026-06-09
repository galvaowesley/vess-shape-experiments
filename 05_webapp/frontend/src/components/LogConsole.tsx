import { useEffect, useRef, useState } from "react";
import { Copy, Eraser, Terminal } from "lucide-react";
import { trainingLogSocket } from "../lib/api";
import { Button } from "./ui";

/** Live training console fed by the /api/training/logs WebSocket. */
export function LogConsole() {
  const [lines, setLines] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  const autoscroll = useRef(true);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      ws = trainingLogSocket();
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        retry = setTimeout(connect, 1500); // auto-reconnect
      };
      ws.onmessage = (ev) => setLines((prev) => [...prev.slice(-4000), ev.data as string]);
    };
    connect();
    return () => {
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, []);

  useEffect(() => {
    if (autoscroll.current && boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight;
    }
  }, [lines]);

  const onScroll = () => {
    const el = boxRef.current;
    if (!el) return;
    autoscroll.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-[#0b1220]">
      <div className="flex items-center justify-between border-b border-[#25324b] px-3 py-2">
        <div className="flex items-center gap-2 text-xs font-medium text-[#93a3bd]">
          <Terminal className="h-4 w-4" />
          Console
          <span
            className={`ml-1 inline-block h-2 w-2 rounded-full ${connected ? "bg-green-400" : "bg-gray-500"}`}
            title={connected ? "connected" : "disconnected"}
          />
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={() => navigator.clipboard.writeText(lines.join("\n"))} title="Copy">
            <Copy className="h-3.5 w-3.5 text-[#93a3bd]" />
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setLines([])} title="Clear">
            <Eraser className="h-3.5 w-3.5 text-[#93a3bd]" />
          </Button>
        </div>
      </div>
      <div
        ref={boxRef}
        onScroll={onScroll}
        className="flex-1 overflow-auto px-3 py-2 font-mono text-[12px] leading-relaxed text-[#cbd5e1]"
      >
        {lines.length === 0 ? (
          <span className="text-[#5b6b86]">Waiting for output… launch a job with “Save &amp; Run”.</span>
        ) : (
          lines.map((l, i) => (
            <div key={i} className="whitespace-pre-wrap break-words">
              {l}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
