import React from 'react';
import { PredictionSignal, MetaResponse } from '../types/api';
import { TimeZoneMode, formatPrice, formatPct, formatTimestamp } from '../lib/formatters';
import { 
  ArrowUpRight, 
  ArrowDownRight, 
  Minus, 
  Lock, 
  ShieldAlert, 
  HelpCircle, 
  Target, 
  CheckCircle2, 
  AlertCircle 
} from 'lucide-react';

interface PredictionCardProps {
  signal?: PredictionSignal;
  meta?: MetaResponse;
  timeZone: TimeZoneMode;
  onOpenGuideAnchor?: (anchor: string) => void;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  signal,
  meta,
  timeZone,
  onOpenGuideAnchor,
}) => {
  if (!signal) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 flex flex-col items-center justify-center text-center gap-3 min-h-[300px]">
        <div className="h-10 w-10 rounded-full bg-slate-800 flex items-center justify-center text-slate-500">
          <Target className="h-5 w-5" />
        </div>
        <div className="text-slate-300 font-medium text-sm">No Active Closed-Candle Prediction</div>
        <p className="text-xs text-slate-500 max-w-xs">
          Click <strong className="text-blue-400">Test Now</strong> (or press <strong>T</strong>) to evaluate the most recent closed 5-minute bar.
        </p>
      </div>
    );
  }

  const isUp = signal.direction === 'Up';
  const isDown = signal.direction === 'Down';
  const isNeutral = signal.direction === 'no clear signal';

  const thetaPct = meta?.theta_pct ?? 0.15;
  const horizonMin = meta?.horizon_minutes ?? 15;

  const tooltipExplanation = isUp
    ? `Calibrated probability that the next ${horizonMin} minutes closes at least +${thetaPct}% higher.`
    : isDown
    ? `Calibrated probability that the next ${horizonMin} minutes closes at least -${thetaPct}% lower.`
    : `Conviction is below threshold (${(signal.conviction).toFixed(4)} < ${signal.threshold.toFixed(4)}). No strong directional movement predicted.`;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col gap-4 shadow-lg shadow-black/20">
      {/* Top Header: Badge & Confidence */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
            Predicted Direction (Next {horizonMin}m)
          </div>

          {/* Direction Badge */}
          <div className="flex items-center gap-2">
            <div
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-bold text-base border shadow-sm ${
                isUp
                  ? 'bg-emerald-950/80 text-emerald-300 border-emerald-700/60'
                  : isDown
                  ? 'bg-rose-950/80 text-rose-300 border-rose-700/60'
                  : 'bg-slate-800 text-slate-300 border-slate-700'
              }`}
            >
              {isUp && <ArrowUpRight className="h-5 w-5 stroke-[2.5]" />}
              {isDown && <ArrowDownRight className="h-5 w-5 stroke-[2.5]" />}
              {isNeutral && <Minus className="h-5 w-5 stroke-[2.5]" />}
              <span>{signal.direction.toUpperCase()}</span>
            </div>

            {/* Confidence Number */}
            <div className="flex items-baseline gap-1" title={tooltipExplanation}>
              <span className={`text-2xl font-extrabold tracking-tight font-mono ${
                isUp ? 'text-emerald-400' : isDown ? 'text-rose-400' : 'text-slate-400'
              }`}>
                {signal.confidence_pct.toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500 font-medium">Confidence</span>
            </div>
          </div>
        </div>

        {/* Locked Status Badge (No Repainting Guarantee) */}
        <div 
          className="flex items-center gap-1 bg-slate-950 px-2 py-1 rounded text-[11px] border border-slate-800 text-slate-400 font-mono"
          title="Signal is permanently immutable in append-only log. It will never repaint or change."
        >
          <Lock className="h-3 w-3 text-emerald-500" />
          <span>Locked</span>
        </div>
      </div>

      {/* Conviction vs Threshold Gauge */}
      <div className="bg-slate-950 p-2.5 rounded-md border border-slate-800/80 flex flex-col gap-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400 flex items-center gap-1">
            Conviction: <strong className="text-slate-200 font-mono">{signal.conviction.toFixed(4)}</strong>
          </span>
          <span className="text-slate-500 text-[11px]">
            Threshold $\tau$: <strong className="text-slate-300 font-mono">{signal.threshold.toFixed(4)}</strong>
          </span>
        </div>

        {/* Visual Progress Gauge */}
        <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden relative">
          {/* Threshold marker tick */}
          <div 
            className="absolute top-0 bottom-0 w-0.5 bg-amber-400 z-10"
            style={{ left: `${Math.min(100, (signal.threshold / 0.5) * 100)}%` }}
            title={`Threshold Cutoff: ${signal.threshold.toFixed(4)}`}
          />
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              signal.is_gated
                ? (isUp ? 'bg-emerald-500' : 'bg-rose-500')
                : 'bg-slate-600'
            }`}
            style={{ width: `${Math.min(100, (signal.conviction / 0.5) * 100)}%` }}
          />
        </div>

        {/* Explainability note */}
        <div className="text-[11px] text-slate-400 italic">
          {signal.is_gated ? (
            <span className="text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Active signal: conviction exceeds calibrated threshold.
            </span>
          ) : (
            <span className="text-slate-400">
              Conviction {signal.conviction.toFixed(2)} is below {signal.threshold.toFixed(2)} needed for directional signal.
            </span>
          )}
        </div>
      </div>

      {/* Probability Distribution Breakdown */}
      <div className="flex flex-col gap-1.5">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center justify-between">
          <span>Calibrated Probabilities</span>
          <span className="text-slate-500 font-mono text-[10px]">Platt Sigmoid (OVR)</span>
        </div>

        <div className="grid grid-cols-3 gap-1.5 text-center text-xs">
          {/* Down */}
          <div className="bg-slate-950 p-2 rounded border border-slate-800/60 flex flex-col gap-0.5">
            <span className="text-rose-400 font-medium">Down</span>
            <span className="text-sm font-bold font-mono text-slate-100">
              {formatPct(signal.probabilities.down * 100)}
            </span>
            <div className="h-1 w-full bg-slate-800 rounded overflow-hidden mt-1">
              <div className="h-full bg-rose-500" style={{ width: `${signal.probabilities.down * 100}%` }} />
            </div>
          </div>

          {/* Neutral */}
          <div className="bg-slate-950 p-2 rounded border border-slate-800/60 flex flex-col gap-0.5">
            <span className="text-slate-400 font-medium">Neutral</span>
            <span className="text-sm font-bold font-mono text-slate-100">
              {formatPct(signal.probabilities.neutral * 100)}
            </span>
            <div className="h-1 w-full bg-slate-800 rounded overflow-hidden mt-1">
              <div className="h-full bg-slate-500" style={{ width: `${signal.probabilities.neutral * 100}%` }} />
            </div>
          </div>

          {/* Up */}
          <div className="bg-slate-950 p-2 rounded border border-slate-800/60 flex flex-col gap-0.5">
            <span className="text-emerald-400 font-medium">Up</span>
            <span className="text-sm font-bold font-mono text-slate-100">
              {formatPct(signal.probabilities.up * 100)}
            </span>
            <div className="h-1 w-full bg-slate-800 rounded overflow-hidden mt-1">
              <div className="h-full bg-emerald-500" style={{ width: `${signal.probabilities.up * 100}%` }} />
            </div>
          </div>
        </div>
      </div>

      {/* Candle Timing & Metadata */}
      <div className="text-xs border-t border-slate-800/80 pt-2 flex flex-col gap-1 text-slate-400">
        <div className="flex items-center justify-between">
          <span>Closed Candle:</span>
          <strong className="text-slate-200 font-mono">{formatTimestamp(signal.bar_time, timeZone)}</strong>
        </div>
        <div className="flex items-center justify-between">
          <span>Close Price:</span>
          <strong className="text-slate-200 font-mono">{formatPrice(signal.close_price)}</strong>
        </div>
        <div className="flex items-center justify-between">
          <span>Horizon Expiry:</span>
          <span className="text-slate-300 font-mono">{formatTimestamp(signal.horizon_end_utc, timeZone)}</span>
        </div>
      </div>

      {/* Advisory Reliability Chips */}
      {signal.reliability_flags && signal.reliability_flags.length > 0 && (
        <div className="border-t border-slate-800/80 pt-2 flex flex-col gap-1.5">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
            <ShieldAlert className="h-3.5 w-3.5 text-blue-400" />
            <span>Advisory Conditions</span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {signal.reliability_flags.map((flag) => (
              <button
                key={flag.id}
                onClick={() => onOpenGuideAnchor && onOpenGuideAnchor(flag.guide_anchor)}
                className={`text-[11px] px-2 py-0.5 rounded border transition flex items-center gap-1 ${
                  flag.level === 'WARNING'
                    ? 'bg-amber-950/60 text-amber-300 border-amber-800 hover:bg-amber-900/60'
                    : 'bg-slate-800/90 text-slate-300 border-slate-700 hover:bg-slate-700'
                }`}
                title={flag.message}
              >
                <span>{flag.name}</span>
                <HelpCircle className="h-3 w-3 opacity-60" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

