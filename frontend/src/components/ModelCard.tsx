import React, { useState } from 'react';
import { MetaResponse } from '../types/api';
import { formatBps, formatPct } from '../lib/formatters';
import { FileCode, ChevronDown, ChevronUp, AlertOctagon, ShieldCheck } from 'lucide-react';

interface ModelCardProps {
  meta?: MetaResponse;
}

export const ModelCard: React.FC<ModelCardProps> = ({ meta }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!meta) return null;

  const testM = meta.headline_metrics.held_out_test;
  const valM = meta.headline_metrics.validation;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 flex flex-col gap-2.5">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between w-full text-left"
      >
        <span className="font-semibold text-xs tracking-wider text-slate-300 uppercase flex items-center gap-1.5">
          <FileCode className="h-4 w-4 text-blue-400" />
          Model Provenance & Card
        </span>
        <div className="flex items-center gap-1 text-slate-400 text-xs hover:text-slate-200">
          <span>{isExpanded ? 'Collapse' : 'Expand'}</span>
          {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </div>
      </button>

      {/* Edge Demonstrated Critical Notice (Always visible!) */}
      <div className={`p-2.5 rounded border text-xs flex items-start gap-2 ${
        meta.edge_demonstrated
          ? 'bg-emerald-950/40 border-emerald-800 text-emerald-300'
          : 'bg-rose-950/40 border-rose-800/80 text-rose-300'
      }`}>
        <AlertOctagon className="h-4 w-4 flex-shrink-0 mt-0.5 text-rose-400" />
        <div>
          <strong className="font-semibold block">
            {meta.edge_demonstrated ? 'Edge Demonstrated' : 'No Profitable Edge After Costs'}
          </strong>
          <span className="text-[11px] text-rose-200/90 leading-relaxed block mt-0.5">
            {meta.edge_statement}
          </span>
        </div>
      </div>

      {/* Target Labeling Formula (Always visible!) */}
      <div className="bg-slate-950/80 border border-slate-800/90 rounded p-2.5 text-[11px] flex flex-col gap-1.5">
        <div className="flex items-center justify-between text-slate-300 font-semibold">
          <span>Target Labeling Formula ({meta.label_name})</span>
          <span className="font-mono text-blue-400">θ = ±{meta.theta_bps} bps (±{meta.theta_pct}%)</span>
        </div>
        <div className="font-mono text-slate-200 bg-slate-900/90 px-2 py-1 rounded border border-slate-800">
          R(t, t+3) = (close[t+3] / close[t]) − 1 &nbsp;|&nbsp; 3-bar (15m) intraday forward return
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5 font-mono text-[10px]">
          <div className="bg-emerald-950/40 border border-emerald-800/50 text-emerald-300 px-2 py-1 rounded">
            Up (+1): R &gt; +0.0015
          </div>
          <div className="bg-slate-900 border border-slate-800 text-slate-300 px-2 py-1 rounded">
            Neutral (0): |R| ≤ 0.0015
          </div>
          <div className="bg-rose-950/40 border border-rose-800/50 text-rose-300 px-2 py-1 rounded">
            Down (−1): R &lt; −0.0015
          </div>
        </div>
        <span className="text-[10px] text-slate-500">
          Intraday only (grouped by session date; last {meta.horizon_bars} bars of each session dropped).
        </span>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="flex flex-col gap-3 pt-2 text-xs border-t border-slate-800 text-slate-300">
          {/* Core Configuration Grid */}
          <div className="grid grid-cols-2 gap-2 text-[11px] bg-slate-950 p-2.5 rounded border border-slate-800/80">
            <div>
              <span className="text-slate-500 block">Target Label (θ):</span>
              <strong className="text-slate-200 font-mono">±{meta.theta_bps} bps (15m)</strong>
            </div>
            <div>
              <span className="text-slate-500 block">Horizon:</span>
              <strong className="text-slate-200 font-mono">{meta.horizon_minutes}m ({meta.horizon_bars} bars)</strong>
            </div>
            <div>
              <span className="text-slate-500 block">Architecture:</span>
              <strong className="text-slate-200 font-mono">50% RF + 50% LR</strong>
            </div>
            <div>
              <span className="text-slate-500 block">Features:</span>
              <strong className="text-slate-200 font-mono">{meta.feature_count} Causal</strong>
            </div>
            <div>
              <span className="text-slate-500 block">Calibration:</span>
              <strong className="text-slate-200 font-mono">Platt Sigmoid (OVR)</strong>
            </div>
            <div>
              <span className="text-slate-500 block">Recency Weight:</span>
              <strong className="text-slate-200 font-mono">{meta.recency_weighting.half_life} (No decay)</strong>
            </div>
            <div>
              <span className="text-slate-500 block">Threshold (τ):</span>
              <strong className="text-slate-200 font-mono">{meta.signal_rule.conviction_threshold_tau.toFixed(4)}</strong>
            </div>
            <div>
              <span className="text-slate-500 block">Target Rate:</span>
              <strong className="text-slate-200 font-mono">{meta.signal_rule.expected_signals_per_day.toFixed(1)} sigs/day</strong>
            </div>
          </div>

          {/* Headline Held-Out Test Metrics Table */}
          <div>
            <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Held-Out Test Results ({testM.signals_emitted} signals across 47 sessions)
            </div>
            <table className="w-full text-left text-[11px] border border-slate-800 rounded overflow-hidden">
              <thead className="bg-slate-950 text-slate-400">
                <tr>
                  <th className="p-1.5">Metric</th>
                  <th className="p-1.5">Test Value</th>
                  <th className="p-1.5">95% Day-Block CI</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 font-mono text-slate-200">
                <tr>
                  <td className="p-1.5 text-slate-400">3-Class Accuracy</td>
                  <td className="p-1.5 font-bold">{formatPct(testM.accuracy)}</td>
                  <td className="p-1.5 text-slate-500">—</td>
                </tr>
                <tr>
                  <td className="p-1.5 text-slate-400">Balanced Accuracy</td>
                  <td className="p-1.5 font-bold text-emerald-400">{formatPct(testM.balanced_accuracy)}</td>
                  <td className="p-1.5 text-slate-400">[{testM.balanced_accuracy_95ci[0]}%, {testM.balanced_accuracy_95ci[1]}%]</td>
                </tr>
                <tr>
                  <td className="p-1.5 text-slate-400">Macro-F1 Score</td>
                  <td className="p-1.5 font-bold text-emerald-400">{testM.macro_f1.toFixed(4)}</td>
                  <td className="p-1.5 text-slate-400">[{testM.macro_f1_95ci[0]}, {testM.macro_f1_95ci[1]}]</td>
                </tr>
                <tr>
                  <td className="p-1.5 text-slate-400">Binary Sign Accuracy</td>
                  <td className="p-1.5 font-bold text-emerald-400">{formatPct(testM.binary_sign_accuracy)}</td>
                  <td className="p-1.5 text-slate-400">[{testM.binary_sign_accuracy_95ci[0]}%, {testM.binary_sign_accuracy_95ci[1]}%]</td>
                </tr>
                <tr>
                  <td className="p-1.5 text-slate-400">Gross Return</td>
                  <td className="p-1.5 font-bold text-emerald-400">{formatBps(testM.gross_return_bps)}</td>
                  <td className="p-1.5 text-slate-400">[{testM.gross_return_95ci[0]} bps, {testM.gross_return_95ci[1]} bps]</td>
                </tr>
                <tr>
                  <td className="p-1.5 text-slate-400">Net Return @ 8bp</td>
                  <td className="p-1.5 font-bold text-rose-400">{formatBps(testM.net_return_8bp_bps)}</td>
                  <td className="p-1.5 text-rose-300">[{testM.net_return_8bp_95ci[0]} bps, {testM.net_return_8bp_95ci[1]} bps]</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Training Date Scope */}
          <div className="text-[11px] text-slate-500 font-mono border-t border-slate-800 pt-1.5">
            Refit dataset: {meta.dataset_scope.training_date_range.start_date} to {meta.dataset_scope.training_date_range.end_date} ({meta.dataset_scope.training_date_range.total_sessions} sessions, {meta.dataset_scope.training_date_range.valid_training_rows.toLocaleString()} bars).
          </div>
        </div>
      )}
    </div>
  );
};

