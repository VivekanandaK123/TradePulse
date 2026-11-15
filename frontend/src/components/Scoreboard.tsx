import React from 'react';
import { OutcomesResponse, MetaResponse } from '../types/api';
import { formatPct } from '../lib/formatters';
import { Award, TrendingUp, AlertCircle } from 'lucide-react';

interface ScoreboardProps {
  outcomes?: OutcomesResponse;
  meta?: MetaResponse;
  signalsTodayCount?: number;
}

export const Scoreboard: React.FC<ScoreboardProps> = ({
  outcomes,
}) => {
  const totalOutcomes = outcomes?.outcomes.length ?? 0;
  const hits = outcomes?.outcomes.filter((o) => o.is_hit).length ?? 0;
  const hitRate = totalOutcomes > 0 ? (hits / totalOutcomes) * 100 : 0;

  const alwaysUpWinRate = 33.3; // Random baseline prior

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 flex flex-col gap-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="font-semibold text-xs tracking-wider text-slate-300 uppercase flex items-center gap-1.5">
          <Award className="h-4 w-4 text-blue-400" />
          Today's Scoreboard
        </span>
      </div>

      <div className="text-xs">
        {/* Realized 15-Min Win Rate */}
        <div className="bg-slate-950 p-2.5 rounded border border-slate-800/80 flex flex-col">
          <span className="text-slate-400">Realized Hit Rate</span>
          <div className="flex items-baseline gap-1 mt-0.5">
            <span className={`text-lg font-bold font-mono ${hitRate >= 50 ? 'text-emerald-400' : 'text-slate-200'}`}>
              {totalOutcomes > 0 ? formatPct(hitRate) : '--%'}
            </span>
            <span className="text-slate-500 text-[11px]">({hits}/{totalOutcomes})</span>
          </div>
        </div>
      </div>

      {/* Baseline comparison */}
      <div className="text-[11px] bg-slate-950/60 p-2 rounded border border-slate-800/60 text-slate-400 flex items-center justify-between">
        <span className="flex items-center gap-1">
          <TrendingUp className="h-3 w-3 text-slate-500" />
          Always-Up Baseline:
        </span>
        <span className="font-mono text-slate-300">{alwaysUpWinRate.toFixed(1)}% win rate</span>
      </div>

      {totalOutcomes < 10 && (
        <div className="text-[11px] text-amber-400/90 flex items-center gap-1 bg-amber-950/20 px-2 py-1 rounded border border-amber-900/40">
          <AlertCircle className="h-3 w-3 flex-shrink-0" />
          <span>Small sample (n={totalOutcomes}): Too few signals to judge accuracy.</span>
        </div>
      )}
    </div>
  );
};

