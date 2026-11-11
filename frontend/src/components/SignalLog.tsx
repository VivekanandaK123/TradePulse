import React, { useState } from 'react';
import { SignalLogItem } from '../types/api';
import { TimeZoneMode, formatPrice, formatBps, formatTimestamp } from '../lib/formatters';
import { ListFilter, Download, ArrowUpRight, ArrowDownRight, Minus, Check, X } from 'lucide-react';

interface SignalLogProps {
  signals: SignalLogItem[];
  timeZone: TimeZoneMode;
}

export const SignalLog: React.FC<SignalLogProps> = ({ signals, timeZone }) => {
  const [filterDirection, setFilterDirection] = useState<string>('ALL');

  const filtered = signals.filter((s) => {
    if (filterDirection === 'ALL') return true;
    if (filterDirection === 'UP') return s.direction === 'Up';
    if (filterDirection === 'DOWN') return s.direction === 'Down';
    if (filterDirection === 'NEUTRAL') return s.direction === 'no clear signal';
    return true;
  });

  const handleExportCSV = () => {
    if (signals.length === 0) return;
    const headers = ['Time_Epoch', 'Time_NY', 'Direction', 'Signal_Code', 'Confidence_Pct', 'Conviction', 'Entry_Price', 'Checksum'];
    const rows = signals.map((s) => [
      s.bar_time,
      `"${s.bar_timestamp_ny}"`,
      s.direction,
      s.signal_code,
      s.confidence_pct,
      s.conviction,
      s.entry_price,
      s.checksum,
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `tradepulse_signals_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 flex flex-col gap-2.5">
      {/* Title & Actions */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="font-semibold text-xs tracking-wider text-slate-300 uppercase flex items-center gap-1.5">
          <ListFilter className="h-4 w-4 text-blue-400" />
          Signal Audit Log ({filtered.length})
        </span>

        <button
          onClick={handleExportCSV}
          disabled={signals.length === 0}
          className="flex items-center gap-1 px-2 py-0.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition"
          title="Export immutable signal log to CSV"
        >
          <Download className="h-3 w-3" />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Filter Chips */}
      <div className="flex items-center gap-1 text-[11px]">
        {['ALL', 'UP', 'DOWN', 'NEUTRAL'].map((f) => (
          <button
            key={f}
            onClick={() => setFilterDirection(f)}
            className={`px-2 py-0.5 rounded transition ${
              filterDirection === f
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:bg-slate-800'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Scrollable Table */}
      <div className="max-h-56 overflow-y-auto text-xs font-mono border border-slate-800 rounded">
        {filtered.length === 0 ? (
          <div className="p-4 text-center text-slate-500 text-xs">
            No signals match the selected filter.
          </div>
        ) : (
          <table className="w-full text-left border-collapse">
            <thead className="bg-slate-950 text-slate-400 text-[10px] uppercase sticky top-0 border-b border-slate-800">
              <tr>
                <th className="p-1.5">Time</th>
                <th className="p-1.5">Dir</th>
                <th className="p-1.5">Conf</th>
                <th className="p-1.5">Price</th>
                <th className="p-1.5">Outcome</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300 text-[11px]">
              {filtered.map((s) => {
                const isUp = s.direction === 'Up';
                const isDown = s.direction === 'Down';

                return (
                  <tr key={s.signal_id} className="hover:bg-slate-800/40">
                    <td className="p-1.5 text-slate-400 whitespace-nowrap">
                      {formatTimestamp(s.bar_time, timeZone).split(' ')[0]}
                    </td>
                    <td className="p-1.5 whitespace-nowrap">
                      <span className={`inline-flex items-center gap-0.5 font-bold ${
                        isUp ? 'text-emerald-400' : isDown ? 'text-rose-400' : 'text-slate-500'
                      }`}>
                        {isUp && <ArrowUpRight className="h-3 w-3" />}
                        {isDown && <ArrowDownRight className="h-3 w-3" />}
                        {s.direction === 'no clear signal' ? 'Neutral' : s.direction}
                      </span>
                    </td>
                    <td className="p-1.5 whitespace-nowrap text-slate-200">
                      {s.confidence_pct.toFixed(0)}%
                    </td>
                    <td className="p-1.5 whitespace-nowrap text-slate-300">
                      {formatPrice(s.entry_price)}
                    </td>
                    <td className="p-1.5 whitespace-nowrap">
                      {s.outcome ? (
                        <span className={`inline-flex items-center gap-0.5 px-1 py-0.2 rounded text-[10px] ${
                          s.outcome.is_hit ? 'bg-emerald-950 text-emerald-300' : 'bg-rose-950 text-rose-300'
                        }`}>
                          {s.outcome.is_hit ? <Check className="h-2.5 w-2.5" /> : <X className="h-2.5 w-2.5" />}
                          {formatBps(s.outcome.realized_return_bps)}
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[10px]">Pending</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

