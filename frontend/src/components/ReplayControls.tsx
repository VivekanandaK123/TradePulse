import React from 'react';
import { ReplayState } from '../types/api';
import { Play, Pause, SkipForward, Square, AlertTriangle, Zap } from 'lucide-react';

interface ReplayControlsProps {
  replayState: ReplayState;
  availableSessions?: string[];
  onSelectSession: (date: string) => void;
  onStartReplay: (date: string, speed: number) => void;
  onStepReplay: () => void;
  onPauseReplay: () => void;
  onResumeReplay: () => void;
  onStopReplay: () => void;
}

export const ReplayControls: React.FC<ReplayControlsProps> = ({
  replayState,
  availableSessions = ['2026-11-09', '2026-11-10', '2026-11-11'],
  onSelectSession,
  onStartReplay,
  onStepReplay,
  onPauseReplay,
  onResumeReplay,
  onStopReplay,
}) => {
  const isPlaying = replayState.status === 'PLAYING';
  const hasStarted = replayState.current_bar_index > 0;
  const isActive = replayState.is_active;

  if (!isActive) return null;

  const progressPct = replayState.total_bars > 0
    ? Math.round((replayState.current_bar_index / replayState.total_bars) * 100)
    : 0;

  return (
    <div className="bg-amber-950/90 border-b border-amber-800/80 px-4 py-2 flex flex-wrap items-center justify-between gap-3 text-amber-200 shadow-md">
      {/* Left: Prominent Banner & Disclaimer */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 font-bold text-xs tracking-wider uppercase bg-amber-900/80 px-2.5 py-1 rounded border border-amber-700/80 text-amber-300 shadow-sm animate-pulse">
          <AlertTriangle className="h-4 w-4 text-amber-400" />
          <span>REPLAY MODE — NOT LIVE DATA</span>
        </div>
        <span className="hidden md:inline text-[11px] text-amber-300/80 italic">
          Demonstrates pipeline mechanics on historical bars, not out-of-sample edge.
        </span>
      </div>

      {/* Middle: Session Date Dropdown, Progress & Speed Multiplier */}
      <div className="flex items-center flex-wrap gap-2 text-xs font-mono">
        {/* Session Dropdown */}
        <div className="flex items-center gap-1.5 bg-amber-900/40 px-2 py-0.5 rounded border border-amber-800/60">
          <span className="text-amber-400">Session:</span>
          <select
            value={replayState.session_date}
            onChange={(e) => onSelectSession(e.target.value)}
            className="bg-transparent text-amber-100 font-bold focus:outline-none cursor-pointer"
          >
            {availableSessions.map((d) => (
              <option key={d} value={d} className="bg-slate-900 text-slate-100">
                {d}
              </option>
            ))}
          </select>
        </div>

        <span className="text-amber-500">•</span>

        {/* Bar Index Progress */}
        <div className="flex items-center gap-1.5">
          <span>
            Bar <strong>{replayState.current_bar_index}</strong> / {replayState.total_bars}
          </span>
          <div className="w-16 h-2 bg-amber-900/60 rounded-full overflow-hidden border border-amber-700/60">
            <div
              className="h-full bg-amber-400 transition-all duration-200"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <span className="text-[10px] text-amber-400">({progressPct}%)</span>
        </div>

        <span className="text-amber-500">•</span>

        <span className="text-[10px] bg-amber-900/50 border border-amber-700/60 px-1.5 py-0.5 rounded text-amber-300 font-bold">
          1x Speed
        </span>
      </div>

      {/* Right: Control Action Buttons */}
      <div className="flex items-center gap-1.5">
        {!hasStarted ? (
          <button
            onClick={() => onStartReplay(replayState.session_date, 1.0)}
            className="flex items-center gap-1.5 px-3.5 py-1 rounded bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-bold shadow-sm transition"
            title="Start demo testing on the selected unseen session"
          >
            <Zap className="h-3.5 w-3.5 fill-slate-950" />
            <span>Test Live Demo</span>
          </button>
        ) : isPlaying ? (
          <button
            onClick={onPauseReplay}
            className="flex items-center gap-1 px-3 py-1 rounded bg-amber-700 hover:bg-amber-600 text-white text-xs font-semibold shadow-sm transition"
            title="Pause replay playback"
          >
            <Pause className="h-3.5 w-3.5 fill-white" />
            <span>Pause</span>
          </button>
        ) : (
          <button
            onClick={onResumeReplay}
            className="flex items-center gap-1 px-3 py-1 rounded bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-semibold shadow-sm transition"
            title="Resume replay playback"
          >
            <Play className="h-3.5 w-3.5 fill-slate-950" />
            <span>Play</span>
          </button>
        )}

        {hasStarted && (
          <button
            onClick={onStepReplay}
            className="flex items-center gap-1 px-2.5 py-1 rounded bg-amber-800/80 hover:bg-amber-700 text-amber-100 text-xs font-medium border border-amber-700/60 transition"
            title="Step forward one 5-minute candle"
          >
            <SkipForward className="h-3.5 w-3.5" />
            <span>Step (+5m)</span>
          </button>
        )}

        <button
          onClick={onStopReplay}
          className="flex items-center gap-1 px-2.5 py-1 rounded bg-amber-950 hover:bg-amber-900 text-amber-300 border border-amber-800 text-xs font-medium transition"
          title="Exit replay and return to live stream"
        >
          <Square className="h-3.5 w-3.5" />
          <span>Exit Replay</span>
        </button>
      </div>
    </div>
  );
};
