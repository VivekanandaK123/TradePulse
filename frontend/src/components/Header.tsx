import React from 'react';
import { MarketStatus } from '../types/api';
import { TimeZoneMode, formatCountdown } from '../lib/formatters';
import { 
  Activity, 
  Clock, 
  Database, 
  ShieldCheck, 
  Sun, 
  Moon, 
  HelpCircle, 
  Globe 
} from 'lucide-react';

interface HeaderProps {
  marketStatus?: MarketStatus;
  feedName?: string;
  modelVersion?: string;
  lastBarTimeNy?: string;
  timeZone: TimeZoneMode;
  onToggleTimeZone: () => void;
  isDarkTheme: boolean;
  onToggleTheme: () => void;
  onOpenGuide: () => void;
  isReplayActive?: boolean;
  onToggleReplayMode?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  marketStatus,
  feedName = 'Alpaca IEX, 5-min',
  modelVersion = 'TradePulse-v1.0-final',
  lastBarTimeNy,
  timeZone,
  onToggleTimeZone,
  isDarkTheme,
  onToggleTheme,
  onOpenGuide,
  isReplayActive = false,
  onToggleReplayMode,
}) => {
  const isOpen = marketStatus?.is_open ?? false;
  const countdownText = marketStatus?.countdown_seconds 
    ? formatCountdown(marketStatus.countdown_seconds) 
    : '--';

  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 text-sm">
      {/* Brand & Symbol */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white shadow-sm shadow-blue-500/20">
            TP
          </div>
          <span className="font-bold text-base tracking-tight text-white">TradePulse</span>
        </div>

        <div className="h-4 w-px bg-slate-700 mx-1" />

        <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/60">
          <span className="font-semibold text-slate-100">META</span>
          <span className="text-xs text-slate-400 font-mono">NASDAQ</span>
          <span className="text-[10px] bg-slate-700 text-slate-300 px-1 py-0.5 rounded font-mono">5m</span>
        </div>
      </div>

      {/* Status Chips */}
      <div className="flex items-center flex-wrap gap-2">
        {/* Market Status Chip */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
          isOpen 
            ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60' 
            : 'bg-amber-950/60 text-amber-300 border-amber-800/60'
        }`}>
          <span className={`h-2 w-2 rounded-full ${isOpen ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
          <span>{isOpen ? 'Market Open' : 'Market Closed'}</span>
          <span className="text-slate-400 text-[11px] font-mono">
            ({isOpen ? `closes in ${countdownText}` : `opens in ${countdownText}`})
          </span>
        </div>

        {/* Data Feed Chip */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs bg-slate-800/70 border border-slate-700/60 text-slate-300">
          <Database className="h-3.5 w-3.5 text-blue-400" />
          <span>{feedName}</span>
        </div>

        {/* Freshness Chip */}
        {lastBarTimeNy && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs bg-slate-800/70 border border-slate-700/60 text-slate-300">
            <Clock className="h-3.5 w-3.5 text-slate-400" />
            <span>Last closed: <strong className="font-mono text-slate-200">{lastBarTimeNy}</strong></span>
          </div>
        )}

        {/* Model Version Chip */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs bg-slate-800/70 border border-slate-700/60 text-slate-300 font-mono">
          <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
          <span>{modelVersion.split(' ')[0]}</span>
        </div>
      </div>

      {/* Control Actions */}
      <div className="flex items-center gap-2">
        {/* Live vs Replay Mode Segmented Switch */}
        <div className="flex items-center rounded-lg bg-slate-800/90 p-0.5 border border-slate-700/80 text-xs font-medium">
          <button
            onClick={() => isReplayActive && onToggleReplayMode && onToggleReplayMode()}
            className={`px-2.5 py-1 rounded-md transition flex items-center gap-1.5 ${
              !isReplayActive
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            title="Live mode: Alpaca IEX real-time feed"
          >
            <span className={`h-1.5 w-1.5 rounded-full ${!isReplayActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
            <span>Live</span>
          </button>
          <button
            onClick={() => !isReplayActive && onToggleReplayMode && onToggleReplayMode()}
            className={`px-2.5 py-1 rounded-md transition flex items-center gap-1.5 ${
              isReplayActive
                ? 'bg-amber-600 text-white shadow-sm font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            title="Replay mode: step through historical sessions"
          >
            <span className={`h-1.5 w-1.5 rounded-full ${isReplayActive ? 'bg-amber-300 animate-pulse' : 'bg-slate-500'}`} />
            <span>Replay</span>
          </button>
        </div>

        {/* TimeZone Toggle */}
        <button
          onClick={onToggleTimeZone}
          className="flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-mono bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition"
          title="Toggle display timezone (ET / Local)"
        >
          <Globe className="h-3.5 w-3.5 text-slate-400" />
          <span>{timeZone}</span>
        </button>

        {/* Theme Toggle */}
        <button
          onClick={onToggleTheme}
          className="p-1.5 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition"
          title="Toggle Theme"
        >
          {isDarkTheme ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-blue-400" />}
        </button>

        {/* User Guide Button */}
        <button
          onClick={onOpenGuide}
          className="flex items-center gap-1 px-2.5 py-1 rounded-md text-xs bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/30 transition font-medium"
        >
          <HelpCircle className="h-3.5 w-3.5" />
          <span>Guide</span>
        </button>
      </div>
    </header>
  );
};

