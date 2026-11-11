import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-950 px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-500">
      <div className="flex items-center gap-2">
        <span className="font-semibold text-slate-400">TradePulse v1.0-final</span>
        <span>•</span>
        <span>Paper trading only. Experimental academic project (IIT Patna IMPACT Lab). Not financial advice.</span>
      </div>

      <div className="flex items-center gap-3">
        <span>
          Financial charts powered by{' '}
          <a
            href="https://www.tradingview.com/lightweight-charts/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-400 hover:text-blue-400 underline transition"
          >
            TradingView Lightweight Charts™
          </a>
        </span>
      </div>
    </footer>
  );
};

