import React from 'react';
import { X, ShieldAlert, AlertTriangle, BookOpen, Clock, Lock, CheckCircle2 } from 'lucide-react';

interface GuideDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  activeAnchor?: string;
}

export const GuideDrawer: React.FC<GuideDrawerProps> = ({
  isOpen,
  onClose,
  activeAnchor,
}) => {
  React.useEffect(() => {
    if (isOpen && activeAnchor) {
      const timer = setTimeout(() => {
        const el = document.getElementById(activeAnchor);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
          el.classList.add('ring-2', 'ring-blue-500');
          setTimeout(() => el.classList.remove('ring-2', 'ring-blue-500'), 2500);
        }
      }, 150);
      return () => clearTimeout(timer);
    }
  }, [isOpen, activeAnchor]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-md bg-slate-900 border-l border-slate-800 h-full overflow-y-auto p-5 flex flex-col gap-4 text-slate-200 shadow-2xl">
        {/* Top Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-blue-400" />
            <span className="font-bold text-base text-white">Trader User Guide</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md hover:bg-slate-800 text-slate-400 hover:text-slate-100 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Section 1: When Not to Trust This */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-amber-400">
            <AlertTriangle className="h-4 w-4" />
            <span>When NOT to Trust the Indicator</span>
          </div>

          {/* Stale Data */}
          <div id="stale-data" className="bg-slate-950 p-3 rounded border border-slate-800 text-xs flex flex-col gap-1">
            <strong className="text-slate-100 flex items-center gap-1.5">
              <Clock className="h-3.5 w-3.5 text-amber-400" />
              1. Stale Data & Feed Latency
            </strong>
            <p className="text-slate-400 leading-relaxed">
              If the latest closed bar is older than 5 minutes during market hours, the data feed is lagging. Never trade on stale signals. The banner at the top of the live chart will alert you when data freshness exceeds expected boundaries.
            </p>
          </div>

          {/* Session Boundaries */}
          <div id="session-boundaries" className="bg-slate-950 p-3 rounded border border-slate-800 text-xs flex flex-col gap-1">
            <strong className="text-slate-100 flex items-center gap-1.5">
              <ShieldAlert className="h-3.5 w-3.5 text-blue-400" />
              2. Opening & Closing Regime Volatility
            </strong>
            <p className="text-slate-400 leading-relaxed">
              The first 15 minutes (09:30–09:45 ET) and last 15 minutes (15:45–16:00 ET) experience heavy institutional auction volume and widening spreads. The model was trained with the final 3 bars dropped per session. Exercise caution on signals near session boundaries.
            </p>
          </div>

          {/* News & Earnings */}
          <div id="news-earnings" className="bg-slate-950 p-3 rounded border border-slate-800 text-xs flex flex-col gap-1">
            <strong className="text-slate-100 flex items-center gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 text-rose-400" />
              3. Scheduled News & Earnings Announcements
            </strong>
            <p className="text-slate-400 leading-relaxed">
              The model relies exclusively on 29 causal technical price-volume features. It does <strong>not</strong> read headlines, quarterly earnings, FOMC rate decisions, or regulatory filings. Do not execute signals around scheduled high-impact news catalysts.
            </p>
          </div>

          {/* Thin Liquidity */}
          <div id="thin-liquidity" className="bg-slate-950 p-3 rounded border border-slate-800 text-xs flex flex-col gap-1">
            <strong className="text-slate-100 flex items-center gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 text-slate-400" />
              4. Half-Day & Low-Volume Sessions
            </strong>
            <p className="text-slate-400 leading-relaxed">
              Early-close sessions (such as the day after Thanksgiving or Christmas Eve) feature distorted liquidity. The indicator's volume and RVOL metrics can misestimate typical turnover dynamics.
            </p>
          </div>
        </div>

        {/* Section 2: Understanding Signals & Probabilities */}
        <div className="flex flex-col gap-2.5 border-t border-slate-800 pt-3">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Signal Interpretation Principles
          </div>

          {/* Calibrated Confidence */}
          <div className="text-xs text-slate-400 space-y-2 leading-relaxed">
            <p>
              <strong className="text-slate-200">What Confidence Means:</strong> The percentage shown (e.g. "Up, 68%") is the Platt-calibrated probability that META will move at least ±15 bps (0.15%) in the designated direction over the next 15 minutes (3 bars). It is an empirical likelihood, <strong>never a guaranteed price target</strong>.
            </p>

            <p>
              <strong className="text-slate-200">What "No Clear Signal" Means:</strong> When conviction falls below the threshold (τ = 0.2635), the indicator gates the output to Neutral. This prevents overtrading during flat consolidation periods (~47% of all bars).
            </p>

            <p>
              <strong className="text-slate-200">Zero Repainting Guarantee:</strong> Signals are evaluated strictly upon closed 5-minute candles and appended immediately to an immutable log file. Once a signal is printed, it is mathematically locked and cannot mutate.
            </p>
          </div>
        </div>

        {/* Section 3: Educational Disclaimer */}
        <div className="bg-slate-950/80 p-3 rounded border border-slate-800 text-xs text-slate-400 flex flex-col gap-1.5 mt-auto">
          <span className="font-semibold text-slate-200">Academic Project Disclaimer</span>
          <p className="text-[11px] leading-relaxed">
            TradePulse was engineered under IIT Patna IMPACT Lab guidelines for End-to-End Machine Learning. Institutional friction (8 bps round-trip) results in net negative expectancy. This indicator is an experimental analytical tool, strictly for educational paper trading.
          </p>
        </div>
      </div>
    </div>
  );
};

