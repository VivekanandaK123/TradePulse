import React, { useState, useEffect, useCallback } from 'react';
import { 
  fetchHealth, 
  fetchMeta, 
  fetchHistory, 
  fetchOOFSignals, 
  callTestNow, 
  fetchSignalsLog, 
  fetchOutcomes,
  startReplay,
  stepReplay,
  pauseReplay,
  stopReplay,
  fetchReplayState,
  fetchReplaySessions
} from './api/client';
import { 
  HealthResponse, 
  MetaResponse, 
  HistoryResponse, 
  OOFSignalsResponse, 
  PredictionSignal, 
  CandleBar, 
  SignalsLogResponse, 
  OutcomesResponse,
  ReplayState
} from './types/api';
import { TimeZoneMode } from './lib/formatters';

import { Header } from './components/Header';
import { HistoricalChart } from './components/HistoricalChart';
import { LiveChart } from './components/LiveChart';
import { PredictionCard } from './components/PredictionCard';
import { Scoreboard } from './components/Scoreboard';
import { SignalLog } from './components/SignalLog';
import { ModelCard } from './components/ModelCard';
import { GuideDrawer } from './components/GuideDrawer';
import { ReplayControls } from './components/ReplayControls';
import { Footer } from './components/Footer';

import { Columns, Rows, AlertTriangle } from 'lucide-react';

export const App: React.FC = () => {
  // Application Data States
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [oofSignals, setOofSignals] = useState<OOFSignalsResponse | null>(null);
  const [signalsLog, setSignalsLog] = useState<SignalsLogResponse | null>(null);
  const [outcomes, setOutcomes] = useState<OutcomesResponse | null>(null);

  // Live Prediction State
  const [latestSignal, setLatestSignal] = useState<PredictionSignal | undefined>();
  const [recentBars, setRecentBars] = useState<CandleBar[]>([]);
  const [formingBar, setFormingBar] = useState<CandleBar | undefined>();
  const [isTestingNow, setIsTestingNow] = useState<boolean>(false);
  const [historyRange, setHistoryRange] = useState<string>('1M');

  // Replay State
  const [replayState, setReplayState] = useState<ReplayState>({
    is_active: false,
    status: 'STOPPED',
    session_date: '2026-11-06',
    current_bar_index: 0,
    total_bars: 78,
    speed_multiplier: 1.0,
    signals_emitted_count: 0,
  });

  // UI Settings (Persisted in localStorage)
  const [timeZone, setTimeZone] = useState<TimeZoneMode>(() => {
    return (localStorage.getItem('tp_timezone') as TimeZoneMode) || 'ET';
  });
  const [isDarkTheme, setIsDarkTheme] = useState<boolean>(() => {
    return localStorage.getItem('tp_theme') !== 'light';
  });
  const [chartLayout, setChartLayout] = useState<'stacked' | 'side-by-side'>(() => {
    return (localStorage.getItem('tp_chart_layout') as any) || 'stacked';
  });

  // Controls & Modals
  const [isAutoMode, setIsAutoMode] = useState<boolean>(false);
  const [isGuideOpen, setIsGuideOpen] = useState<boolean>(false);
  const [guideAnchor, setGuideAnchor] = useState<string | undefined>();
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [availableSessions, setAvailableSessions] = useState<string[]>([
    '2026-11-06', '2026-11-05', '2026-11-04', '2026-11-03', '2026-10-31'
  ]);

  // Countdown timer for 5-minute candle boundary
  const [secondsToNextClose, setSecondsToNextClose] = useState<number>(180);

  // 1. Initial Data Fetching
  const loadInitialData = useCallback(async () => {
    try {
      const [hData, mData, histData, oofData, sigsData, outData, sessData] = await Promise.all([
        fetchHealth(),
        fetchMeta(),
        fetchHistory(historyRange),
        fetchOOFSignals(),
        fetchSignalsLog(),
        fetchOutcomes(),
        fetchReplaySessions(),
      ]);

      setHealth(hData);
      setMeta(mData);
      setHistory(histData);
      setOofSignals(oofData);
      setSignalsLog(sigsData);
      setOutcomes(outData);
      if (sessData && sessData.length > 0) {
        setAvailableSessions(sessData);
      }

      // Perform initial test-now to populate live chart with latest candle
      const testRes = await callTestNow();
      setLatestSignal(testRes.signal);
      setRecentBars(testRes.recent_bars);
      setFormingBar(testRes.forming_bar);
    } catch (err: any) {
      console.error('Failed to load initial data:', err);
      setErrorMessage(`Connection Error: ${err.message}`);
    }
  }, [historyRange]);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // Handle Range change on Historical Chart
  const handleSelectRange = async (range: string) => {
    setHistoryRange(range);
    try {
      const res = await fetchHistory(range);
      setHistory(res);
    } catch (err: any) {
      console.error('Error changing range:', err);
    }
  };

  // 2. Test Now Trigger
  const handleTestNow = async () => {
    setIsTestingNow(true);
    setErrorMessage(null);
    try {
      const res = await callTestNow();
      setLatestSignal(res.signal);
      setRecentBars(res.recent_bars);
      setFormingBar(res.forming_bar);

      // Refresh signals log and outcomes
      const [sigs, outs] = await Promise.all([fetchSignalsLog(), fetchOutcomes()]);
      setSignalsLog(sigs);
      setOutcomes(outs);
    } catch (err: any) {
      console.error('Test Now failed:', err);
      setErrorMessage(`Test Now failed: ${err.message}. Retrying...`);
    } finally {
      setIsTestingNow(false);
    }
  };

  // 3. Auto Mode SSE Stream / Polling
  useEffect(() => {
    if (!isAutoMode) return;

    let eventSource: EventSource | null = null;
    try {
      eventSource = new EventSource('/api/live/stream');
      eventSource.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.data) {
            setLatestSignal(payload.data.signal);
            setRecentBars(payload.data.recent_bars);
            setFormingBar(payload.data.forming_bar);
          }
        } catch (e) {
          console.error('Error parsing SSE event:', e);
        }
      };
      eventSource.onerror = () => {
        // Fallback polling if SSE is disconnected
        handleTestNow();
      };
    } catch (e) {
      console.warn('SSE not supported, fallback to interval polling.');
    }

    // Interval countdown
    const interval = setInterval(() => {
      setSecondsToNextClose((prev) => (prev <= 1 ? 300 : prev - 1));
    }, 1000);

    return () => {
      if (eventSource) eventSource.close();
      clearInterval(interval);
    };
  }, [isAutoMode]);

  // 4. Replay Mode Handlers
  const handleStartReplay = async (date: string, speed: number) => {
    try {
      const state = await startReplay(date, speed);
      setReplayState(state);
    } catch (err: any) {
      setErrorMessage(`Failed to start replay: ${err.message}`);
    }
  };

  const handleStepReplay = async () => {
    try {
      const state = await stepReplay();
      setReplayState(state);
      if (state.current_signal) setLatestSignal(state.current_signal);
      if (state.current_bar) setRecentBars((prev) => [...prev.slice(-149), state.current_bar!]);
    } catch (err: any) {
      setErrorMessage(`Step replay failed: ${err.message}`);
    }
  };

  const handlePauseReplay = async () => {
    try {
      const state = await pauseReplay();
      setReplayState(state);
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleStopReplay = async () => {
    try {
      const state = await stopReplay();
      setReplayState(state);
      loadInitialData();
    } catch (err: any) {
      console.error(err);
    }
  };

  // Replay Auto-Playback Interval
  useEffect(() => {
    if (!replayState.is_active || replayState.status !== 'PLAYING') return;

    const multiplier = replayState.speed_multiplier || 1.0;
    const intervalMs = Math.max(250, Math.round(2500 / multiplier));

    const timer = setInterval(() => {
      handleStepReplay();
    }, intervalMs);

    return () => clearInterval(timer);
  }, [replayState.is_active, replayState.status, replayState.speed_multiplier, replayState.current_bar_index]);

  const handleToggleReplayMode = () => {
    if (replayState.is_active) {
      handleStopReplay();
    } else {
      const defaultDate = availableSessions[availableSessions.length - 1] || '2026-11-06';
      handleStartReplay(defaultDate, 1.0);
    }
  };

  // 5. Global Keyboard Shortcuts: T (Test Now), A (Auto Mode), R (Replay), ? (Help)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        handleTestNow();
      } else if (e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        setIsAutoMode((prev) => !prev);
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleToggleReplayMode();
      } else if (e.key === '?') {
        e.preventDefault();
        setIsGuideOpen((prev) => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [replayState.is_active, availableSessions]);

  // Persist Preferences
  const handleToggleTimeZone = () => {
    const nextTz: TimeZoneMode = timeZone === 'ET' ? 'LOCAL' : 'ET';
    setTimeZone(nextTz);
    localStorage.setItem('tp_timezone', nextTz);
  };

  const handleToggleTheme = () => {
    const nextTheme = !isDarkTheme;
    setIsDarkTheme(nextTheme);
    localStorage.setItem('tp_theme', nextTheme ? 'dark' : 'light');
    if (nextTheme) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  const handleToggleChartLayout = () => {
    const next = chartLayout === 'stacked' ? 'side-by-side' : 'stacked';
    setChartLayout(next);
    localStorage.setItem('tp_chart_layout', next);
  };

  return (
    <div className={`min-h-screen flex flex-col ${
      isDarkTheme ? 'dark bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'
    } ${replayState.is_active ? 'border-t-2 border-amber-500' : ''}`}>
      {/* Top Header */}
      <Header
        marketStatus={health?.market_status}
        feedName={health?.feed_name}
        modelVersion={health?.model_version}
        lastBarTimeNy={latestSignal?.bar_timestamp_ny}
        timeZone={timeZone}
        onToggleTimeZone={handleToggleTimeZone}
        isDarkTheme={isDarkTheme}
        onToggleTheme={handleToggleTheme}
        onOpenGuide={() => setIsGuideOpen(true)}
        isReplayActive={replayState.is_active}
        onToggleReplayMode={handleToggleReplayMode}
      />

      {/* Replay Controls Banner if active */}
      <ReplayControls
        replayState={replayState}
        availableSessions={availableSessions}
        onStartReplay={handleStartReplay}
        onStepReplay={handleStepReplay}
        onPauseReplay={handlePauseReplay}
        onStopReplay={handleStopReplay}
      />

      {/* Error Toast */}
      {errorMessage && (
        <div className="bg-rose-950 border-b border-rose-800 text-rose-200 px-4 py-2 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-xs text-rose-300 hover:text-white underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 p-3.5 max-w-[1920px] mx-auto w-full flex flex-col lg:flex-row gap-3.5">
        {/* Left Column: Charts Area */}
        <div className="flex-1 flex flex-col gap-3 min-w-0">
          {/* Chart Controls Bar */}
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-semibold text-slate-300">
              Interactive Financial Charts (5-Minute Timeframe)
            </span>
            <button
              onClick={handleToggleChartLayout}
              className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded text-slate-300 transition"
              title="Toggle between Stacked and Side-by-Side chart layout"
            >
              {chartLayout === 'stacked' ? <Columns className="h-3.5 w-3.5" /> : <Rows className="h-3.5 w-3.5" />}
              <span>Layout: {chartLayout === 'stacked' ? 'Stacked' : 'Side-by-Side'}</span>
            </button>
          </div>

          {/* Charts Display (Stacked or Side-by-Side) */}
          <div className={chartLayout === 'side-by-side' ? 'grid grid-cols-2 gap-3' : 'flex flex-col gap-3'}>
            {/* Chart 1: Historical Training Data */}
            <HistoricalChart
              bars={history?.bars ?? []}
              periodBands={history?.period_bands ?? []}
              oofSignals={oofSignals?.signals}
              selectedRange={historyRange}
              onSelectRange={handleSelectRange}
              timeZone={timeZone}
            />

            {/* Chart 2: Live Testing Chart (Main Screen) */}
            <LiveChart
              recentBars={recentBars}
              formingBar={formingBar}
              latestSignal={latestSignal}
              marketStatus={health?.market_status}
              isAutoMode={isAutoMode}
              onToggleAutoMode={() => setIsAutoMode(!isAutoMode)}
              onTestNow={handleTestNow}
              isTestingNow={isTestingNow}
              timeZone={timeZone}
              secondsToNextClose={secondsToNextClose}
            />
          </div>
        </div>

        {/* Right Sidebar: One-Second Trader Read */}
        <aside className="w-full lg:w-96 flex flex-col gap-3 flex-shrink-0">
          {/* 1. Prediction Card (Dominant Read) */}
          <PredictionCard
            signal={latestSignal}
            meta={meta || undefined}
            timeZone={timeZone}
            onOpenGuideAnchor={(anchor) => {
              setGuideAnchor(anchor);
              setIsGuideOpen(true);
            }}
          />

          {/* 2. Today's Scoreboard */}
          <Scoreboard
            outcomes={outcomes || undefined}
            meta={meta || undefined}
            signalsTodayCount={signalsLog?.count ?? 0}
          />

          {/* 3. Signal Audit Log */}
          <SignalLog
            signals={signalsLog?.signals ?? []}
            timeZone={timeZone}
          />

          {/* 4. Collapsible Model Card */}
          <ModelCard meta={meta || undefined} />
        </aside>
      </main>

      {/* User Guide Drawer */}
      <GuideDrawer
        isOpen={isGuideOpen}
        onClose={() => setIsGuideOpen(false)}
        activeAnchor={guideAnchor}
      />

      {/* Footer */}
      <Footer />
    </div>
  );
};

