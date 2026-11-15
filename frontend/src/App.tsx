import React, { useState, useEffect, useCallback } from 'react';
import { 
  fetchHealth, 
  fetchMeta, 
  fetchHistory, 
  fetchOOFSignals, 
  callTestNow, 
  clearLiveSession,
  fetchSignalsLog, 
  fetchOutcomes,
  startReplay,
  stepReplay,
  pauseReplay,
  resumeReplay,
  setReplaySpeed,
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

import { AlertTriangle } from 'lucide-react';

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
    session_date: '2026-11-09',
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

  // Controls & Modals
  const [isAutoMode, setIsAutoMode] = useState<boolean>(false);
  const [isGuideOpen, setIsGuideOpen] = useState<boolean>(false);
  const [guideAnchor, setGuideAnchor] = useState<string | undefined>();
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [availableSessions, setAvailableSessions] = useState<string[]>([
    '2026-11-09', '2026-11-10', '2026-11-11'
  ]);

  const getSecondsToNext5Min = () => {
    const nowSec = Math.floor(Date.now() / 1000);
    const rem = nowSec % 300;
    return rem === 0 ? 300 : 300 - rem;
  };

  // Countdown timer for 5-minute candle boundary (always ticks dynamically)
  const [secondsToNextClose, setSecondsToNextClose] = useState<number>(getSecondsToNext5Min);

  useEffect(() => {
    const timer = setInterval(() => {
      setSecondsToNextClose(getSecondsToNext5Min());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const isLiveTradingEnabled = (health?.market_status?.is_open ?? false) || replayState.is_active;

  useEffect(() => {
    if (!isLiveTradingEnabled && isAutoMode) {
      setIsAutoMode(false);
    }
  }, [isLiveTradingEnabled, isAutoMode]);

  const getTodayEtDate = () => {
    return new Intl.DateTimeFormat('en-CA', {
      timeZone: 'America/New_York',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).format(new Date());
  };

  // 1. Initial Data Fetching
  const loadInitialData = useCallback(async () => {
    try {
      const todayEt = getTodayEtDate();
      const [hData, mData, histData, oofData, sessData] = await Promise.all([
        fetchHealth(),
        fetchMeta(),
        fetchHistory(historyRange),
        fetchOOFSignals(),
        fetchReplaySessions(),
      ]);

      setHealth(hData);
      setMeta(mData);
      setHistory(histData);
      setOofSignals(oofData);
      if (sessData && sessData.length > 0) {
        setAvailableSessions(sessData);
      }

      if (hData.market_status?.is_open) {
        // Live trading session: load only today's session signals (starts empty before first prediction)
        const [sigsData, outData] = await Promise.all([
          fetchSignalsLog(todayEt),
          fetchOutcomes(),
        ]);
        setSignalsLog(sigsData);
        setOutcomes(outData);
      } else {
        // Outside trading hours: start Signal Audit Log empty until Replay or Live session begins
        setSignalsLog({ session: todayEt, count: 0, signals: [] });
        setOutcomes({ count: 0, outcomes: [] });
      }

      // Populate live chart with latest candle context
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

  // Clear all live/demo testing outputs to starting empty state
  const handleClear = async () => {
    setIsAutoMode(false);
    setRecentBars([]);
    setFormingBar(undefined);
    setLatestSignal(undefined);
    setSignalsLog({ session: replayState.is_active ? 'replay' : getTodayEtDate(), count: 0, signals: [] });
    setOutcomes({ count: 0, outcomes: [] });
    setErrorMessage(null);

    if (replayState.is_active) {
      try {
        await pauseReplay();
      } catch {}
      setReplayState((prev) => ({
        ...prev,
        status: 'STOPPED',
        current_bar_index: 0,
        signals_emitted_count: 0,
        current_bar: undefined,
        current_signal: undefined,
      }));
    } else {
      try {
        await clearLiveSession();
      } catch (err: any) {
        console.error('Failed to clear live session baseline:', err);
      }
    }
  };

  // Toggle Auto Mode: when turning ON, start fresh from the next closing candle
  const handleToggleAutoMode = async () => {
    if (!isLiveTradingEnabled) return;
    if (!isAutoMode) {
      await handleClear();
      setIsAutoMode(true);
    } else {
      setIsAutoMode(false);
    }
  };

  // 2. Test Now Trigger
  const handleTestNow = async () => {
    if (!isLiveTradingEnabled) return;
    setIsTestingNow(true);
    setErrorMessage(null);
    try {
      const res = await callTestNow();
      setLatestSignal(res.signal ?? undefined);
      setRecentBars(res.recent_bars ?? []);
      setFormingBar(res.forming_bar);

      // Refresh signals log and outcomes for today's live session
      const todayEt = getTodayEtDate();
      const [sigs, outs] = await Promise.all([fetchSignalsLog(todayEt), fetchOutcomes()]);
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
      eventSource.onmessage = async (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.data) {
            setLatestSignal(payload.data.signal ?? undefined);
            setRecentBars(payload.data.recent_bars ?? []);
            setFormingBar(payload.data.forming_bar);
            const todayEt = getTodayEtDate();
            const [sigs, outs] = await Promise.all([fetchSignalsLog(todayEt), fetchOutcomes()]);
            setSignalsLog(sigs);
            setOutcomes(outs);
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

    return () => {
      if (eventSource) eventSource.close();
    };
  }, [isAutoMode]);

  // 4. Replay Mode Handlers
  const handleStartReplay = async (date: string, speed: number) => {
    try {
      // Clear log & chart immediately so demo testing starts fresh
      setSignalsLog({ session: 'replay', count: 0, signals: [] });
      setOutcomes({ count: 0, outcomes: [] });
      setFormingBar(undefined);

      const state = await startReplay(date, speed);
      setReplayState(state);
      if (state.current_signal) setLatestSignal(state.current_signal);
      setRecentBars(state.current_bar ? [state.current_bar] : []);

      const [sigs, outs] = await Promise.all([fetchSignalsLog('replay'), fetchOutcomes('replay')]);
      setSignalsLog(sigs);
      setOutcomes(outs);
    } catch (err: any) {
      setErrorMessage(`Failed to start replay: ${err.message}`);
    }
  };

  const handleStepReplay = async () => {
    try {
      const state = await stepReplay();
      setReplayState(state);
      if (state.current_signal) setLatestSignal(state.current_signal);
      if (state.current_bar) {
        setRecentBars((prev) => {
          const filtered = prev.filter((b) => b.time !== state.current_bar!.time);
          return [...filtered.slice(-149), state.current_bar!];
        });
      }

      const [sigs, outs] = await Promise.all([fetchSignalsLog('replay'), fetchOutcomes('replay')]);
      setSignalsLog(sigs);
      setOutcomes(outs);
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

  const handleResumeReplay = async () => {
    try {
      const state = await resumeReplay();
      setReplayState(state);
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleChangeReplaySpeed = async (speed: number) => {
    if (replayState.current_bar_index === 0) {
      setReplayState((prev) => ({ ...prev, speed_multiplier: speed }));
      return;
    }
    try {
      const state = await setReplaySpeed(speed);
      setReplayState(state);
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleSelectReplaySession = (date: string) => {
    setSignalsLog({ session: 'replay', count: 0, signals: [] });
    setOutcomes({ count: 0, outcomes: [] });
    setRecentBars([]);
    setLatestSignal(undefined);
    setFormingBar(undefined);
    setReplayState((prev) => ({
      ...prev,
      is_active: true,
      status: 'STOPPED',
      session_date: date,
      current_bar_index: 0,
      signals_emitted_count: 0,
      current_bar: undefined,
      current_signal: undefined,
    }));
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
      const defaultDate = availableSessions[0] || '2026-11-09';
      handleSelectReplaySession(defaultDate);
    }
  };

  // 5. Global Keyboard Shortcuts: T (Test Now), A (Auto Mode), R (Replay), ? (Help)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (e.key === 't' || e.key === 'T') {
        if (!isLiveTradingEnabled) return;
        e.preventDefault();
        handleTestNow();
      } else if (e.key === 'a' || e.key === 'A') {
        if (!isLiveTradingEnabled) return;
        e.preventDefault();
        handleToggleAutoMode();
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
  }, [replayState.is_active, availableSessions, isLiveTradingEnabled]);

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
        onSelectSession={handleSelectReplaySession}
        onStartReplay={handleStartReplay}
        onStepReplay={handleStepReplay}
        onPauseReplay={handlePauseReplay}
        onResumeReplay={handleResumeReplay}
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
      <main className="flex-1 p-3.5 max-w-[1920px] mx-auto w-full flex flex-col gap-3.5">
        {/* Top Row: Full-width Historical Training Data */}
        <HistoricalChart
          bars={history?.bars ?? []}
          periodBands={history?.period_bands ?? []}
          oofSignals={oofSignals?.signals}
          selectedRange={historyRange}
          onSelectRange={handleSelectRange}
          timeZone={timeZone}
        />

        {/* Bottom Area: 2-Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5 items-start">
          {/* Left Column (~2/3): Live Testing Chart + Model Provenance and Card */}
          <div className="lg:col-span-8 flex flex-col gap-3.5 min-w-0">
            <LiveChart
              recentBars={recentBars}
              formingBar={formingBar}
              latestSignal={latestSignal}
              marketStatus={health?.market_status}
              isAutoMode={isAutoMode}
              onToggleAutoMode={handleToggleAutoMode}
              onTestNow={handleTestNow}
              onClear={handleClear}
              isTestingNow={isTestingNow}
              timeZone={timeZone}
              secondsToNextClose={secondsToNextClose}
              isReplayActive={replayState.is_active}
            />

            <ModelCard meta={meta || undefined} />
          </div>

          {/* Right Column (~1/3): Predicted Direction, Today's Scoreboard, Signal Audit Log */}
          <aside className="lg:col-span-4 flex flex-col gap-3.5 min-w-0">
            <PredictionCard
              signal={latestSignal}
              meta={meta || undefined}
              timeZone={timeZone}
              recentBars={recentBars}
              onOpenGuideAnchor={(anchor) => {
                setGuideAnchor(anchor);
                setIsGuideOpen(true);
              }}
            />

            <SignalLog
              signals={signalsLog?.signals ?? []}
              timeZone={timeZone}
            />

            <Scoreboard
              outcomes={outcomes || undefined}
              meta={meta || undefined}
              signalsTodayCount={signalsLog?.count ?? 0}
            />
          </aside>
        </div>
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

