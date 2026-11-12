import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData, UTCTimestamp } from 'lightweight-charts';
import { CandleBar, PredictionSignal, MarketStatus } from '../types/api';
import { TimeZoneMode, formatPrice, formatDateWithTime, formatCountdown, formatChartTick, formatChartTime } from '../lib/formatters';
import { Zap, Play, Pause, AlertTriangle, Clock, RefreshCw, Trash2 } from 'lucide-react';

interface LiveChartProps {
  recentBars: CandleBar[];
  formingBar?: CandleBar;
  latestSignal?: PredictionSignal;
  marketStatus?: MarketStatus;
  isAutoMode: boolean;
  onToggleAutoMode: () => void;
  onTestNow: () => void;
  onClear: () => void;
  isTestingNow: boolean;
  timeZone: TimeZoneMode;
  secondsToNextClose?: number;
  isReplayActive?: boolean;
}

export const LiveChart: React.FC<LiveChartProps> = ({
  recentBars,
  formingBar,
  latestSignal,
  marketStatus,
  isAutoMode,
  onToggleAutoMode,
  onTestNow,
  onClear,
  isTestingNow,
  timeZone,
  secondsToNextClose = 180,
  isReplayActive = false,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const [hoveredCandle, setHoveredCandle] = useState<CandleBar | null>(null);

  const isOpen = marketStatus?.is_open ?? false;
  const isSectionEnabled = isOpen || isReplayActive;

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 420,
      layout: {
        background: { color: '#0f172a' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      crosshair: {
        vertLine: { color: '#3b82f6', width: 1, style: 2 },
        horzLine: { color: '#3b82f6', width: 1, style: 2 },
      },
      localization: {
        timeFormatter: (t: any) => formatChartTime(t, timeZone),
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (t: any, markType: any) => formatChartTick(t, markType, timeZone),
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderUpColor: '#10b981',
      borderDownColor: '#ef4444',
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    const volumeSeries = chart.addHistogramSeries({
      color: '#334155',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    });
    chart.priceScale('').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.seriesData) {
        setHoveredCandle(null);
        return;
      }
      const data = param.seriesData.get(candleSeries) as any;
      if (data) {
        setHoveredCandle({
          time: Number(param.time),
          open: data.open,
          high: data.high,
          low: data.low,
          close: data.close,
          volume: 0,
        });
      }
    });

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Update Data and Markers
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current) return;

    if (recentBars.length === 0 && !formingBar) {
      candleSeriesRef.current.setData([]);
      volumeSeriesRef.current.setData([]);
      candleSeriesRef.current.setMarkers([]);
      setHoveredCandle(null);
      return;
    }

    const candleData: CandlestickData[] = [];
    const volumeData: HistogramData[] = [];

    const allBars = [...recentBars];
    if (formingBar) {
      allBars.push(formingBar);
    }

    // Sort chronologically
    allBars.sort((a, b) => a.time - b.time);

    allBars.forEach((b) => {
      const isForming = formingBar && b.time === formingBar.time;
      const isUp = b.close >= b.open;

      candleData.push({
        time: b.time as UTCTimestamp,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
        // Muted style for forming bar
        color: isForming ? '#475569' : undefined,
        borderColor: isForming ? '#64748b' : undefined,
        wickColor: isForming ? '#64748b' : undefined,
      });

      volumeData.push({
        time: b.time as UTCTimestamp,
        value: b.volume,
        color: isForming ? '#33415555' : (isUp ? '#05966955' : '#dc262655'),
      });
    });

    candleSeriesRef.current.setData(candleData);
    volumeSeriesRef.current.setData(volumeData);

    // Attach Signal Markers for closed candles
    const markers: any[] = [];
    if (latestSignal) {
      const isUp = latestSignal.direction === 'Up';
      const isDown = latestSignal.direction === 'Down';
      const isNeutral = latestSignal.direction === 'no clear signal';

      if (!isNeutral) {
        markers.push({
          time: latestSignal.bar_time as UTCTimestamp,
          position: isUp ? 'belowBar' : 'aboveBar',
          color: isUp ? '#10b981' : '#ef4444',
          shape: isUp ? 'arrowUp' : 'arrowDown',
          text: `${latestSignal.direction} ${latestSignal.confidence_pct.toFixed(0)}%`,
        });
      } else {
        markers.push({
          time: latestSignal.bar_time as UTCTimestamp,
          position: 'aboveBar',
          color: '#64748b',
          shape: 'circle',
          text: 'No Signal',
        });
      }

      // If outcome badge exists
      if (latestSignal.outcome) {
        const hit = latestSignal.outcome.is_hit;
        markers.push({
          time: (latestSignal.bar_time + 900) as UTCTimestamp, // 15m ahead
          position: hit ? 'aboveBar' : 'belowBar',
          color: hit ? '#10b981' : '#f59e0b',
          shape: 'square',
          text: hit ? `HIT (${latestSignal.outcome.realized_return_bps.toFixed(0)}bp)` : `MISS (${latestSignal.outcome.realized_return_bps.toFixed(0)}bp)`,
        });
      }
    }

    candleSeriesRef.current.setMarkers(markers);
    chartRef.current?.timeScale().fitContent();
  }, [recentBars, formingBar, latestSignal]);

  // Update Timezone formatting
  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.applyOptions({
      localization: {
        timeFormatter: (t: any) => formatChartTime(t, timeZone),
      },
      timeScale: {
        tickMarkFormatter: (t: any, markType: any) => formatChartTick(t, markType, timeZone),
      },
    });
  }, [timeZone]);

  return (
    <div className={`bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-col gap-2.5 transition-opacity ${
      !isSectionEnabled ? 'opacity-65' : ''
    }`}>
      {/* Header & Primary Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-2.5">
        <div className="flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${isSectionEnabled ? 'bg-blue-500 animate-ping' : 'bg-slate-600'}`} />
          <span className="font-semibold text-sm tracking-wider text-slate-200 uppercase">
            Live Testing Chart
          </span>
          <span className="text-xs bg-slate-800 border border-slate-700 text-slate-300 px-2 py-0.5 rounded font-mono">
            5-Minute Closed Bars
          </span>
          {!isSectionEnabled && (
            <span className="text-[11px] bg-amber-950/60 border border-amber-700/60 text-amber-300 px-2 py-0.5 rounded font-medium">
              Disabled (Non-Trading Hours)
            </span>
          )}
        </div>

        {/* Primary Action Buttons */}
        <div className="flex items-center gap-2">
          {/* Clear Button */}
          <button
            onClick={onClear}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border bg-slate-800 text-slate-300 border-slate-700 hover:bg-rose-950/60 hover:text-rose-200 hover:border-rose-800/70 transition"
            title="Clear live chart, predicted direction, audit log, and scoreboard before starting a new test"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Clear</span>
          </button>

          {/* Auto Mode Toggle */}
          <button
            onClick={onToggleAutoMode}
            disabled={!isSectionEnabled}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition ${
              !isSectionEnabled
                ? 'bg-slate-800/50 text-slate-500 border-slate-800 cursor-not-allowed'
                : isAutoMode
                ? 'bg-blue-600/20 text-blue-300 border-blue-500/40 shadow-sm'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
            title={isSectionEnabled ? 'Auto-refresh on every candle close (Key: A)' : 'Available during trading hours only'}
          >
            {isAutoMode && isSectionEnabled ? <Play className="h-3.5 w-3.5 text-blue-400 animate-spin" /> : <Pause className="h-3.5 w-3.5" />}
            <span>Auto Mode {isAutoMode && isSectionEnabled ? 'ON' : 'OFF'}</span>
            <span className="text-[10px] text-slate-400 font-mono">(A)</span>
          </button>

          {/* Test Now Button */}
          <button
            onClick={onTestNow}
            disabled={isTestingNow || !isSectionEnabled}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-semibold text-white shadow-md transition ${
              !isSectionEnabled
                ? 'bg-slate-700/60 text-slate-400 cursor-not-allowed shadow-none'
                : isTestingNow
                ? 'bg-blue-500/50 cursor-wait'
                : 'bg-blue-600 hover:bg-blue-500 active:scale-95 shadow-blue-500/20'
            }`}
            title={isSectionEnabled ? 'Fetch latest closed bar and evaluate prediction (Key: T)' : 'Available during trading hours only'}
          >
            {isTestingNow ? (
              <RefreshCw className="h-4 w-4 animate-spin text-white" />
            ) : (
              <Zap className={`h-4 w-4 ${isSectionEnabled ? 'text-amber-300 fill-amber-300' : 'text-slate-400'}`} />
            )}
            <span>Test Now</span>
            <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${isSectionEnabled ? 'bg-blue-700 text-blue-200' : 'bg-slate-800 text-slate-500'}`}>T</span>
          </button>
        </div>
      </div>

      {/* Status Bar / Forming Bar Notification */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs bg-slate-950/70 px-3 py-1.5 rounded border border-slate-800">
        <div className="flex items-center gap-2 text-slate-300">
          <Clock className="h-3.5 w-3.5 text-slate-400" />
          <span>
            Next 5-min candle closes in:{' '}
            <strong className="font-mono text-blue-400">
              {isOpen ? formatCountdown(secondsToNextClose) : '--:--'}
            </strong>
          </span>
        </div>

        {!isOpen && (
          <div className="flex items-center gap-1.5 text-amber-400">
            <AlertTriangle className="h-3.5 w-3.5" />
            <span>
              {isReplayActive
                ? 'Market closed — Replay Mode active'
                : 'Market closed — live testing enabled during trading hours only (09:30–16:00 ET)'}
            </span>
          </div>
        )}

        {formingBar && (
          <div className="flex items-center gap-1.5 text-slate-400 font-mono text-[11px]">
            <span className="h-1.5 w-1.5 rounded-full bg-slate-500" />
            <span>Muted candle = forming live bar (no signal attached)</span>
          </div>
        )}
      </div>

      {/* Chart Canvas */}
      <div className={`relative ${!isSectionEnabled ? 'pointer-events-none select-none' : ''}`}>
        {/* Tooltip Overlay */}
        {hoveredCandle && (
          <div className="absolute top-2 left-2 z-10 bg-slate-950/90 border border-slate-800 rounded px-2.5 py-1 text-xs font-mono flex items-center gap-3 text-slate-300 shadow">
            <span>{formatDateWithTime(hoveredCandle.time, timeZone)}</span>
            <span>O: <strong className="text-slate-100">{formatPrice(hoveredCandle.open)}</strong></span>
            <span>H: <strong className="text-emerald-400">{formatPrice(hoveredCandle.high)}</strong></span>
            <span>L: <strong className="text-rose-400">{formatPrice(hoveredCandle.low)}</strong></span>
            <span>C: <strong className="text-slate-100">{formatPrice(hoveredCandle.close)}</strong></span>
          </div>
        )}

        <div ref={chartContainerRef} className="w-full rounded overflow-hidden" />
      </div>
    </div>
  );
};
