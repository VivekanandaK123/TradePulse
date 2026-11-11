import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData, UTCTimestamp } from 'lightweight-charts';
import { CandleBar, PredictionSignal, MarketStatus } from '../types/api';
import { TimeZoneMode, formatPrice, formatDateWithTime, formatCountdown } from '../lib/formatters';
import { Zap, Play, Pause, AlertTriangle, Clock, RefreshCw } from 'lucide-react';

interface LiveChartProps {
  recentBars: CandleBar[];
  formingBar?: CandleBar;
  latestSignal?: PredictionSignal;
  marketStatus?: MarketStatus;
  isAutoMode: boolean;
  onToggleAutoMode: () => void;
  onTestNow: () => void;
  isTestingNow: boolean;
  timeZone: TimeZoneMode;
  secondsToNextClose?: number;
}

export const LiveChart: React.FC<LiveChartProps> = ({
  recentBars,
  formingBar,
  latestSignal,
  marketStatus,
  isAutoMode,
  onToggleAutoMode,
  onTestNow,
  isTestingNow,
  timeZone,
  secondsToNextClose = 180,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const [hoveredCandle, setHoveredCandle] = useState<CandleBar | null>(null);

  const isOpen = marketStatus?.is_open ?? false;

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
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
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
    if (!candleSeriesRef.current || !volumeSeriesRef.current || recentBars.length === 0) return;

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

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-col gap-2.5">
      {/* Header & Primary Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-2.5">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-blue-500 animate-ping" />
          <span className="font-semibold text-sm tracking-wider text-slate-200 uppercase">
            Live Testing Chart
          </span>
          <span className="text-xs bg-slate-800 border border-slate-700 text-slate-300 px-2 py-0.5 rounded font-mono">
            5-Minute Closed Bars
          </span>
        </div>

        {/* Primary Action Buttons */}
        <div className="flex items-center gap-2">
          {/* Auto Mode Toggle */}
          <button
            onClick={onToggleAutoMode}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition ${
              isAutoMode
                ? 'bg-blue-600/20 text-blue-300 border-blue-500/40 shadow-sm'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
            title="Auto-refresh on every candle close (Key: A)"
          >
            {isAutoMode ? <Play className="h-3.5 w-3.5 text-blue-400 animate-spin" /> : <Pause className="h-3.5 w-3.5" />}
            <span>Auto Mode {isAutoMode ? 'ON' : 'OFF'}</span>
            <span className="text-[10px] text-slate-400 font-mono">(A)</span>
          </button>

          {/* Test Now Button */}
          <button
            onClick={onTestNow}
            disabled={isTestingNow || (!isOpen && false)}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-semibold text-white shadow-md transition ${
              isTestingNow
                ? 'bg-blue-500/50 cursor-wait'
                : 'bg-blue-600 hover:bg-blue-500 active:scale-95 shadow-blue-500/20'
            }`}
            title="Fetch latest closed bar and evaluate prediction (Key: T)"
          >
            {isTestingNow ? (
              <RefreshCw className="h-4 w-4 animate-spin text-white" />
            ) : (
              <Zap className="h-4 w-4 text-amber-300 fill-amber-300" />
            )}
            <span>Test Now</span>
            <span className="text-xs bg-blue-700 px-1.5 py-0.5 rounded font-mono text-blue-200">T</span>
          </button>
        </div>
      </div>

      {/* Status Bar / Forming Bar Notification */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs bg-slate-950/70 px-3 py-1.5 rounded border border-slate-800">
        <div className="flex items-center gap-2 text-slate-300">
          <Clock className="h-3.5 w-3.5 text-slate-400" />
          <span>Next 5-min candle closes in: <strong className="font-mono text-blue-400">{formatCountdown(secondsToNextClose)}</strong></span>
        </div>

        {!isOpen && (
          <div className="flex items-center gap-1.5 text-amber-400">
            <AlertTriangle className="h-3.5 w-3.5" />
            <span>Market closed — showing last completed session</span>
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
      <div className="relative">
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
