import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData, LineData, UTCTimestamp } from 'lightweight-charts';
import { CandleBar, PredictionSignal, MarketStatus } from '../types/api';
import { TimeZoneMode, formatPrice, formatDateWithTime, formatCountdown, formatChartTick, formatChartTime } from '../lib/formatters';
import { Zap, Play, Pause, AlertTriangle, Clock, RefreshCw, Trash2, Layers, ShieldCheck, ShieldAlert, Crosshair } from 'lucide-react';

export interface SRZone {
  topPrice: number;
  bottomPrice: number;
  type: 'resistance' | 'support';
  title: string;
  badgeLabel: string;
  actionGuidance: string;
}

export interface ComputedSRData {
  resistancePrice: number | null;
  supportPrice: number | null;
  resistanceZone: SRZone | null;
  supportZone: SRZone | null;
  targetZone: SRZone | null;
  invalidationZone: SRZone | null;
}

interface VisualZoneItem extends SRZone {
  yTop: number;
  yBottom: number;
  height: number;
}

export function calculateSRData(bars: CandleBar[], signal?: PredictionSignal): ComputedSRData {
  if (bars.length < 3) {
    return {
      resistancePrice: null,
      supportPrice: null,
      resistanceZone: null,
      supportZone: null,
      targetZone: null,
      invalidationZone: null,
    };
  }

  const allHighs = bars.map((b) => b.high);
  const allLows = bars.map((b) => b.low);
  const sessionHigh = Math.max(...allHighs);
  const sessionLow = Math.min(...allLows);
  const lastClose = bars[bars.length - 1].close;

  // Local swing peaks & troughs (3-bar rolling window)
  const swingHighs: number[] = [];
  const swingLows: number[] = [];

  for (let i = 1; i < bars.length - 1; i++) {
    if (bars[i].high >= bars[i - 1].high && bars[i].high >= bars[i + 1].high) {
      swingHighs.push(bars[i].high);
    }
    if (bars[i].low <= bars[i - 1].low && bars[i].low <= bars[i + 1].low) {
      swingLows.push(bars[i].low);
    }
  }

  // Classic Floor Pivot Points as baseline anchor
  const pivot = (sessionHigh + sessionLow + lastClose) / 3;
  const r1 = 2 * pivot - sessionLow;
  const s1 = 2 * pivot - sessionHigh;

  // Resistance (Red / Supply / When to Back Out)
  const validHighsAbove = swingHighs.filter((h) => h > lastClose);
  const resPrice = validHighsAbove.length > 0
    ? Math.min(...validHighsAbove)
    : (sessionHigh > lastClose ? sessionHigh : r1);

  const resBuffer = resPrice * 0.0007; // ~7 bps zone width
  const resTop = Math.max(resPrice + resBuffer, sessionHigh);
  const resBottom = Math.max(resPrice - resBuffer, lastClose * 1.0002);

  const resistanceZone: SRZone = {
    topPrice: resTop,
    bottomPrice: resBottom,
    type: 'resistance',
    title: 'SUPPLY / RESISTANCE ZONE',
    badgeLabel: 'WHEN TO BACK OUT / TAKE PROFIT',
    actionGuidance: 'Strong seller interest. Do not chase longs here; look to lock in gains or prepare stop.',
  };

  // Support (Green / Demand / When to Join)
  const validLowsBelow = swingLows.filter((l) => l < lastClose);
  const suppPrice = validLowsBelow.length > 0
    ? Math.max(...validLowsBelow)
    : (sessionLow < lastClose ? sessionLow : s1);

  const suppBuffer = suppPrice * 0.0007; // ~7 bps zone width
  const suppTop = Math.min(suppPrice + suppBuffer, lastClose * 0.9998);
  const suppBottom = Math.min(suppPrice - suppBuffer, sessionLow);

  const supportZone: SRZone = {
    topPrice: suppTop,
    bottomPrice: suppBottom,
    type: 'support',
    title: 'DEMAND / SUPPORT ZONE',
    badgeLabel: 'WHEN TO JOIN TRADE / REBOUND',
    actionGuidance: 'Strong buyer defense. High-probability area to join long positions or set stops.',
  };

  // Active Trade Execution Corridor (if signal exists)
  let targetZone: SRZone | null = null;
  let invalidationZone: SRZone | null = null;

  if (signal && (signal.direction === 'Up' || signal.direction === 'Down')) {
    const isUp = signal.direction === 'Up';
    const thetaPct = 0.0015; // 15 bps
    const targetPrice = isUp
      ? signal.close_price * (1 + thetaPct)
      : signal.close_price * (1 - thetaPct);
    const stopPrice = isUp
      ? signal.close_price * (1 - thetaPct)
      : signal.close_price * (1 + thetaPct);

    if (isUp) {
      targetZone = {
        topPrice: targetPrice,
        bottomPrice: signal.close_price,
        type: 'support',
        title: 'TARGET EXPANSION ZONE (+15 bps)',
        badgeLabel: 'PROFIT CAPTURE ZONE',
        actionGuidance: `Aim to capture +15 bps up to $${targetPrice.toFixed(2)}.`,
      };
      invalidationZone = {
        topPrice: signal.close_price,
        bottomPrice: stopPrice,
        type: 'resistance',
        title: 'INVALIDATION CUTOFF (-15 bps)',
        badgeLabel: 'BACK OUT IMMEDIATELY',
        actionGuidance: `Hypothesis fails if price falls below $${stopPrice.toFixed(2)}. Exit immediately.`,
      };
    } else {
      targetZone = {
        topPrice: signal.close_price,
        bottomPrice: targetPrice,
        type: 'support',
        title: 'SHORT TARGET ZONE (-15 bps)',
        badgeLabel: 'PROFIT CAPTURE ZONE',
        actionGuidance: `Aim to capture -15 bps down to $${targetPrice.toFixed(2)}.`,
      };
      invalidationZone = {
        topPrice: stopPrice,
        bottomPrice: signal.close_price,
        type: 'resistance',
        title: 'INVALIDATION CUTOFF (+15 bps)',
        badgeLabel: 'BACK OUT IMMEDIATELY',
        actionGuidance: `Hypothesis fails if price rises above $${stopPrice.toFixed(2)}. Exit immediately.`,
      };
    }
  }

  return {
    resistancePrice: resPrice,
    supportPrice: suppPrice,
    resistanceZone,
    supportZone,
    targetZone,
    invalidationZone,
  };
}

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
  const vwapSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const targetPriceLineRef = useRef<any>(null);
  const stopPriceLineRef = useRef<any>(null);
  const resistancePriceLineRef = useRef<any>(null);
  const supportPriceLineRef = useRef<any>(null);

  const [showVwap, setShowVwap] = useState<boolean>(true);
  const [showZones, setShowZones] = useState<boolean>(true);
  const [showSRLines, setShowSRLines] = useState<boolean>(true);
  const [visualZones, setVisualZones] = useState<VisualZoneItem[]>([]);
  const [containerWidth, setContainerWidth] = useState<number>(600);
  const [hoveredCandle, setHoveredCandle] = useState<CandleBar | null>(null);

  const isOpen = marketStatus?.is_open ?? false;
  const isSectionEnabled = isOpen || isReplayActive;

  // Recompute visual pixel coordinates for S/R shaded zones
  const updateZoneOverlay = useCallback(() => {
    if (!candleSeriesRef.current || !chartContainerRef.current) return;
    const allBars = [...recentBars];
    if (formingBar) allBars.push(formingBar);
    if (allBars.length === 0) {
      setVisualZones([]);
      return;
    }

    const srData = calculateSRData(allBars, latestSignal);
    const zonesToDraw: SRZone[] = [];
    if (srData.resistanceZone) zonesToDraw.push(srData.resistanceZone);
    if (srData.supportZone) zonesToDraw.push(srData.supportZone);

    const mapped: VisualZoneItem[] = [];
    zonesToDraw.forEach((z) => {
      const y1 = candleSeriesRef.current?.priceToCoordinate(z.topPrice);
      const y2 = candleSeriesRef.current?.priceToCoordinate(z.bottomPrice);
      if (y1 !== null && y2 !== null && y1 !== undefined && y2 !== undefined) {
        const topY = Math.min(y1, y2);
        const bottomY = Math.max(y1, y2);
        const height = Math.max(bottomY - topY, 14);
        mapped.push({
          ...z,
          yTop: topY,
          yBottom: bottomY,
          height: height,
        });
      }
    });

    setVisualZones(mapped);
    if (chartContainerRef.current) {
      setContainerWidth(chartContainerRef.current.clientWidth);
    }
  }, [recentBars, formingBar, latestSignal]);

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

    const vwapSeries = chart.addLineSeries({
      color: '#38bdf8', // sky-400
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      title: 'VWAP',
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    vwapSeriesRef.current = vwapSeries;

    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.seriesData) {
        setHoveredCandle(null);
        return;
      }
      const data = param.seriesData.get(candleSeries) as any;
      const vwapVal = param.seriesData.get(vwapSeries) as any;
      if (data) {
        setHoveredCandle({
          time: Number(param.time),
          open: data.open,
          high: data.high,
          low: data.low,
          close: data.close,
          volume: 0,
          vwap: vwapVal?.value ?? undefined,
        });
      }
    });

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        setContainerWidth(chartContainerRef.current.clientWidth);
        updateZoneOverlay();
      }
    };
    window.addEventListener('resize', handleResize);

    // Re-sync zones when user zooms or pans
    chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
      updateZoneOverlay();
    });

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [updateZoneOverlay]);

  // Update Data and Markers
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current || !vwapSeriesRef.current) return;

    if (recentBars.length === 0 && !formingBar) {
      candleSeriesRef.current.setData([]);
      volumeSeriesRef.current.setData([]);
      vwapSeriesRef.current.setData([]);
      candleSeriesRef.current.setMarkers([]);
      if (targetPriceLineRef.current) {
        candleSeriesRef.current.removePriceLine(targetPriceLineRef.current);
        targetPriceLineRef.current = null;
      }
      if (stopPriceLineRef.current) {
        candleSeriesRef.current.removePriceLine(stopPriceLineRef.current);
        stopPriceLineRef.current = null;
      }
      setHoveredCandle(null);
      return;
    }

    const candleData: CandlestickData[] = [];
    const volumeData: HistogramData[] = [];
    const vwapData: LineData[] = [];

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

      if (b.vwap !== undefined && b.vwap !== null) {
        vwapData.push({
          time: b.time as UTCTimestamp,
          value: b.vwap,
        });
      }
    });

    candleSeriesRef.current.setData(candleData);
    volumeSeriesRef.current.setData(volumeData);

    if (showVwap) {
      vwapSeriesRef.current.setData(vwapData);
    } else {
      vwapSeriesRef.current.setData([]);
    }

    // Dynamic Target & Invalidation Stop Price Lines on Candlestick Chart
    if (targetPriceLineRef.current) {
      candleSeriesRef.current.removePriceLine(targetPriceLineRef.current);
      targetPriceLineRef.current = null;
    }
    if (stopPriceLineRef.current) {
      candleSeriesRef.current.removePriceLine(stopPriceLineRef.current);
      stopPriceLineRef.current = null;
    }

    if (latestSignal && (latestSignal.direction === 'Up' || latestSignal.direction === 'Down')) {
      const isUp = latestSignal.direction === 'Up';
      const thetaPct = 0.0015; // 15 bps
      const targetPrice = isUp
        ? latestSignal.close_price * (1 + thetaPct)
        : latestSignal.close_price * (1 - thetaPct);
      const stopPrice = isUp
        ? latestSignal.close_price * (1 - thetaPct)
        : latestSignal.close_price * (1 + thetaPct);

      targetPriceLineRef.current = candleSeriesRef.current.createPriceLine({
        price: Number(targetPrice.toFixed(2)),
        color: isUp ? '#10b981' : '#ef4444',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: isUp ? 'TARGET (+15bp)' : 'TARGET (-15bp)',
      });

      stopPriceLineRef.current = candleSeriesRef.current.createPriceLine({
        price: Number(stopPrice.toFixed(2)),
        color: '#f59e0b',
        lineWidth: 1,
        lineStyle: 1, // Dotted
        axisLabelVisible: true,
        title: 'STOP LOSS',
      });
    }

    // Dynamic Support & Resistance Price Lines
    if (resistancePriceLineRef.current) {
      candleSeriesRef.current.removePriceLine(resistancePriceLineRef.current);
      resistancePriceLineRef.current = null;
    }
    if (supportPriceLineRef.current) {
      candleSeriesRef.current.removePriceLine(supportPriceLineRef.current);
      supportPriceLineRef.current = null;
    }

    if (showSRLines && allBars.length >= 3) {
      const srData = calculateSRData(allBars, latestSignal);
      if (srData.resistancePrice) {
        resistancePriceLineRef.current = candleSeriesRef.current.createPriceLine({
          price: Number(srData.resistancePrice.toFixed(2)),
          color: '#ef4444',
          lineWidth: 1,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: 'Resistance',
        });
      }
      if (srData.supportPrice) {
        supportPriceLineRef.current = candleSeriesRef.current.createPriceLine({
          price: Number(srData.supportPrice.toFixed(2)),
          color: '#10b981',
          lineWidth: 1,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: 'Support',
        });
      }
    }

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

    // Recompute SVG zone overlay coordinates after chart layout
    const timer = setTimeout(() => {
      updateZoneOverlay();
    }, 60);
    return () => clearTimeout(timer);
  }, [recentBars, formingBar, latestSignal, showVwap, showSRLines, showZones, updateZoneOverlay]);

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
          {/* S/R Shaded Zones Toggle */}
          <button
            onClick={() => setShowZones(!showZones)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono font-medium border transition ${
              showZones
                ? 'bg-emerald-950/70 text-emerald-300 border-emerald-700/80 shadow-sm'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
            title="Toggle TradingView-style Support (Green/Join) and Resistance (Red/Back) Shaded Zones"
          >
            <ShieldAlert className="h-3.5 w-3.5 text-emerald-400" />
            <span>Zones {showZones ? 'ON' : 'OFF'}</span>
          </button>

          {/* S/R Horizontal Price Lines Toggle */}
          <button
            onClick={() => setShowSRLines(!showSRLines)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono font-medium border transition ${
              showSRLines
                ? 'bg-blue-950/60 text-blue-300 border-blue-700/80 shadow-sm'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
            title="Toggle Horizontal Support & Resistance Pivot Lines on Chart"
          >
            <Crosshair className="h-3.5 w-3.5 text-blue-400" />
            <span>S/R Lines {showSRLines ? 'ON' : 'OFF'}</span>
          </button>

          {/* VWAP Overlay Toggle */}
          <button
            onClick={() => setShowVwap(!showVwap)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono font-medium border transition ${
              showVwap
                ? 'bg-sky-950/60 text-sky-300 border-sky-700/80'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
            title="Toggle Intraday VWAP (Volume-Weighted Average Price) overlay"
          >
            <Layers className="h-3.5 w-3.5 text-sky-400" />
            <span>VWAP {showVwap ? 'ON' : 'OFF'}</span>
          </button>

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

      {/* Trader Actionable Zone Guide Banner */}
      {showZones && (
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs bg-slate-950/90 px-3 py-1.5 rounded border border-slate-800">
          <div className="flex items-center flex-wrap gap-2.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1">
              <ShieldCheck className="h-3.5 w-3.5 text-blue-400" />
              Trading Zones:
            </span>
            {/* Green Demand Zone */}
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-950/70 border border-emerald-700/60 text-emerald-300 text-[11px] font-mono">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              <span><strong>Green (Support / Demand):</strong> When to Join (Buyer Defense / Accumulation)</span>
            </div>
            {/* Red Supply Zone */}
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-rose-950/70 border border-rose-700/60 text-rose-300 text-[11px] font-mono">
              <span className="h-2 w-2 rounded-full bg-rose-400" />
              <span><strong>Red (Resistance / Supply):</strong> When to Back Out (Seller Wall / Lock Profit)</span>
            </div>
          </div>
          <span className="text-[10px] text-slate-400 font-mono hidden md:inline">
            θ = ±15 bps execution bands
          </span>
        </div>
      )}

      {/* Chart Canvas with S/R Shaded Zones Overlay */}
      <div className={`relative ${!isSectionEnabled ? 'pointer-events-none select-none' : ''}`}>
        {/* Tooltip Overlay */}
        {hoveredCandle && (
          <div className="absolute top-2 left-2 z-10 bg-slate-950/90 border border-slate-800 rounded px-2.5 py-1 text-xs font-mono flex items-center gap-3 text-slate-300 shadow">
            <span>{formatDateWithTime(hoveredCandle.time, timeZone)}</span>
            <span>O: <strong className="text-slate-100">{formatPrice(hoveredCandle.open)}</strong></span>
            <span>H: <strong className="text-emerald-400">{formatPrice(hoveredCandle.high)}</strong></span>
            <span>L: <strong className="text-rose-400">{formatPrice(hoveredCandle.low)}</strong></span>
            <span>C: <strong className="text-slate-100">{formatPrice(hoveredCandle.close)}</strong></span>
            {hoveredCandle.vwap !== undefined && (
              <span>VWAP: <strong className="text-sky-400">{formatPrice(hoveredCandle.vwap)}</strong></span>
            )}
          </div>
        )}

        {/* TradingView-Style Horizontal Shaded S/R Zones SVG Overlay */}
        {showZones && visualZones.length > 0 && (
          <svg
            className="absolute inset-0 pointer-events-none w-full h-full z-[5]"
            style={{ overflow: 'hidden' }}
          >
            <defs>
              <linearGradient id="liveGreenZoneGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.14" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.05" />
              </linearGradient>
              <linearGradient id="liveRedZoneGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity="0.14" />
                <stop offset="100%" stopColor="#ef4444" stopOpacity="0.05" />
              </linearGradient>
            </defs>

            {visualZones.map((z, idx) => {
              const isGreen = z.type === 'support';
              const fill = isGreen ? 'url(#liveGreenZoneGrad)' : 'url(#liveRedZoneGrad)';
              const stroke = isGreen ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)';
              const plotWidth = Math.max(100, containerWidth - 62); // Leave 62px for right price scale

              return (
                <rect
                  key={`${z.type}-${idx}`}
                  x={2}
                  y={z.yTop}
                  width={plotWidth - 4}
                  height={z.height}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={1}
                  strokeDasharray="4 3"
                  rx={2}
                />
              );
            })}
          </svg>
        )}

        <div ref={chartContainerRef} className="w-full rounded overflow-hidden" />
      </div>
    </div>
  );
};
