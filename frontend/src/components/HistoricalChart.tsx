import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData, LineData, UTCTimestamp } from 'lightweight-charts';
import { CandleBar, PeriodBand, OOFSignalItem } from '../types/api';
import { TimeZoneMode, formatPrice, formatDateWithTime } from '../lib/formatters';
import { Layers, Eye, Info } from 'lucide-react';

interface HistoricalChartProps {
  bars: CandleBar[];
  periodBands: PeriodBand[];
  oofSignals?: OOFSignalItem[];
  selectedRange: string;
  onSelectRange: (r: string) => void;
  timeZone: TimeZoneMode;
  isLoading?: boolean;
}

export const HistoricalChart: React.FC<HistoricalChartProps> = ({
  bars,
  periodBands,
  oofSignals = [],
  selectedRange,
  onSelectRange,
  timeZone,
  isLoading = false,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const vwapSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  // Toggles
  const [showVwap, setShowVwap] = useState(false);
  const [showVolume, setShowVolume] = useState(true);
  const [showOofSignals, setShowOofSignals] = useState(false);
  const [hoveredCandle, setHoveredCandle] = useState<CandleBar | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Initialize Lightweight Chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 360,
      layout: {
        background: { color: '#0f172a' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      crosshair: {
        vertLine: { color: '#475569', width: 1, style: 2 },
        horzLine: { color: '#475569', width: 1, style: 2 },
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Candlestick Series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderUpColor: '#10b981',
      borderDownColor: '#ef4444',
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    // Volume Histogram Series (scaled down in secondary bottom pane)
    const volumeSeries = chart.addHistogramSeries({
      color: '#334155',
      priceFormat: { type: 'volume' },
      priceScaleId: '', // overlay
    });
    chart.priceScale('').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // VWAP Line Series
    const vwapSeries = chart.addLineSeries({
      color: '#f59e0b',
      lineWidth: 2,
      lineStyle: 1, // dotted
      visible: false,
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    vwapSeriesRef.current = vwapSeries;

    // Tooltip Crosshair move listener
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

    // Handle Resize
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

  // Update Data when bars change
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current || !vwapSeriesRef.current || bars.length === 0) return;

    const candleData: CandlestickData[] = [];
    const volumeData: HistogramData[] = [];
    const vwapData: LineData[] = [];

    // Sort strictly chronologically
    const sorted = [...bars].sort((a, b) => a.time - b.time);

    sorted.forEach((b) => {
      const isUp = b.close >= b.open;
      const isDimmed = b.is_filled;

      candleData.push({
        time: b.time as UTCTimestamp,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
        color: isDimmed ? (isUp ? '#064e3b' : '#7f1d1d') : undefined,
      });

      volumeData.push({
        time: b.time as UTCTimestamp,
        value: b.volume,
        color: isDimmed ? '#1e293b' : (isUp ? '#05966955' : '#dc262655'),
      });

      if (b.vwap) {
        vwapData.push({
          time: b.time as UTCTimestamp,
          value: b.vwap,
        });
      }
    });

    candleSeriesRef.current.setData(candleData);
    volumeSeriesRef.current.setData(volumeData);
    vwapSeriesRef.current.setData(vwapData);

    // Apply OOF Signal Markers if toggled
    if (showOofSignals && oofSignals.length > 0) {
      const markers = oofSignals.map((s) => ({
        time: s.time as UTCTimestamp,
        position: s.direction === 'Up' ? 'belowBar' as const : 'aboveBar' as const,
        color: s.direction === 'Up' ? '#10b981' : (s.direction === 'Down' ? '#ef4444' : '#64748b'),
        shape: s.direction === 'Up' ? 'arrowUp' as const : (s.direction === 'Down' ? 'arrowDown' as const : 'circle' as const),
        text: s.direction !== 'no clear signal' ? `${s.direction} ${s.confidence_pct.toFixed(0)}%` : '',
      }));
      candleSeriesRef.current.setMarkers(markers);
    } else {
      candleSeriesRef.current.setMarkers([]);
    }

    chartRef.current?.timeScale().fitContent();
  }, [bars, showOofSignals, oofSignals]);

  // Update VWAP visibility
  useEffect(() => {
    vwapSeriesRef.current?.applyOptions({ visible: showVwap });
  }, [showVwap]);

  // Update Volume visibility
  useEffect(() => {
    volumeSeriesRef.current?.applyOptions({ visible: showVolume });
  }, [showVolume]);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-col gap-2">
      {/* Top Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-xs tracking-wider text-slate-300 uppercase flex items-center gap-1.5">
            <Layers className="h-4 w-4 text-blue-400" />
            Historical Training Data
          </span>
          <span className="text-[11px] text-slate-500 font-mono">({bars.length.toLocaleString()} bars)</span>
        </div>

        {/* Range Selector */}
        <div className="flex items-center gap-1 bg-slate-950 p-0.5 rounded-md border border-slate-800">
          {['1D', '5D', '1M', '3M', 'All'].map((r) => (
            <button
              key={r}
              onClick={() => onSelectRange(r)}
              className={`px-2 py-0.5 text-xs font-medium rounded transition ${
                selectedRange === r
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              {r}
            </button>
          ))}
        </div>

        {/* Indicator Toggles */}
        <div className="flex items-center gap-2 text-xs">
          <label className="flex items-center gap-1 cursor-pointer text-slate-400 hover:text-slate-200">
            <input
              type="checkbox"
              checked={showVwap}
              onChange={(e) => setShowVwap(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-amber-500 focus:ring-0"
            />
            <span>VWAP</span>
          </label>

          <label className="flex items-center gap-1 cursor-pointer text-slate-400 hover:text-slate-200">
            <input
              type="checkbox"
              checked={showVolume}
              onChange={(e) => setShowVolume(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-blue-500 focus:ring-0"
            />
            <span>Volume</span>
          </label>

          <label className="flex items-center gap-1 cursor-pointer text-slate-400 hover:text-slate-200">
            <input
              type="checkbox"
              checked={showOofSignals}
              onChange={(e) => setShowOofSignals(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0"
            />
            <span>OOF Signals</span>
          </label>
        </div>
      </div>

      {/* Period Bands Legend & Notice */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] bg-slate-950/60 px-2.5 py-1 rounded border border-slate-800/60">
        <div className="flex items-center flex-wrap gap-3">
          <span className="text-slate-400 font-medium">Split Bands:</span>
          {periodBands.map((band) => (
            <div key={band.name} className="flex items-center gap-1.5" title={band.tooltip}>
              <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: band.color.replace('0.08', '0.6') }} />
              <span className="text-slate-300">{band.label.split(' ')[0]}</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5 text-slate-400">
            <span className="h-2 w-2 rounded-sm bg-slate-700" />
            <span>Dimmed: Gap-filled</span>
          </div>
        </div>

        <div className="flex items-center gap-1 text-slate-400" title="The final model was refit on all periods; bands show the earlier experiment split.">
          <Info className="h-3 w-3 text-slate-500" />
          <span>Final model refit on all data; bands show earlier split</span>
        </div>
      </div>

      {/* Chart Canvas Container */}
      <div className="relative">
        {isLoading && (
          <div className="absolute inset-0 bg-slate-950/60 flex items-center justify-center z-10 rounded">
            <div className="flex items-center gap-2 text-sm text-blue-400 font-medium">
              <div className="h-4 w-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              <span>Loading Historical Bars...</span>
            </div>
          </div>
        )}

        {/* Crosshair OHLCV Tooltip overlay */}
        {hoveredCandle && (
          <div className="absolute top-2 left-2 z-10 bg-slate-950/90 border border-slate-800 rounded px-2 py-1 text-xs font-mono flex items-center gap-3 text-slate-300 shadow">
            <span>{formatDateWithTime(hoveredCandle.time, timeZone)}</span>
            <span>O: <strong className="text-slate-100">{formatPrice(hoveredCandle.open)}</strong></span>
            <span>H: <strong className="text-emerald-400">{formatPrice(hoveredCandle.high)}</strong></span>
            <span>L: <strong className="text-rose-400">{formatPrice(hoveredCandle.low)}</strong></span>
            <span>C: <strong className="text-slate-100">{formatPrice(hoveredCandle.close)}</strong></span>
          </div>
        )}

        <div ref={chartContainerRef} className="w-full rounded overflow-hidden" />
      </div>

      {showOofSignals && (
        <div className="text-[11px] text-amber-400/90 bg-amber-950/30 border border-amber-800/40 px-2 py-1 rounded">
          * Notice: Past markers are out-of-fold historical cross-validation results, not live trading results.
        </div>
      )}
    </div>
  );
};
