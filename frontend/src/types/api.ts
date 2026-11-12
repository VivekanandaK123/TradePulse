export interface MarketStatus {
  is_open: boolean;
  session_state: 'OPEN' | 'CLOSED' | 'EARLY_CLOSE' | 'PRE_MARKET' | 'AFTER_HOURS';
  current_time_utc: number;
  next_open_utc: number;
  next_close_utc: number;
  countdown_seconds: number;
  description: string;
}

export interface HealthResponse {
  status: 'HEALTHY' | 'DEGRADED' | 'OFFLINE';
  uptime_seconds: number;
  feed_connectivity: 'CONNECTED' | 'REPLAY_MODE' | 'DISCONNECTED';
  feed_name: string;
  active_symbol: string;
  market_status: MarketStatus;
  artifact_hashes: Record<string, string>;
  model_version: string;
  last_data_freshness_seconds?: number;
}

export interface SplitBoundary {
  start_date?: string;
  end_date?: string;
  date?: string;
  sessions: number;
  valid_bars?: number;
}

export interface PeriodBand {
  name: string;
  label: string;
  start_time: number;
  end_time: number;
  color: string;
  tooltip: string;
}

export interface CandleBar {
  time: number; // UTC epoch seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number;
  is_filled?: boolean;
}

export interface HistoryResponse {
  symbol: string;
  range: string;
  bars_count: number;
  bars: CandleBar[];
  period_bands: PeriodBand[];
  prev_close?: number;
  etag: string;
}

export interface OOFSignalItem {
  time: number;
  direction: 'Up' | 'Down' | 'no clear signal';
  signal_code: number;
  confidence_pct: number;
  conviction: number;
  is_gated: boolean;
}

export interface OOFSignalsResponse {
  available: boolean;
  count: number;
  signals: OOFSignalItem[];
  note: string;
}

export interface ReliabilityFlag {
  id: string;
  name: string;
  level: 'NORMAL' | 'ADVISORY' | 'WARNING';
  message: string;
  guide_anchor: string;
}

export interface ProbabilityDistribution {
  down: number;
  neutral: number;
  up: number;
}

export interface PredictionSignal {
  bar_time: number;
  bar_timestamp_ny: string;
  close_price: number;
  direction: 'Up' | 'Down' | 'no clear signal';
  signal_code: number; // 1, -1, 0
  confidence_pct: number;
  conviction: number;
  threshold: number;
  is_gated: boolean;
  probabilities: ProbabilityDistribution;
  locked_at_utc: number;
  horizon_end_utc: number;
  model_version: string;
  reliability_flags: ReliabilityFlag[];
  outcome?: {
    realized_return_bps: number;
    directional_return_bps: number;
    net_return_bps_8bp: number;
    is_hit: boolean;
  };
}

export interface TestNowResponse {
  status: 'SUCCESS' | 'ALREADY_STORED' | 'MARKET_CLOSED';
  signal: PredictionSignal;
  latest_closed_bar: CandleBar;
  forming_bar?: CandleBar;
  recent_bars: CandleBar[];
  market_status: MarketStatus;
}

export interface SignalLogItem {
  signal_id: string;
  bar_time: number;
  bar_timestamp_ny: string;
  direction: 'Up' | 'Down' | 'no clear signal';
  signal_code: number;
  confidence_pct: number;
  conviction: number;
  entry_price: number;
  is_gated: boolean;
  checksum: string;
  outcome?: {
    exit_price: number;
    realized_return_bps: number;
    net_return_bps_8bp: number;
    is_hit: boolean;
    actual_direction?: string;
  };
}

export interface SignalsLogResponse {
  session: string;
  count: number;
  signals: SignalLogItem[];
}

export interface OutcomeItem {
  signal_id: string;
  signal_time: number;
  resolution_time: number;
  direction: string;
  signal_code: number;
  confidence_pct: number;
  entry_price: number;
  exit_price: number;
  realized_return_bps: number;
  directional_return_bps: number;
  net_return_bps_8bp: number;
  is_hit: boolean;
}

export interface OutcomesResponse {
  count: number;
  outcomes: OutcomeItem[];
}

export interface MetaResponse {
  asset: string;
  exchange: string;
  timeframe: string;
  label_name: string;
  theta_bps: number;
  theta_pct: number;
  horizon_bars: number;
  horizon_minutes: number;
  classes: Record<string, number>;
  feature_count: number;
  features_summary: string;
  model_architecture: string;
  probability_calibration: string;
  recency_weighting: {
    half_life: string;
    decay_rate_alpha: number;
    rationale: string;
  };
  signal_rule: {
    conviction_threshold_tau: number;
    formula: string;
    expected_signals_per_day: number;
    expected_coverage_pct: number;
    expected_zero_signal_days_pct: number;
    direction_logic: string;
  };
  dataset_scope: {
    training_date_range: {
      start_date: string;
      end_date: string;
      total_sessions: number;
      total_raw_bars: number;
      valid_training_rows: number;
      dropped_eod_bars: number;
    };
    experiment_splits: {
      train: SplitBoundary;
      embargo_1: SplitBoundary;
      validation: SplitBoundary;
      embargo_2: SplitBoundary;
      test: SplitBoundary;
      split_note: string;
    };
  };
  headline_metrics: {
    validation: {
      dataset: string;
      accuracy: number;
      balanced_accuracy: number;
      macro_f1: number;
      binary_sign_accuracy: number;
      updown_auc: number;
      gross_return_bps: number;
      net_return_8bp_bps: number;
    };
    held_out_test: {
      dataset: string;
      signals_emitted: number;
      signals_per_day: number;
      actual_coverage_pct: number;
      zero_signal_days_pct: number;
      accuracy: number;
      balanced_accuracy: number;
      balanced_accuracy_95ci: [number, number];
      macro_f1: number;
      macro_f1_95ci: [number, number];
      binary_sign_accuracy: number;
      binary_sign_accuracy_95ci: [number, number];
      updown_auc: number;
      gross_return_bps: number;
      gross_return_95ci: [number, number];
      net_return_5bp_bps: number;
      net_return_8bp_bps: number;
      net_return_8bp_95ci: [number, number];
      net_return_12bp_bps: number;
    };
  };
  cost_assumption: {
    institutional_friction_bps: number;
    conservative_friction_bps: number;
    low_friction_bps: number;
    convention: string;
  };
  edge_demonstrated: boolean;
  edge_statement: string;
  disclaimer: string;
  market_status: MarketStatus;
}

export interface ReplayState {
  is_active: boolean;
  status: 'STOPPED' | 'PLAYING' | 'PAUSED';
  session_date: string;
  current_bar_index: number;
  total_bars: number;
  speed_multiplier: number;
  current_bar?: CandleBar;
  current_signal?: PredictionSignal;
  signals_emitted_count: number;
}
