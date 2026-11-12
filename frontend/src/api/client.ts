import {
  HealthResponse,
  MetaResponse,
  HistoryResponse,
  OOFSignalsResponse,
  TestNowResponse,
  SignalsLogResponse,
  OutcomesResponse,
  ReplayState,
} from '../types/api';

const BASE_URL = '/api';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchMeta(): Promise<MetaResponse> {
  const res = await fetch(`${BASE_URL}/meta`);
  if (!res.ok) throw new Error(`Fetch metadata failed: ${res.statusText}`);
  return res.json();
}

export async function fetchHistory(range: string = '1M'): Promise<HistoryResponse> {
  const res = await fetch(`${BASE_URL}/history?range=${range}`);
  if (!res.ok) throw new Error(`Fetch history failed: ${res.statusText}`);
  return res.json();
}

export async function fetchOOFSignals(): Promise<OOFSignalsResponse> {
  const res = await fetch(`${BASE_URL}/history/oof-signals`);
  if (!res.ok) throw new Error(`Fetch OOF signals failed: ${res.statusText}`);
  return res.json();
}

export async function callTestNow(): Promise<TestNowResponse> {
  const res = await fetch(`${BASE_URL}/live/test-now`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) throw new Error(`Test Now failed: ${res.statusText}`);
  return res.json();
}

export async function clearLiveSession(): Promise<{ status: string; live_since_epoch: number }> {
  const res = await fetch(`${BASE_URL}/live/clear`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) throw new Error(`Clear live session failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSignalsLog(session?: string): Promise<SignalsLogResponse> {
  const url = session ? `${BASE_URL}/signals?session=${encodeURIComponent(session)}` : `${BASE_URL}/signals`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch signals log failed: ${res.statusText}`);
  return res.json();
}

export async function fetchOutcomes(mode?: string): Promise<OutcomesResponse> {
  const url = mode ? `${BASE_URL}/outcomes?mode=${encodeURIComponent(mode)}` : `${BASE_URL}/outcomes`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch outcomes failed: ${res.statusText}`);
  return res.json();
}

export async function startReplay(sessionDate: string = '2026-11-06', speed: number = 1.0): Promise<ReplayState> {
  const res = await fetch(`${BASE_URL}/replay/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_date: sessionDate, speed_multiplier: speed }),
  });
  if (!res.ok) throw new Error(`Start replay failed: ${res.statusText}`);
  return res.json();
}

export async function stepReplay(): Promise<ReplayState> {
  const res = await fetch(`${BASE_URL}/replay/step`, { method: 'POST' });
  if (!res.ok) throw new Error(`Step replay failed: ${res.statusText}`);
  return res.json();
}

export async function pauseReplay(): Promise<ReplayState> {
  const res = await fetch(`${BASE_URL}/replay/pause`, { method: 'POST' });
  if (!res.ok) throw new Error(`Pause replay failed: ${res.statusText}`);
  return res.json();
}

export async function resumeReplay(): Promise<ReplayState> {
  const res = await fetch(`${BASE_URL}/replay/resume`, { method: 'POST' });
  if (!res.ok) throw new Error(`Resume replay failed: ${res.statusText}`);
  return res.json();
}

export async function setReplaySpeed(speed: number): Promise<ReplayState> {
  const res = await fetch(`${BASE_URL}/replay/speed?speed_multiplier=${encodeURIComponent(speed)}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Set replay speed failed: ${res.statusText}`);
  return res.json();
}

export async function stopReplay(): Promise<ReplayState> {
  const res = await fetch(`${BASE_URL}/replay/stop`, { method: 'POST' });
  if (!res.ok) throw new Error(`Stop replay failed: ${res.statusText}`);
  return res.json();
}

export async function fetchReplayState(): Promise<ReplayState> {
  const res = await fetch(`${BASE_URL}/replay/state`);
  if (!res.ok) throw new Error(`Fetch replay state failed: ${res.statusText}`);
  return res.json();
}

export async function fetchReplaySessions(): Promise<string[]> {
  try {
    const res = await fetch(`${BASE_URL}/replay/sessions`);
    if (!res.ok) return ['2026-11-06', '2026-11-05', '2026-11-04', '2026-11-03', '2026-10-31'];
    const data = await res.json();
    return data.sessions || ['2026-11-06', '2026-11-05', '2026-11-04', '2026-11-03', '2026-10-31'];
  } catch {
    return ['2026-11-06', '2026-11-05', '2026-11-04', '2026-11-03', '2026-10-31'];
  }
}


