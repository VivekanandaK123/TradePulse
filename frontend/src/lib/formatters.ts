export type TimeZoneMode = 'ET' | 'LOCAL';

export function formatTimestamp(epochSeconds: number, tz: TimeZoneMode = 'ET'): string {
  if (!epochSeconds) return '--:--';
  const date = new Date(epochSeconds * 1000);

  if (tz === 'ET') {
    return new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    }).format(date) + ' ET';
  } else {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    }).format(date) + ' Local';
  }
}

export function formatDateWithTime(epochSeconds: number, tz: TimeZoneMode = 'ET'): string {
  if (!epochSeconds) return '--';
  const date = new Date(epochSeconds * 1000);

  return new Intl.DateTimeFormat('en-US', {
    timeZone: tz === 'ET' ? 'America/New_York' : undefined,
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date);
}

export function formatCountdown(seconds: number): string {
  if (seconds <= 0) return '00:00';
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hrs > 0) {
    return `${hrs}h ${mins}m`;
  }
  return `${mins}m ${secs < 10 ? '0' : ''}${secs}s`;
}

export function formatPrice(val: number): string {
  if (val === undefined || val === null || isNaN(val)) return '$--.--';
  return `$${val.toFixed(2)}`;
}

export function formatBps(val: number): string {
  if (val === undefined || val === null || isNaN(val)) return '-- bps';
  const sign = val > 0 ? '+' : '';
  return `${sign}${val.toFixed(2)} bps`;
}

export function formatPct(val: number): string {
  if (val === undefined || val === null || isNaN(val)) return '--%';
  return `${val.toFixed(1)}%`;
}

