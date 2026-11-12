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

export function formatChartTick(
  time: any,
  tickMarkType: number,
  tz: TimeZoneMode = 'ET'
): string {
  const epoch = typeof time === 'number' ? time : (time?.timestamp ?? Number(time));
  if (!epoch || isNaN(epoch)) return '';
  const date = new Date(epoch * 1000);
  const timeZone = tz === 'ET' ? 'America/New_York' : undefined;

  // TickMarkType: Year=0, Month=1, DayOfMonth=2, Time=3, TimeWithSeconds=4
  if (tickMarkType === 0) {
    return new Intl.DateTimeFormat('en-US', { timeZone, year: 'numeric' }).format(date);
  }
  if (tickMarkType === 1) {
    return new Intl.DateTimeFormat('en-US', { timeZone, month: 'short' }).format(date);
  }
  if (tickMarkType === 2) {
    return new Intl.DateTimeFormat('en-US', { timeZone, month: 'short', day: 'numeric' }).format(date);
  }
  return new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date);
}

export function formatChartTime(
  time: any,
  tz: TimeZoneMode = 'ET'
): string {
  const epoch = typeof time === 'number' ? time : (time?.timestamp ?? Number(time));
  if (!epoch || isNaN(epoch)) return '';
  const date = new Date(epoch * 1000);
  const timeZone = tz === 'ET' ? 'America/New_York' : undefined;
  return new Intl.DateTimeFormat('en-US', {
    timeZone,
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date);
}

