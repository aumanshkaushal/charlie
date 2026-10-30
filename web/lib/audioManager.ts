// Web Audio API Multi-Track Stem Engine with Sample-Accurate Sync
export interface StemTrack {
  id: string;
  url: string;
  buffer?: AudioBuffer;
  duration: number;
  gainNode?: GainNode;
  enabled: boolean; // on/off in master mix
}

export class CharlieAudioManager {
  private ctx: AudioContext | null = null;
  private buffers = new Map<string, AudioBuffer>();
  private activeSources: AudioBufferSourceNode[] = [];
  private stemGains = new Map<string, GainNode>();
  private masterGain: GainNode | null = null;

  public activeTrackId: string | null = null; // 'input' | 'master' | stemId
  public isPlaying = false;
  private playbackStartTime = 0;
  private startOffset = 0;
  private onTimeUpdateCallback?: (currentTime: number, duration: number) => void;
  private onEndCallback?: () => void;
  private rafId: number | null = null;

  constructor() {
    // Lazy initialized on first user gesture
  }

  private initContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      this.ctx = new AudioCtx();
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    return this.ctx;
  }

  public async loadBuffer(id: string, url: string): Promise<AudioBuffer> {
    const cached = this.buffers.get(id);
    if (cached) return cached;

    const ctx = this.initContext();
    const res = await fetch(url);
    const arrayBuf = await res.arrayBuffer();
    const audioBuf = await ctx.decodeAudioData(arrayBuf);
    this.buffers.set(id, audioBuf);
    return audioBuf;
  }

  public getDuration(id: string): number {
    return this.buffers.get(id)?.duration || 0;
  }

  public getCurrentTime(): number {
    if (!this.isPlaying || !this.ctx) return this.startOffset;
    const elapsed = this.ctx.currentTime - this.playbackStartTime;
    return Math.max(0, this.startOffset + elapsed);
  }

  public stop() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    for (const src of this.activeSources) {
      try {
        src.stop();
        src.disconnect();
      } catch {}
    }
    this.activeSources = [];
    this.isPlaying = false;
    this.startOffset = 0;
    this.activeTrackId = null;
  }

  public pause() {
    if (!this.isPlaying) return;
    this.startOffset = this.getCurrentTime();
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    for (const src of this.activeSources) {
      try {
        src.stop();
        src.disconnect();
      } catch {}
    }
    this.activeSources = [];
    this.isPlaying = false;
  }

  // Play a single track (e.g. input sound or solo stem)
  public async playSingle(
    id: string,
    url: string,
    seekTime = 0,
    onTime?: (t: number, d: number) => void,
    onEnd?: () => void
  ) {
    const ctx = this.initContext();
    this.stop();

    const buffer = await this.loadBuffer(id, url);
    const duration = buffer.duration;
    this.activeTrackId = id;
    this.startOffset = Math.max(0, Math.min(seekTime, duration));
    this.onTimeUpdateCallback = onTime;
    this.onEndCallback = onEnd;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    this.activeSources = [source];

    this.playbackStartTime = ctx.currentTime;
    source.start(0, this.startOffset);
    this.isPlaying = true;

    source.onended = () => {
      if (this.activeSources.includes(source)) {
        this.stop();
        onEnd?.();
      }
    };

    this.startSyncLoop(duration);
  }

  // Play Master with live on/off stem mix
  public async playMasterMix(
    stems: Record<string, { audioUrl: string }>,
    stemMutes: Record<string, boolean>, // key -> isEnabled (true = on, false = off)
    seekTime = 0,
    fallbackMasterUrl?: string,
    onTime?: (t: number, d: number) => void,
    onEnd?: () => void
  ) {
    const ctx = this.initContext();
    this.stop();

    const stemKeys = Object.keys(stems);
    if (stemKeys.length === 0 && fallbackMasterUrl) {
      return this.playSingle('master', fallbackMasterUrl, seekTime, onTime, onEnd);
    }

    // Load all stem buffers in parallel
    const loaded = await Promise.all(
      stemKeys.map(async (k) => ({
        key: k,
        buf: await this.loadBuffer(k, stems[k].audioUrl),
      }))
    );

    const maxDuration = Math.max(...loaded.map((item) => item.buf.duration), 1);
    this.activeTrackId = 'master';
    this.startOffset = Math.max(0, Math.min(seekTime, maxDuration));
    this.onTimeUpdateCallback = onTime;
    this.onEndCallback = onEnd;

    this.masterGain = ctx.createGain();
    this.masterGain.connect(ctx.destination);

    this.activeSources = [];
    this.stemGains.clear();

    for (const item of loaded) {
      const src = ctx.createBufferSource();
      src.buffer = item.buf;

      const gain = ctx.createGain();
      const isEnabled = stemMutes[item.key] !== false; // default true
      gain.gain.value = isEnabled ? 1 : 0;

      src.connect(gain);
      gain.connect(this.masterGain);

      this.stemGains.set(item.key, gain);
      this.activeSources.push(src);

      src.start(0, this.startOffset);
    }

    this.playbackStartTime = ctx.currentTime;
    this.isPlaying = true;

    // Track end using first source
    if (this.activeSources[0]) {
      this.activeSources[0].onended = () => {
        this.stop();
        onEnd?.();
      };
    }

    this.startSyncLoop(maxDuration);
  }

  // Live toggle stem on/off without restarting audio
  public setStemEnabled(stemKey: string, isEnabled: boolean) {
    const gain = this.stemGains.get(stemKey);
    if (gain && this.ctx) {
      gain.gain.setTargetAtTime(isEnabled ? 1 : 0, this.ctx.currentTime, 0.015);
    }
  }

  private startSyncLoop(duration: number) {
    const update = () => {
      if (!this.isPlaying) return;
      const t = this.getCurrentTime();
      if (t >= duration) {
        this.stop();
        this.onEndCallback?.();
        return;
      }
      this.onTimeUpdateCallback?.(t, duration);
      this.rafId = requestAnimationFrame(update);
    };
    this.rafId = requestAnimationFrame(update);
  }
}
