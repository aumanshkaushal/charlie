import React, { useState, useRef, useEffect, useCallback } from 'react';
import Head from 'next/head';
import SpectrogramCanvas from '@/components/SpectrogramCanvas';

const GENRES = ['random', 'lofi', 'rnb', 'house', 'trap', 'synthwave', 'afrobeat', 'ambient'];

interface StemData {
  audioUrl: string;
  spectrogram: number[][];
  duration?: number;
}

export default function Home() {
  const [fileName, setFileName] = useState<string>('');
  const [filePath, setFilePath] = useState<string>('');
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [analysis, setAnalysis] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  // Neural config
  const [musical, setMusical] = useState<boolean>(true);
  const [genre, setGenre] = useState<string>('random');
  const [seed, setSeed] = useState<number>(() => Math.floor(1000 + Math.random() * 900000));

  // Generation state
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [stems, setStems] = useState<Record<string, StemData>>({});
  const [masterUrl, setMasterUrl] = useState<string>('');

  // Audio Playback & Interactive Playhead State
  const [playingTrackId, setPlayingTrackId] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef<number | null>(null);

  // Update playhead smoothly using requestAnimationFrame
  const updatePlayhead = useCallback(() => {
    if (audioRef.current && !audioRef.current.paused) {
      setCurrentTime(audioRef.current.currentTime);
      rafRef.current = requestAnimationFrame(updatePlayhead);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (audioRef.current) audioRef.current.pause();
    };
  }, []);

  // Play / Pause / Seek Handler
  const playTrack = (trackId: string, url: string, seekTo?: number) => {
    if (!url) return;

    if (playingTrackId === trackId && seekTo === undefined) {
      // Toggle pause if already playing same track and no seek requested
      audioRef.current?.pause();
      setPlayingTrackId(null);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }

    if (audioRef.current) {
      audioRef.current.pause();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    }

    const audio = new Audio(url);
    audioRef.current = audio;
    setPlayingTrackId(trackId);

    audio.onloadedmetadata = () => {
      setAudioDuration(audio.duration);
      if (seekTo !== undefined) {
        audio.currentTime = Math.min(seekTo, audio.duration);
      }
      audio.play().catch(console.error);
      rafRef.current = requestAnimationFrame(updatePlayhead);
    };

    // If metadata already loaded
    if (audio.readyState >= 1) {
      setAudioDuration(audio.duration);
      if (seekTo !== undefined) {
        audio.currentTime = Math.min(seekTo, audio.duration);
      }
      audio.play().catch(console.error);
      rafRef.current = requestAnimationFrame(updatePlayhead);
    }

    audio.onended = () => {
      setPlayingTrackId(null);
      setCurrentTime(0);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  };

  const handleSeek = (trackId: string, url: string, seekTime: number) => {
    if (playingTrackId === trackId && audioRef.current) {
      audioRef.current.currentTime = seekTime;
      setCurrentTime(seekTime);
      if (audioRef.current.paused) {
        audioRef.current.play().catch(console.error);
        rafRef.current = requestAnimationFrame(updatePlayhead);
      }
    } else {
      playTrack(trackId, url, seekTime);
    }
  };

  // Upload handler
  const handleUpload = async (file: File) => {
    setFileName(file.name);
    setIsAnalyzing(true);
    setStems({});

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        setFilePath(data.filePath);
        setAudioUrl(data.audioUrl);
        setAnalysis(data.analysis);
      }
    } catch (e) {
      console.error('Upload error:', e);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Load preset with real STFT
  const loadPreset = async (name: string) => {
    setFileName(name);
    setFilePath(name);
    setAudioUrl(`/api/audio?file=${encodeURIComponent(name)}`);
    setIsAnalyzing(true);
    setStems({});

    try {
      const res = await fetch(`/api/analyze?file=${encodeURIComponent(name)}`);
      const data = await res.json();
      setAnalysis(data);
    } catch (e) {
      console.error('Analyze error:', e);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Generate composition
  const handleGenerate = async () => {
    if (!filePath) return;
    setIsGenerating(true);

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filePath,
          genre: genre === 'random' ? undefined : genre,
          musical,
          seed,
          bars: 8,
        }),
      });

      const data = await res.json();
      if (data.success) {
        setMasterUrl(data.masterUrl);
        setStems(data.stems || {});
      }
    } catch (e) {
      console.error('Generate error:', e);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#07070a] text-white p-6 md:p-10 flex flex-col font-serif selection:bg-cyan-500/30">
      <Head>
        <title>hey charlie</title>
        <meta name="description" content="Turn environmental audio into music" />
        <link rel="icon" href="data:," />
      </Head>

      {/* Header: only 'hey charlie' is funky handwriting, everything else is serif */}
      <header className="mb-6 flex items-baseline justify-between max-w-7xl mx-auto w-full">
        <h1 className="font-funky text-5xl md:text-6xl text-white select-none">
          hey charlie
        </h1>
        <div className="flex items-center gap-3 text-sm">
          <button
            onClick={() => loadPreset('mug-beat.mp3')}
            className="px-2.5 py-1 border border-white/20 rounded hover:border-white text-slate-300 transition cursor-pointer"
          >
            mug-beat.mp3
          </button>
          <button
            onClick={() => loadPreset('sample-2.mp3')}
            className="px-2.5 py-1 border border-white/20 rounded hover:border-white text-slate-300 transition cursor-pointer"
          >
            sample-2.mp3
          </button>
        </div>
      </header>

      {/* Main Workspace Frame (Matching Wireframe) */}
      <main className="flex-1 max-w-7xl mx-auto w-full flex items-center justify-center">
        {!filePath ? (
          /* STATE 1: Initial Empty Wireframe State */
          <div className="border border-white/30 rounded-2xl p-6 md:p-10 grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl bg-black/40">
            {/* Upload MP3 Box */}
            <label className="border border-white/40 rounded-xl p-8 flex items-center justify-center min-h-[320px] cursor-pointer hover:border-white transition group bg-white/[0.02]">
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
              />
              <span className="text-2xl font-serif text-white group-hover:scale-105 transition">
                Upload MP3
              </span>
            </label>

            {/* Empty Spectrogram Box with Real Ambient Spectrogram Visual */}
            <div className="border border-white/40 rounded-xl p-3 flex flex-col justify-between min-h-[320px] relative bg-black/50">
              <div className="flex-1 w-full rounded overflow-hidden">
                <SpectrogramCanvas
                  isEmpty
                  emptyText="Empty spectrogram"
                  height={260}
                />
              </div>
              <div className="flex justify-end p-2">
                <button
                  onClick={() => loadPreset('mug-beat.mp3')}
                  title="Try preset mug-beat.mp3"
                  className="w-10 h-10 rounded-full border border-white/50 flex items-center justify-center text-lg hover:border-white transition cursor-pointer"
                >
                  ▶
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* STATE 2: Active Neural Network Diagram (Exact Wireframe Layout) */
          <div className="border border-white/30 rounded-2xl p-6 md:p-8 w-full bg-black/40 relative">
            
            {/* SVG Synaptic Branching Connecting Lines */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none hidden lg:block z-0"
              xmlns="http://www.w3.org/2000/svg"
            >
              {/* Line from Input Box to Generate */}
              <path
                d="M 380 140 C 440 140, 460 210, 520 210"
                fill="none"
                stroke="rgba(255,255,255,0.35)"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />
              {/* Branch 1: Generate -> Original */}
              <path
                d="M 640 210 C 700 210, 720 70, 770 70"
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="1.5"
              />
              {/* Branch 2: Generate -> Melody */}
              <path
                d="M 640 210 C 700 210, 720 165, 770 165"
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="1.5"
              />
              {/* Branch 3: Generate -> Drums */}
              <path
                d="M 640 210 C 700 210, 720 260, 770 260"
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="1.5"
              />
              {/* Branch 4: Generate -> Harmony */}
              <path
                d="M 640 210 C 700 210, 720 355, 770 355"
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="1.5"
              />
            </svg>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start relative z-10">
              
              {/* LEFT COLUMN: Input Spectrogram + Name + Preview + Config (approx 4.5 cols) */}
              <div className="lg:col-span-5 flex flex-col gap-3">
                {/* Main Input Spectrogram Box */}
                <div className="border border-white/60 rounded-xl p-2 bg-black/60 relative">
                  <div className="h-[180px] w-full rounded overflow-hidden">
                    <SpectrogramCanvas
                      matrix={analysis?.spectrogram}
                      isEmpty={!analysis?.spectrogram}
                      duration={analysis?.duration || audioDuration || 3.5}
                      currentTime={playingTrackId === 'input' ? currentTime : 0}
                      isPlaying={playingTrackId === 'input'}
                      onSeek={(t) => handleSeek('input', audioUrl, t)}
                    />
                  </div>
                </div>

                {/* File Name directly below spectrogram */}
                <div className="text-center font-serif text-slate-300 text-base">
                  {fileName || 'audio.mp3'}
                </div>

                {/* Mini Preview Player Bar */}
                <div className="border border-white/40 rounded-lg p-2 flex items-center justify-between gap-3 bg-black/30">
                  <div className="flex-1 h-8 rounded overflow-hidden">
                    <SpectrogramCanvas
                      matrix={analysis?.spectrogram}
                      isEmpty={!analysis?.spectrogram}
                      isMini
                      height={32}
                      duration={analysis?.duration || audioDuration || 3.5}
                      currentTime={playingTrackId === 'input' ? currentTime : 0}
                      isPlaying={playingTrackId === 'input'}
                      onSeek={(t) => handleSeek('input', audioUrl, t)}
                    />
                  </div>
                  <button
                    onClick={() => playTrack('input', audioUrl)}
                    className="px-4 py-1 border border-white/40 rounded text-sm hover:border-white transition font-serif cursor-pointer"
                  >
                    {playingTrackId === 'input' ? 'pause' : 'play'}
                  </button>
                </div>

                {/* Config Label */}
                <div className="text-slate-400 text-sm italic font-serif mt-2">
                  config
                </div>

                {/* Blue Outline Config Box (Matching Wireframe) */}
                <div className="border-2 border-blue-500 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 bg-blue-950/10">
                  {/* [x] musical */}
                  <label className="flex items-center gap-2 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={musical}
                      onChange={(e) => setMusical(e.target.checked)}
                      className="w-4 h-4 rounded accent-blue-500 cursor-pointer"
                    />
                    <span className="font-serif text-sm">
                      [x] musical
                    </span>
                  </label>

                  {/* genre dropdown / selector with random */}
                  <div className="flex items-center gap-1.5 font-serif text-sm">
                    <span className="text-slate-400">genre:</span>
                    <select
                      value={genre}
                      onChange={(e) => setGenre(e.target.value)}
                      className="bg-black/70 border border-blue-400/50 rounded px-2 py-0.5 text-white text-sm focus:outline-none cursor-pointer"
                    >
                      {GENRES.map((g) => (
                        <option key={g} value={g}>
                          {g === 'random' ? '🎲 random' : g}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* seed box */}
                  <div className="flex items-center gap-1 border border-blue-400/50 rounded px-2 py-0.5 bg-black/70">
                    <span className="text-xs text-slate-400 font-serif">seed:</span>
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value) || 1)}
                      className="w-16 bg-transparent text-xs text-white text-right focus:outline-none"
                    />
                    <button
                      onClick={() => setSeed(Math.floor(1000 + Math.random() * 900000))}
                      title="Random seed"
                      className="text-xs hover:text-cyan-400 cursor-pointer"
                    >
                      🎲
                    </button>
                  </div>
                </div>
              </div>

              {/* CENTER COLUMN: Generate Button with Green Arrow (2.5 cols) */}
              <div className="lg:col-span-2 flex flex-col items-center justify-center my-auto py-12">
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className={`flex items-center gap-2 px-5 py-3 border border-white/60 rounded-xl text-lg font-serif transition cursor-pointer hover:border-emerald-400 hover:text-emerald-300 group bg-black/60 ${
                    isGenerating ? 'opacity-50 animate-pulse' : 'hover:scale-105'
                  }`}
                >
                  <span>{isGenerating ? 'generating' : 'generate'}</span>
                  {/* Green hand-drawn arrow style */}
                  <span className="text-2xl text-emerald-400 font-bold group-hover:translate-x-1 transition-transform">
                    ➔
                  </span>
                </button>
              </div>

              {/* RIGHT COLUMN: 4 Output Stem Cards (approx 4.5 cols) */}
              <div className="lg:col-span-5 flex flex-col gap-3">
                {[
                  { id: 'original', label: 'original' },
                  { id: 'melody', label: 'melody' },
                  { id: 'drums', label: 'drums' },
                  { id: stems['chords'] ? 'chords' : 'harmony', label: 'harmony' },
                ].map((item) => {
                  const stemKey = item.id === 'harmony' && stems['bass'] && !stems['chords'] ? 'bass' : item.id;
                  const stem = stems[stemKey];
                  const hasStem = Boolean(stem && stem.spectrogram && stem.spectrogram.length > 0);

                  return (
                    <div key={item.id} className="flex flex-col">
                      {/* Stem label above box */}
                      <span className="font-serif text-sm text-slate-300 ml-1 mb-0.5">
                        {item.label}
                      </span>

                      {/* Stem Box with Spectrogram + Play Button */}
                      <div className="border border-white/40 rounded-xl p-1.5 flex items-center justify-between gap-3 bg-black/50">
                        {/* Interactive Click-to-Seek Spectrogram */}
                        <div className="flex-1 h-14 rounded overflow-hidden bg-black/40">
                          {hasStem ? (
                            <SpectrogramCanvas
                              matrix={stem.spectrogram}
                              isMini
                              height={56}
                              duration={stem.duration || 14}
                              currentTime={playingTrackId === item.id ? currentTime : 0}
                              isPlaying={playingTrackId === item.id}
                              onSeek={(t) => handleSeek(item.id, stem.audioUrl, t)}
                            />
                          ) : (
                            <SpectrogramCanvas
                              isEmpty
                              emptyText=""
                              isMini
                              height={56}
                            />
                          )}
                        </div>

                        {/* Round Play Button */}
                        <button
                          disabled={!hasStem}
                          onClick={() => stem && playTrack(item.id, stem.audioUrl)}
                          className={`w-9 h-9 rounded-full border flex items-center justify-center text-sm transition cursor-pointer shrink-0 ${
                            hasStem
                              ? playingTrackId === item.id
                                ? 'border-white bg-white text-black font-bold'
                                : 'border-white/50 text-white hover:border-white'
                              : 'border-white/10 text-white/20 cursor-not-allowed'
                          }`}
                        >
                          {playingTrackId === item.id ? '⏸' : '▶'}
                        </button>
                      </div>
                    </div>
                  );
                })}

                {/* Master Mix Download Link */}
                {masterUrl && (
                  <div className="mt-2 text-right">
                    <button
                      onClick={() => playTrack('master', masterUrl)}
                      className="text-xs font-serif text-emerald-400 hover:underline mr-4 cursor-pointer"
                    >
                      {playingTrackId === 'master' ? '⏸ pause master' : '▶ play master'}
                    </button>
                    <a
                      href={masterUrl}
                      download="charlie_master.wav"
                      className="text-xs font-serif text-slate-400 hover:text-white underline"
                    >
                      download master mix
                    </a>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
