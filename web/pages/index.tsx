import React, { useState, useRef, useEffect, useCallback } from 'react';
import Head from 'next/head';
import { FaDice, FaCheck, FaPlay, FaPause, FaDownload } from 'react-icons/fa6';
import SpectrogramCanvas from '@/components/SpectrogramCanvas';
import { CharlieAudioManager } from '@/lib/audioManager';

const GENRES = ['random', 'lofi', 'rnb', 'house', 'trap', 'synthwave', 'afrobeat', 'ambient'];

interface StemData {
  audioUrl: string;
  spectrogram: number[][];
  duration: number;
}

export default function Home() {
  const [fileName, setFileName] = useState<string>('');
  const [filePath, setFilePath] = useState<string>('');
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [analysis, setAnalysis] = useState<any>(null);

  // Conditioning config
  const [musical, setMusical] = useState<boolean>(true);
  const [genre, setGenre] = useState<string>('random');
  const [seed, setSeed] = useState<number>(() => Math.floor(1000 + Math.random() * 900000));

  // Generation state
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [stems, setStems] = useState<Record<string, StemData>>({});
  const [masterUrl, setMasterUrl] = useState<string>('');
  const [masterSpectrogram, setMasterSpectrogram] = useState<number[][]>([]);
  const [masterDuration, setMasterDuration] = useState<number>(14);

  // Stems on/off live toggles in master mix (default: all ON)
  const [stemMutes, setStemMutes] = useState<Record<string, boolean>>({
    original: true,
    melody: true,
    drums: true,
    harmony: true,
    chords: true,
    bass: true,
  });

  // Web Audio API Engine
  const audioManagerRef = useRef<CharlieAudioManager | null>(null);
  const [activeTrackId, setActiveTrackId] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [activeDuration, setActiveDuration] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // Initialize AudioManager once
  useEffect(() => {
    audioManagerRef.current = new CharlieAudioManager();
    return () => {
      audioManagerRef.current?.stop();
    };
  }, []);

  // Sync callbacks
  const handleTimeUpdate = useCallback((t: number, d: number) => {
    setCurrentTime(t);
    setActiveDuration(d);
  }, []);

  const handlePlaybackEnd = useCallback(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setActiveTrackId(null);
  }, []);

  // Play / Pause / Seek Handler
  const handlePlayOrSeek = async (trackId: string, url: string, knownDur: number, seekTime?: number) => {
    const mgr = audioManagerRef.current;
    if (!mgr || !url) return;

    // Toggle Pause if same track clicked without seek
    if (activeTrackId === trackId && seekTime === undefined) {
      if (isPlaying) {
        mgr.pause();
        setIsPlaying(false);
      } else {
        // Resume from current paused time
        if (trackId === 'master') {
          await mgr.playMasterMix(stems, stemMutes, currentTime, masterUrl, handleTimeUpdate, handlePlaybackEnd);
        } else {
          await mgr.playSingle(trackId, url, currentTime, handleTimeUpdate, handlePlaybackEnd);
        }
        setIsPlaying(true);
      }
      return;
    }

    // Play or seek
    const targetSeek = seekTime !== undefined ? seekTime : 0;
    setActiveTrackId(trackId);
    setCurrentTime(targetSeek);
    setActiveDuration(knownDur);
    setIsPlaying(true);

    if (trackId === 'master') {
      await mgr.playMasterMix(stems, stemMutes, targetSeek, masterUrl, handleTimeUpdate, handlePlaybackEnd);
    } else {
      await mgr.playSingle(trackId, url, targetSeek, handleTimeUpdate, handlePlaybackEnd);
    }
  };

  // Toggle Stem ON / OFF in Master Mix
  const toggleStemMute = (stemKey: string) => {
    setStemMutes((prev) => {
      const nextState = !prev[stemKey];
      const updated = { ...prev, [stemKey]: nextState };
      // Live update Web Audio gain without interrupting playback!
      audioManagerRef.current?.setStemEnabled(stemKey, nextState);
      return updated;
    });
  };

  // Upload sound file
  const handleUpload = async (file: File) => {
    audioManagerRef.current?.stop();
    setIsPlaying(false);
    setActiveTrackId(null);
    setCurrentTime(0);

    setFileName(file.name);
    setStems({});
    setMasterUrl('');
    setMasterSpectrogram([]);

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
    }
  };

  // Load preset audio
  const loadPreset = async (name: string) => {
    audioManagerRef.current?.stop();
    setIsPlaying(false);
    setActiveTrackId(null);
    setCurrentTime(0);

    setFileName(name);
    setFilePath(name);
    setAudioUrl(`/api/audio?file=${encodeURIComponent(name)}`);
    setStems({});
    setMasterUrl('');
    setMasterSpectrogram([]);

    try {
      const res = await fetch(`/api/analyze?file=${encodeURIComponent(name)}`);
      const data = await res.json();
      setAnalysis(data);
    } catch (e) {
      console.error('Analyze error:', e);
    }
  };

  // Generate multi-stem & master composition
  const handleGenerate = async () => {
    if (!filePath) return;
    audioManagerRef.current?.stop();
    setIsPlaying(false);
    setActiveTrackId(null);
    setCurrentTime(0);
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
        setMasterSpectrogram(data.masterSpectrogram || []);
        setMasterDuration(data.masterDuration || 14);
        setStems(data.stems || {});
      }
    } catch (e) {
      console.error('Generate error:', e);
    } finally {
      setIsGenerating(false);
    }
  };

  const inputDur = analysis?.duration || 3.5;

  return (
    <div className="min-h-screen bg-[#060609] text-white p-4 md:p-8 flex flex-col font-sans selection:bg-cyan-500/30">
      <Head>
        <title>hey charlie</title>
        <meta name="description" content="Neural audio architecture" />
        <link rel="icon" href="data:," />
      </Head>

      {/* Header: strictly only 'hey charlie' has funky handwriting, everything else is sans-serif */}
      <header className="mb-4 flex items-baseline justify-between max-w-7xl mx-auto w-full">
        <h1 className="font-funky text-5xl md:text-6xl text-white select-none">
          hey charlie
        </h1>
        <div className="flex items-center gap-2 text-xs font-medium">
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

      {/* Main Neural Workspace */}
      <main className="flex-1 max-w-7xl mx-auto w-full flex items-center justify-center">
        {!filePath ? (
          /* STATE 1: Empty Wireframe State */
          <div className="border border-white/30 rounded-2xl p-6 md:p-10 grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl bg-black/40">
            {/* Upload MP3 Box */}
            <label className="border border-white/40 rounded-xl p-8 flex items-center justify-center min-h-[300px] cursor-pointer hover:border-white transition group bg-white/[0.02]">
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
              />
              <span className="text-xl font-medium text-white group-hover:scale-105 transition">
                Upload MP3
              </span>
            </label>

            {/* Empty Spectrogram Box with Real Ambient Spectrogram Visual */}
            <div className="border border-white/40 rounded-xl p-3 flex flex-col justify-between min-h-[300px] relative bg-black/50">
              <div className="flex-1 w-full rounded overflow-hidden">
                <SpectrogramCanvas
                  isEmpty
                  emptyText="Empty spectrogram"
                  height={240}
                />
              </div>
              <div className="flex justify-end p-2">
                <button
                  onClick={() => loadPreset('mug-beat.mp3')}
                  title="Try preset mug-beat.mp3"
                  className="w-10 h-10 rounded-full border border-white/50 flex items-center justify-center text-sm hover:border-white transition cursor-pointer"
                >
                  <FaPlay className="ml-0.5" />
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* STATE 2: Active Neural Network Pipeline (Input -> 4 Stems -> Converging to Master) */
          <div className="border border-white/30 rounded-2xl p-4 md:p-6 w-full bg-black/40 relative">
            
            {/* SVG Synaptic Neural Paths: Input -> Generate -> 4 Stems -> Master */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none hidden xl:block z-0"
              xmlns="http://www.w3.org/2000/svg"
            >
              {/* Line 1: Input Spectrogram -> Generate Button */}
              <path
                d="M 330 130 C 370 130, 390 220, 430 220"
                fill="none"
                stroke="rgba(255,255,255,0.3)"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />

              {/* Fan-out: Generate -> 4 Stems */}
              <path
                d="M 525 220 C 560 220, 570 70, 605 70"
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="1.5"
              />
              <path
                d="M 525 220 C 560 220, 570 155, 605 155"
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="1.5"
              />
              <path
                d="M 525 220 C 560 220, 570 240, 605 240"
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="1.5"
              />
              <path
                d="M 525 220 C 560 220, 570 325, 605 325"
                fill="none"
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="1.5"
              />

              {/* Fan-in: 4 Stems Converging Back into 1 Master Node */}
              <path
                d="M 875 70 C 915 70, 925 200, 960 200"
                fill="none"
                stroke="rgba(16, 185, 129, 0.45)"
                strokeWidth="1.5"
              />
              <path
                d="M 875 155 C 915 155, 925 200, 960 200"
                fill="none"
                stroke="rgba(16, 185, 129, 0.45)"
                strokeWidth="1.5"
              />
              <path
                d="M 875 240 C 915 240, 925 200, 960 200"
                fill="none"
                stroke="rgba(16, 185, 129, 0.45)"
                strokeWidth="1.5"
              />
              <path
                d="M 875 325 C 915 325, 925 200, 960 200"
                fill="none"
                stroke="rgba(16, 185, 129, 0.45)"
                strokeWidth="1.5"
              />
            </svg>

            {/* Grid Layout: Input + Config (4 cols) | Generate (1.5 cols) | 4 Stems (3.5 cols) | Master (3 cols) */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start relative z-10">
              
              {/* COLUMN 1: Input Spectrogram + Preview + Config */}
              <div className="xl:col-span-4 flex flex-col gap-2.5">
                {/* Main Input Spectrogram Card */}
                <div className="border border-white/60 rounded-xl p-2 bg-black/60 relative">
                  <div className="h-[170px] w-full rounded overflow-hidden">
                    <SpectrogramCanvas
                      matrix={analysis?.spectrogram}
                      isEmpty={!analysis?.spectrogram}
                      duration={inputDur}
                      currentTime={activeTrackId === 'input' ? currentTime : 0}
                      isPlaying={activeTrackId === 'input' && isPlaying}
                      onSeek={(t) => handlePlayOrSeek('input', audioUrl, inputDur, t)}
                    />
                  </div>
                </div>

                {/* File Name directly below spectrogram */}
                <div className="text-center text-slate-300 text-xs font-medium truncate px-2">
                  {fileName || 'audio.mp3'}
                </div>

                {/* Mini Preview Player Bar with Download Button */}
                <div className="border border-white/40 rounded-lg p-1.5 flex items-center justify-between gap-2 bg-black/40">
                  <div className="flex-1 h-7 rounded overflow-hidden">
                    <SpectrogramCanvas
                      matrix={analysis?.spectrogram}
                      isEmpty={!analysis?.spectrogram}
                      isMini
                      height={28}
                      duration={inputDur}
                      currentTime={activeTrackId === 'input' ? currentTime : 0}
                      isPlaying={activeTrackId === 'input' && isPlaying}
                      onSeek={(t) => handlePlayOrSeek('input', audioUrl, inputDur, t)}
                    />
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button
                      onClick={() => handlePlayOrSeek('input', audioUrl, inputDur)}
                      className="px-2.5 py-0.5 border border-white/40 rounded text-xs hover:border-white transition cursor-pointer flex items-center gap-1"
                    >
                      {activeTrackId === 'input' && isPlaying ? <FaPause className="text-[10px]" /> : <FaPlay className="text-[10px]" />}
                      <span>{activeTrackId === 'input' && isPlaying ? 'pause' : 'play'}</span>
                    </button>
                    {audioUrl && (
                      <a
                        href={audioUrl}
                        download={fileName || 'input_audio.mp3'}
                        title="Download input sound"
                        className="p-1 border border-white/30 rounded hover:border-white text-slate-300 transition cursor-pointer flex items-center justify-center"
                      >
                        <FaDownload className="text-[10px]" />
                      </a>
                    )}
                  </div>
                </div>

                {/* Config Label */}
                <div className="text-slate-400 text-xs mt-1">
                  config
                </div>

                {/* Blue Outline Config Box */}
                <div className="border-2 border-blue-500 rounded-xl p-2.5 flex flex-wrap items-center justify-between gap-2 bg-blue-950/10">
                  {/* musical checkmark checkbox */}
                  <button
                    onClick={() => setMusical(!musical)}
                    className="flex items-center gap-2 cursor-pointer select-none group"
                  >
                    <div
                      className={`w-4 h-4 rounded border flex items-center justify-center transition ${
                        musical ? 'bg-blue-600 border-blue-400 text-white' : 'border-slate-500 bg-transparent'
                      }`}
                    >
                      {musical && <FaCheck className="text-[10px]" />}
                    </div>
                    <span className="text-xs text-slate-200 group-hover:text-white font-medium">
                      musical
                    </span>
                  </button>

                  {/* genre dropdown with react-icons dice */}
                  <div className="flex items-center gap-1.5 text-xs">
                    <span className="text-slate-400 text-xs">genre:</span>
                    <div className="relative flex items-center">
                      <select
                        value={genre}
                        onChange={(e) => setGenre(e.target.value)}
                        className="bg-black/80 border border-blue-400/50 rounded px-2 py-0.5 text-white text-xs focus:outline-none cursor-pointer"
                      >
                        {GENRES.map((g) => (
                          <option key={g} value={g}>
                            {g === 'random' ? 'random' : g}
                          </option>
                        ))}
                      </select>
                      {genre === 'random' && (
                        <FaDice className="ml-1 text-blue-400 text-xs pointer-events-none" />
                      )}
                    </div>
                  </div>

                  {/* seed box with react-icons dice */}
                  <div className="flex items-center gap-1 border border-blue-400/50 rounded px-1.5 py-0.5 bg-black/80">
                    <span className="text-[11px] text-slate-400">seed:</span>
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value) || 1)}
                      className="w-14 bg-transparent text-xs text-white text-right focus:outline-none"
                    />
                    <button
                      onClick={() => setSeed(Math.floor(1000 + Math.random() * 900000))}
                      title="Randomize seed"
                      className="text-slate-300 hover:text-cyan-300 transition cursor-pointer p-0.5"
                    >
                      <FaDice className="text-xs" />
                    </button>
                  </div>
                </div>
              </div>

              {/* COLUMN 2: Generate Button with Green Arrow */}
              <div className="xl:col-span-1 flex flex-col items-center justify-center my-auto py-8">
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className={`flex items-center gap-1.5 px-4 py-2.5 border border-white/60 rounded-xl text-sm font-medium transition cursor-pointer hover:border-emerald-400 hover:text-emerald-300 group bg-black/60 ${
                    isGenerating ? 'opacity-50 animate-pulse' : 'hover:scale-105'
                  }`}
                >
                  <span>{isGenerating ? 'generating' : 'generate'}</span>
                  <span className="text-xl text-emerald-400 font-bold group-hover:translate-x-1 transition-transform">
                    ➔
                  </span>
                </button>
              </div>

              {/* COLUMN 3: 4 Output Stems with On/Off Switch & Download Buttons */}
              <div className="xl:col-span-4 flex flex-col gap-2.5">
                {[
                  { id: 'original', label: 'original' },
                  { id: 'melody', label: 'melody' },
                  { id: 'drums', label: 'drums' },
                  { id: stems['chords'] ? 'chords' : 'harmony', label: 'harmony' },
                ].map((item) => {
                  const stemKey = item.id === 'harmony' && stems['bass'] && !stems['chords'] ? 'bass' : item.id;
                  const stem = stems[stemKey];
                  const hasStem = Boolean(stem && stem.spectrogram && stem.spectrogram.length > 0);
                  const stemDur = stem?.duration || 14;
                  const isTrackActive = activeTrackId === item.id;
                  const isEnabledInMaster = stemMutes[stemKey] !== false;

                  return (
                    <div key={item.id} className="flex flex-col">
                      {/* Top Bar above stem: Label + On/Off Switch in Master */}
                      <div className="flex items-center justify-between px-1 mb-0.5 text-xs">
                        <span className="text-slate-300 font-medium">
                          {item.label}
                        </span>

                        {/* On / Off Switch for inclusion in Master Mix */}
                        <button
                          onClick={() => toggleStemMute(stemKey)}
                          title={`Turn ${item.label} ${isEnabledInMaster ? 'OFF' : 'ON'} in Master`}
                          className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition cursor-pointer font-medium ${
                            isEnabledInMaster
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50'
                              : 'bg-red-950/30 text-slate-500 border border-slate-700 hover:text-slate-300'
                          }`}
                        >
                          <span className={`w-1.5 h-1.5 rounded-full ${isEnabledInMaster ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
                          <span>{isEnabledInMaster ? 'in master: ON' : 'in master: OFF'}</span>
                        </button>
                      </div>

                      {/* Stem Box: Spectrogram + Download + Play */}
                      <div className={`border rounded-xl p-1.5 flex items-center justify-between gap-2 bg-black/50 transition ${
                        isEnabledInMaster ? 'border-white/40' : 'border-white/15 opacity-60'
                      }`}>
                        {/* Interactive Click-to-Seek Spectrogram */}
                        <div className="flex-1 h-12 rounded overflow-hidden bg-black/40">
                          {hasStem ? (
                            <SpectrogramCanvas
                              matrix={stem.spectrogram}
                              isMini
                              height={48}
                              duration={stemDur}
                              currentTime={isTrackActive ? currentTime : 0}
                              isPlaying={isTrackActive && isPlaying}
                              onSeek={(t) => handlePlayOrSeek(item.id, stem.audioUrl, stemDur, t)}
                            />
                          ) : (
                            <SpectrogramCanvas
                              isEmpty
                              emptyText=""
                              isMini
                              height={48}
                            />
                          )}
                        </div>

                        {/* Action buttons: Download + Play Button */}
                        <div className="flex items-center gap-1.5 shrink-0">
                          {/* Download Button Next to Play */}
                          {hasStem && (
                            <a
                              href={stem.audioUrl}
                              download={`${item.label}.wav`}
                              title={`Download ${item.label} stem WAV`}
                              className="w-8 h-8 rounded-full border border-white/30 hover:border-white text-slate-300 hover:text-white flex items-center justify-center text-xs transition cursor-pointer"
                            >
                              <FaDownload className="text-[10px]" />
                            </a>
                          )}

                          {/* Round Play Button */}
                          <button
                            disabled={!hasStem}
                            onClick={() => stem && handlePlayOrSeek(item.id, stem.audioUrl, stemDur)}
                            title={`Play ${item.label} solo`}
                            className={`w-8 h-8 rounded-full border flex items-center justify-center text-xs transition cursor-pointer ${
                              hasStem
                                ? isTrackActive && isPlaying
                                  ? 'border-white bg-white text-black'
                                  : 'border-white/50 text-white hover:border-white'
                                : 'border-white/10 text-white/20 cursor-not-allowed'
                            }`}
                          >
                            {isTrackActive && isPlaying ? <FaPause /> : <FaPlay className="ml-0.5" />}
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* COLUMN 4: CONVERGING MASTER NODE */}
              <div className="xl:col-span-3 flex flex-col justify-center my-auto pt-2 xl:pt-0">
                <div className="border border-emerald-500/60 rounded-xl p-3 bg-emerald-950/20 flex flex-col gap-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-emerald-300 uppercase tracking-wider">
                      master
                    </span>
                    {masterUrl && (
                      <span className="text-[11px] text-emerald-400/80">
                        active stems live mix
                      </span>
                    )}
                  </div>

                  {/* Master Real STFT Spectrogram (Interactive Click-to-Seek) */}
                  <div className="h-28 w-full rounded overflow-hidden bg-black/60 border border-white/20">
                    {masterUrl ? (
                      <SpectrogramCanvas
                        matrix={masterSpectrogram}
                        isEmpty={masterSpectrogram.length === 0}
                        duration={masterDuration}
                        currentTime={activeTrackId === 'master' ? currentTime : 0}
                        isPlaying={activeTrackId === 'master' && isPlaying}
                        onSeek={(t) => handlePlayOrSeek('master', masterUrl, masterDuration, t)}
                      />
                    ) : (
                      <SpectrogramCanvas
                        isEmpty
                        emptyText="Awaiting Master"
                        isMini
                        height={112}
                      />
                    )}
                  </div>

                  {/* Prominent Master Play & Download Bar */}
                  <div className="flex items-center gap-2">
                    <button
                      disabled={!masterUrl}
                      onClick={() => masterUrl && handlePlayOrSeek('master', masterUrl, masterDuration)}
                      className={`flex-1 py-2.5 rounded-lg border text-xs font-medium flex items-center justify-center gap-2 transition cursor-pointer ${
                        masterUrl
                          ? activeTrackId === 'master' && isPlaying
                            ? 'bg-emerald-500 border-emerald-400 text-black font-bold'
                            : 'border-emerald-400/80 text-emerald-300 hover:bg-emerald-500/10'
                          : 'border-white/10 text-white/20 cursor-not-allowed'
                      }`}
                    >
                      {activeTrackId === 'master' && isPlaying ? (
                        <>
                          <FaPause className="text-[11px]" />
                          <span>pause master</span>
                        </>
                      ) : (
                        <>
                          <FaPlay className="text-[11px]" />
                          <span>play master</span>
                        </>
                      )}
                    </button>

                    {/* Download Button Next to Master Play Button */}
                    {masterUrl && (
                      <a
                        href={masterUrl}
                        download="charlie_master.wav"
                        title="Download Master WAV"
                        className="p-2.5 rounded-lg border border-emerald-400/80 text-emerald-300 hover:bg-emerald-500/10 transition flex items-center justify-center cursor-pointer"
                      >
                        <FaDownload className="text-xs" />
                      </a>
                    )}
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}
      </main>
    </div>
  );
}
