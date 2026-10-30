import React, { useState, useRef, useEffect } from 'react';
import Head from 'next/head';
import SpectrogramCanvas from '@/components/SpectrogramCanvas';

const GENRES = ['lofi', 'house', 'trap', 'synthwave', 'afrobeat', 'ambient', 'rnb'];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [filePath, setFilePath] = useState<string>('');
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [analysis, setAnalysis] = useState<any>(null);

  // Config options
  const [musical, setMusical] = useState<boolean>(true);
  const [genre, setGenre] = useState<string>('lofi');
  const [seed, setSeed] = useState<number>(() => Math.floor(1000 + Math.random() * 900000));
  const [bars, setBars] = useState<number>(8);

  // Generation state
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generationDone, setGenerationDone] = useState<boolean>(false);
  const [masterUrl, setMasterUrl] = useState<string>('');
  const [stems, setStems] = useState<Record<string, string>>({});

  // Audio Playback states
  const [playingTrack, setPlayingTrack] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Play / Pause handler
  const togglePlay = (url: string, trackName: string) => {
    if (playingTrack === trackName) {
      audioRef.current?.pause();
      setPlayingTrack(null);
    } else {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.play();
      setPlayingTrack(trackName);
      audio.onended = () => setPlayingTrack(null);
    }
  };

  // Upload handler
  const handleUpload = async (selectedFile: File) => {
    setFile(selectedFile);
    setFileName(selectedFile.name);
    setGenerationDone(false);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setFilePath(data.filePath);
        setAudioUrl(data.audioUrl);
        setAnalysis(data.analysis);
      }
    } catch (err) {
      console.error('Upload error:', err);
    }
  };

  // Quick load existing demo files
  const loadPreset = async (presetPath: string) => {
    setFileName(presetPath);
    setFilePath(presetPath);
    setAudioUrl(`/api/audio?file=${encodeURIComponent(presetPath)}`);
    setGenerationDone(false);

    // Fetch analysis
    try {
      const res = await fetch(`/api/audio?file=${encodeURIComponent(presetPath)}`);
      // Trigger analyze via upload API or dummy
      setAnalysis({
        bpm: 136,
        key: presetPath.includes('mug') ? 'D' : 'B',
        spectrogram: Array.from({ length: 32 }, () =>
          Array.from({ length: 64 }, () => Math.random() * 0.8)
        ),
      });
    } catch (e) {
      console.error(e);
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
          genre,
          musical,
          seed,
          bars,
        }),
      });

      const data = await res.json();
      if (data.success) {
        setMasterUrl(data.masterUrl);
        setStems(data.stems || {});
        setGenerationDone(true);
      }
    } catch (err) {
      console.error('Generation error:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0d0d12] text-white p-6 md:p-10 flex flex-col font-['Gaegu']">
      <Head>
        <title>hey charlie</title>
        <meta name="description" content="Turn anything into music" />
        <link rel="icon" href="data:," />
      </Head>

      {/* Header - Minimal, no logos, funky handwriting squiggly font */}
      <header className="mb-8 flex items-baseline justify-between border-b border-white/10 pb-4">
        <div>
          <h1 className="text-5xl md:text-6xl font-bold tracking-wider text-white">
            hey charlie
          </h1>
          <p className="text-xl text-white/50 mt-1">turn anything into music</p>
        </div>

        {/* Quick sample pickers */}
        <div className="flex items-center gap-3 text-lg">
          <span className="text-white/40">try sample:</span>
          <button
            onClick={() => loadPreset('mug-beat.mp3')}
            className="px-3 py-1 sketch-border text-sm hover:bg-white/10 transition cursor-pointer"
          >
            mug-beat.mp3
          </button>
          <button
            onClick={() => loadPreset('sample-2.mp3')}
            className="px-3 py-1 sketch-border text-sm hover:bg-white/10 transition cursor-pointer"
          >
            sample-2.mp3
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      {!filePath ? (
        /* STATE 1: Empty / Upload state (Left of Excalidraw Sketch) */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 my-auto max-w-5xl mx-auto w-full">
          {/* Box 1: Upload MP3 */}
          <label className="sketch-border bg-[#13131a] p-12 flex flex-col items-center justify-center min-h-[380px] cursor-pointer hover:border-white transition group">
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
            />
            <div className="text-4xl font-bold mb-4 group-hover:scale-105 transition">
              Upload MP3
            </div>
            <p className="text-xl text-white/50 text-center">
              drop your mug beat, knock, clap, or humming sound here
            </p>
            <div className="mt-8 px-5 py-2 sketch-border text-lg bg-white/5 hover:bg-white/10">
              browse file
            </div>
          </label>

          {/* Box 2: Empty Spectrogram */}
          <div className="sketch-border bg-[#13131a] p-6 flex flex-col justify-between min-h-[380px] relative">
            <div className="w-full h-[280px]">
              <SpectrogramCanvas isEmpty emptyText="Empty spectrogram" />
            </div>
            {/* Play button icon in bottom right */}
            <div className="flex justify-end mt-2">
              <div className="w-12 h-12 sketch-border flex items-center justify-center text-xl text-white/40 rounded-full">
                ▶
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* STATE 2: Uploaded & Active Workspace (Right of Excalidraw Sketch) */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start max-w-7xl mx-auto w-full">
          {/* LEFT COLUMN: Main Spectrogram + File Info + Config (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            {/* Main Spectrogram Box */}
            <div className="sketch-border bg-[#13131a] p-4 relative">
              <div className="h-[220px] w-full overflow-hidden rounded-lg">
                <SpectrogramCanvas
                  matrix={analysis?.spectrogram}
                  isEmpty={!analysis?.spectrogram}
                />
              </div>
              {/* File name */}
              <div className="mt-3 flex items-center justify-between text-xl font-bold">
                <span className="truncate">{fileName}</span>
                {analysis?.bpm && (
                  <span className="text-sm px-2 py-0.5 sketch-border bg-white/5">
                    {analysis.bpm} BPM • Key: {analysis.key || 'D'}
                  </span>
                )}
              </div>
            </div>

            {/* Mini Player Bar */}
            <div className="sketch-border bg-[#13131a] px-5 py-3 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-6 w-32 bg-gradient-to-r from-purple-800 via-orange-600 to-yellow-400 rounded opacity-75" />
                <span className="text-lg text-white/70">preview audio</span>
              </div>
              <button
                onClick={() => togglePlay(audioUrl, 'preview')}
                className="px-4 py-1.5 sketch-border text-lg hover:bg-white/10 transition cursor-pointer"
              >
                {playingTrack === 'preview' ? '⏸ pause' : '▶ play'}
              </button>
            </div>

            {/* CONFIG BOX (Blue outline as in drawing) */}
            <div className="sketch-border-blue bg-[#121624] p-5 flex flex-col gap-4">
              <div className="text-xl text-blue-400 font-bold uppercase tracking-wider">
                config
              </div>

              {/* [x] musical checkbox */}
              <label className="flex items-center gap-3 text-2xl cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={musical}
                  onChange={(e) => setMusical(e.target.checked)}
                  className="w-6 h-6 rounded accent-blue-500 cursor-pointer"
                />
                <span className={musical ? 'text-white font-bold' : 'text-white/60'}>
                  [x] musical chords & space
                </span>
              </label>

              {/* Genre Pills */}
              <div>
                <div className="text-lg text-white/60 mb-2">genre:</div>
                <div className="flex flex-wrap gap-2">
                  {GENRES.map((g) => (
                    <button
                      key={g}
                      onClick={() => setGenre(g)}
                      className={`px-3 py-1 rounded-full text-base transition cursor-pointer ${
                        genre === g
                          ? 'bg-blue-600 text-white font-bold border border-blue-400'
                          : 'sketch-border bg-white/5 text-white/70 hover:bg-white/10'
                      }`}
                    >
                      {g}
                    </button>
                  ))}
                </div>
              </div>

              {/* Seed & Bars */}
              <div className="grid grid-cols-2 gap-3 pt-2">
                <div>
                  <div className="text-sm text-white/60 mb-1">seed:</div>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value) || 1)}
                      className="w-full bg-[#1b2138] sketch-border px-3 py-1 text-lg text-white focus:outline-none"
                    />
                    <button
                      onClick={() => setSeed(Math.floor(1000 + Math.random() * 900000))}
                      title="Randomize Seed"
                      className="p-1.5 sketch-border hover:bg-white/10 text-lg cursor-pointer"
                    >
                      🎲
                    </button>
                  </div>
                </div>

                <div>
                  <div className="text-sm text-white/60 mb-1">length:</div>
                  <select
                    value={bars}
                    onChange={(e) => setBars(parseInt(e.target.value))}
                    className="w-full bg-[#1b2138] sketch-border px-3 py-1.5 text-lg text-white focus:outline-none cursor-pointer"
                  >
                    <option value={4}>4 bars (short)</option>
                    <option value={8}>8 bars (full)</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* CENTER: Generate Action Button & Hand-Drawn Flow (2 cols) */}
          <div className="lg:col-span-2 flex flex-col items-center justify-center my-auto py-6">
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className={`w-full py-6 px-4 sketch-border-green bg-[#122818] hover:bg-[#1a3822] text-green-400 font-bold text-3xl transition transform active:scale-95 cursor-pointer flex flex-col items-center justify-center gap-2 shadow-lg shadow-green-950/40 ${
                isGenerating ? 'opacity-50 animate-pulse' : ''
              }`}
            >
              <span>{isGenerating ? 'generating...' : 'generate'}</span>
              <span className="text-4xl">➔</span>
            </button>

            {/* Hand-drawn branching wireframe indicator */}
            <div className="hidden lg:flex flex-col items-center mt-6 text-white/30 text-xs">
              <span className="text-2xl text-green-500">⤹ ⤸</span>
              <span>resynthesizing stems</span>
            </div>
          </div>

          {/* RIGHT COLUMN: Stacked Stem Cards (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="text-2xl font-bold flex items-center justify-between border-b border-white/10 pb-2">
              <span>stems</span>
              {masterUrl && (
                <button
                  onClick={() => togglePlay(masterUrl, 'master')}
                  className="px-4 py-1 sketch-border text-base bg-green-500/20 text-green-400 hover:bg-green-500/30 transition cursor-pointer"
                >
                  {playingTrack === 'master' ? '⏸ pause full mix' : '▶ play full mix'}
                </button>
              )}
            </div>

            {/* 4 Stacked Stem Cards exactly like the sketch */}
            {[
              { id: 'original', label: 'original', color: 'from-purple-900 to-indigo-700' },
              { id: 'melody', label: 'melody', color: 'from-orange-800 to-yellow-600' },
              { id: 'drums', label: 'drums', color: 'from-red-900 to-rose-700' },
              { id: stems['chords'] ? 'chords' : 'bass', label: 'harmony', color: 'from-blue-900 to-cyan-700' },
            ].map((stem) => {
              const stemUrl = stems[stem.id] || stems['bass'];
              const isReady = Boolean(stemUrl);

              return (
                <div
                  key={stem.id}
                  className={`sketch-border bg-[#13131a] p-3 flex items-center justify-between gap-4 transition ${
                    isReady ? 'border-white' : 'opacity-40'
                  }`}
                >
                  {/* Stem Label */}
                  <div className="w-24 text-2xl font-bold text-white capitalize">
                    {stem.label}
                  </div>

                  {/* Mini Spectrogram Slice */}
                  <div className="flex-1 h-12 bg-[#0c0c10] rounded-lg overflow-hidden border border-white/10 flex items-center px-1">
                    {isReady ? (
                      <div
                        className={`w-full h-8 bg-gradient-to-r ${stem.color} rounded opacity-80 animate-pulse`}
                      />
                    ) : (
                      <div className="w-full text-center text-xs text-white/30">
                        waiting for generate...
                      </div>
                    )}
                  </div>

                  {/* Play Button Icon */}
                  <button
                    disabled={!isReady}
                    onClick={() => stemUrl && togglePlay(stemUrl, stem.id)}
                    className={`w-12 h-12 sketch-border flex items-center justify-center text-xl transition cursor-pointer rounded-full ${
                      isReady
                        ? playingTrack === stem.id
                          ? 'bg-white text-black font-bold'
                          : 'hover:bg-white/10'
                        : 'opacity-30 cursor-not-allowed'
                    }`}
                  >
                    {playingTrack === stem.id ? '⏸' : '▶'}
                  </button>
                </div>
              );
            })}

            {/* Master WAV Download Link */}
            {masterUrl && (
              <div className="mt-4 pt-2">
                <a
                  href={masterUrl}
                  download="charlie_composition.wav"
                  className="block text-center py-3 sketch-border bg-white/5 hover:bg-white/15 text-xl font-bold transition"
                >
                  💾 Download Master Composition (WAV)
                </a>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
