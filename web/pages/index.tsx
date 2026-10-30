import React, { useState, useRef, useEffect } from 'react';
import Head from 'next/head';
import SpectrogramCanvas from '@/components/SpectrogramCanvas';

const GENRES = ['random', 'lofi', 'rnb', 'house', 'trap', 'synthwave', 'afrobeat', 'ambient'];

interface StemData {
  audioUrl: string;
  spectrogram: number[][];
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
  const [bars, setBars] = useState<number>(8);

  // Generation state
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generationDone, setGenerationDone] = useState<boolean>(false);
  const [masterUrl, setMasterUrl] = useState<string>('');
  const [stems, setStems] = useState<Record<string, StemData>>({});

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

  // Upload handler with real analysis
  const handleUpload = async (selectedFile: File) => {
    setFileName(selectedFile.name);
    setGenerationDone(false);
    setIsAnalyzing(true);

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
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Load preset with REAL STFT analysis from backend
  const loadPreset = async (presetFile: string) => {
    setFileName(presetFile);
    setFilePath(presetFile);
    setAudioUrl(`/api/audio?file=${encodeURIComponent(presetFile)}`);
    setGenerationDone(false);
    setIsAnalyzing(true);

    try {
      const res = await fetch(`/api/analyze?file=${encodeURIComponent(presetFile)}`);
      const data = await res.json();
      setAnalysis(data);
    } catch (e) {
      console.error('Analysis error:', e);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Generate composition & fetch real stem spectrograms
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
    <div className="min-h-screen bg-[#08080d] text-slate-100 p-6 md:p-10 flex flex-col font-serif selection:bg-cyan-500/30">
      <Head>
        <title>hey charlie</title>
        <meta name="description" content="AI Sound-to-Music Neural Generator" />
        <link rel="icon" href="data:," />
      </Head>

      {/* Header: Funky squiggly title, strictly serif everywhere else */}
      <header className="mb-8 flex flex-col md:flex-row md:items-end justify-between border-b border-white/10 pb-5 gap-4">
        <div>
          {/* Strictly only 'hey charlie' has funky handwriting squiggly font */}
          <h1 className="font-funky text-6xl md:text-7xl font-bold tracking-wider text-white select-none">
            hey charlie
          </h1>
          <p className="text-lg italic text-slate-400 mt-1 font-serif">
            neural audio architecture & environmental sound resynthesis
          </p>
        </div>

        {/* Neural Network Presets */}
        <div className="flex items-center gap-3 text-base">
          <span className="text-slate-400 italic">Input Presets:</span>
          <button
            onClick={() => loadPreset('mug-beat.mp3')}
            className="px-3 py-1.5 neural-node text-sm hover:border-cyan-400 text-slate-200 transition cursor-pointer"
          >
            mug-beat.mp3
          </button>
          <button
            onClick={() => loadPreset('sample-2.mp3')}
            className="px-3 py-1.5 neural-node text-sm hover:border-cyan-400 text-slate-200 transition cursor-pointer"
          >
            sample-2.mp3
          </button>
        </div>
      </header>

      {/* WORKSPACE */}
      {!filePath ? (
        /* STATE 1: Empty / Input Node (Left of wireframe sketch) */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 my-auto max-w-5xl mx-auto w-full">
          {/* Node 1: Upload MP3 */}
          <label className="neural-node p-12 flex flex-col items-center justify-center min-h-[360px] cursor-pointer hover:border-cyan-400/80 transition group relative overflow-hidden">
            <div className="absolute top-3 left-4 text-xs tracking-widest uppercase text-cyan-400/70 font-mono">
              [LAYER 0 : AUDIO INPUT]
            </div>
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
            />
            <div className="w-16 h-16 rounded-full border border-cyan-400/40 flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition bg-cyan-950/20 text-cyan-300">
              ⭳
            </div>
            <div className="text-3xl font-bold mb-2 text-white">Upload MP3</div>
            <p className="text-base text-slate-400 text-center italic max-w-sm">
              feed environmental audio (glass tap, knock, clap, hum) into the neural network
            </p>
          </label>

          {/* Node 2: Empty Spectrogram */}
          <div className="neural-node p-5 flex flex-col justify-between min-h-[360px] relative">
            <div className="flex items-center justify-between text-xs tracking-widest uppercase text-slate-400/70 font-mono mb-2">
              <span>[STFT SPECTROGRAM MATRIX]</span>
              <span className="text-cyan-400">22,050 Hz</span>
            </div>
            <div className="w-full h-[260px] rounded-lg overflow-hidden border border-white/5">
              <SpectrogramCanvas isEmpty emptyText="Empty Spectrogram (Upload sound)" />
            </div>
            <div className="flex justify-end mt-3">
              <div className="w-10 h-10 rounded-full border border-white/20 flex items-center justify-center text-sm text-slate-400">
                ▶
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* STATE 2: ACTIVE NEURAL NETWORK PIPELINE (Right of wireframe sketch) */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start max-w-7xl mx-auto w-full relative">
          
          {/* LEFT COLUMN: Input Node + Real STFT + Neural Config (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            
            {/* Input Node Card */}
            <div className="neural-node p-5 relative">
              <div className="flex items-center justify-between text-xs tracking-widest uppercase text-cyan-400 font-mono mb-3">
                <span>[INPUT NODE: {fileName}]</span>
                {analysis?.bpm && (
                  <span className="text-emerald-400">
                    {analysis.bpm} BPM • Key: {analysis.key || 'D'}
                  </span>
                )}
              </div>

              {/* Real STFT Spectrogram Canvas */}
              <div className="h-[210px] w-full overflow-hidden rounded-lg border border-white/10 relative">
                {isAnalyzing ? (
                  <div className="w-full h-full flex items-center justify-center bg-black/60 text-cyan-400 italic">
                    Extracting STFT Frequency Bands...
                  </div>
                ) : (
                  <SpectrogramCanvas
                    matrix={analysis?.spectrogram}
                    isEmpty={!analysis?.spectrogram}
                  />
                )}
              </div>

              {/* Audio Preview Bar */}
              <div className="mt-4 pt-3 border-t border-white/10 flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-slate-300 italic">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                  <span>Input Waveform Ready</span>
                </div>
                <button
                  onClick={() => togglePlay(audioUrl, 'preview')}
                  className="px-4 py-1.5 neural-node text-sm hover:border-cyan-400 text-slate-200 transition cursor-pointer"
                >
                  {playingTrack === 'preview' ? '⏸ Pause' : '▶ Play Source'}
                </button>
              </div>
            </div>

            {/* NEURAL CONFIG NODE (Blue Box as in wireframe drawing) */}
            <div className="neural-node-blue p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between text-xs tracking-widest uppercase text-blue-400 font-mono">
                <span>[ADAPTER CONDITIONING CONFIG]</span>
                <span className="text-blue-300/60">Bi-GRU + Multi-Head</span>
              </div>

              {/* [x] musical toggle */}
              <label className="flex items-center gap-3 text-xl cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={musical}
                  onChange={(e) => setMusical(e.target.checked)}
                  className="w-5 h-5 rounded accent-blue-500 cursor-pointer"
                />
                <span className={musical ? 'text-white font-semibold' : 'text-slate-400'}>
                  [x] Musical Chords & Studio Reverb
                </span>
              </label>

              {/* Genre Pills with Random Option */}
              <div>
                <div className="text-sm italic text-slate-300 mb-2">Target Musical Genre:</div>
                <div className="flex flex-wrap gap-2">
                  {GENRES.map((g) => (
                    <button
                      key={g}
                      onClick={() => setGenre(g)}
                      className={`px-3 py-1 rounded-full text-sm font-serif transition cursor-pointer ${
                        genre === g
                          ? 'bg-blue-600 text-white font-bold border border-blue-400 shadow-md shadow-blue-500/30'
                          : 'neural-node text-slate-300 hover:border-blue-400/60'
                      }`}
                    >
                      {g === 'random' ? '🎲 Random' : g}
                    </button>
                  ))}
                </div>
              </div>

              {/* Seed & Length */}
              <div className="grid grid-cols-2 gap-4 pt-2">
                <div>
                  <div className="text-xs text-slate-300 mb-1 font-mono uppercase">Seed (Weights):</div>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value) || 1)}
                      className="w-full bg-[#0a0e1c] border border-blue-500/40 rounded px-3 py-1 text-sm text-white focus:outline-none"
                    />
                    <button
                      onClick={() => setSeed(Math.floor(1000 + Math.random() * 900000))}
                      title="Generate Random Seed"
                      className="p-1.5 neural-node text-sm hover:border-blue-400 cursor-pointer"
                    >
                      🎲
                    </button>
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-300 mb-1 font-mono uppercase">Bars / Duration:</div>
                  <select
                    value={bars}
                    onChange={(e) => setBars(parseInt(e.target.value))}
                    className="w-full bg-[#0a0e1c] border border-blue-500/40 rounded px-3 py-1.5 text-sm text-white focus:outline-none cursor-pointer"
                  >
                    <option value={4}>4 Bars (~7s)</option>
                    <option value={8}>8 Bars (~14s)</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* CENTER: Synaptic Activation Nexus & Generate Button (2 cols) */}
          <div className="lg:col-span-2 flex flex-col items-center justify-center my-auto py-8 relative">
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className={`w-full py-8 px-4 neural-node-green text-emerald-300 font-bold text-2xl transition transform active:scale-95 cursor-pointer flex flex-col items-center justify-center gap-3 relative overflow-hidden group ${
                isGenerating ? 'opacity-60 animate-pulse' : 'hover:scale-[1.02]'
              }`}
            >
              <div className="absolute inset-0 bg-emerald-500/10 blur-xl group-hover:bg-emerald-500/20 transition" />
              <span className="tracking-wide relative z-10">
                {isGenerating ? 'Synthesizing...' : 'Generate'}
              </span>
              <span className="text-3xl text-emerald-400 relative z-10">➔</span>
            </button>

            {/* Neural Synapse Branching Indicators */}
            <div className="hidden lg:flex flex-col items-center mt-6 text-emerald-400/60 font-mono text-xs text-center">
              <span className="text-2xl animate-pulse">⤹ ⤸</span>
              <span>4-Channel Stem Resynthesis</span>
            </div>
          </div>

          {/* RIGHT COLUMN: Output Stem Nodes with REAL Spectrograms (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-2">
              <span className="text-xs tracking-widest uppercase text-purple-400 font-mono">
                [OUTPUT LAYER : RESYNTHESIZED STEMS]
              </span>
              {masterUrl && (
                <button
                  onClick={() => togglePlay(masterUrl, 'master')}
                  className="px-3 py-1 neural-node text-xs text-emerald-400 hover:border-emerald-400 transition cursor-pointer"
                >
                  {playingTrack === 'master' ? '⏸ Pause Full Mix' : '▶ Play Full Mix'}
                </button>
              )}
            </div>

            {/* Stem Output Nodes with REAL Spectrograms */}
            {(stems['chords']
              ? [
                  { id: 'original', label: 'Original', desc: 'Beat Slice Pocket' },
                  { id: 'melody', label: 'Melody', desc: 'Pitched Harmonic Lead' },
                  { id: 'chords', label: 'Harmony', desc: 'Polyphonic 7th Chords' },
                  { id: 'bass', label: 'Bass', desc: 'Resynthesized Sub-Bass' },
                  { id: 'drums', label: 'Drums', desc: 'Kick, Snare & Hi-Hat' },
                ]
              : [
                  { id: 'original', label: 'Original', desc: 'Beat Slice Pocket' },
                  { id: 'melody', label: 'Melody', desc: 'Pitched Harmonic Lead' },
                  { id: 'bass', label: 'Bass', desc: 'Resynthesized Sub-Bass' },
                  { id: 'drums', label: 'Drums', desc: 'Kick, Snare & Hi-Hat' },
                ]
            ).map((stem) => {
              const stemData = stems[stem.id];
              const isReady = Boolean(stemData);

              return (
                <div
                  key={stem.id}
                  className={`neural-node p-3 flex items-center justify-between gap-4 transition ${
                    isReady ? 'border-white/30' : 'opacity-40'
                  }`}
                >
                  {/* Stem Label */}
                  <div className="w-28 flex flex-col">
                    <span className="text-lg font-bold text-white capitalize">
                      {stem.label}
                    </span>
                    <span className="text-xs text-slate-400 italic">
                      {stem.desc}
                    </span>
                  </div>

                  {/* REAL Mini Spectrogram Canvas */}
                  <div className="flex-1 h-14 bg-[#0a0b12] rounded-lg overflow-hidden border border-white/10 relative">
                    {isReady && stemData?.spectrogram && stemData.spectrogram.length > 0 ? (
                      <SpectrogramCanvas
                        matrix={stemData.spectrogram}
                        isMini
                        height={56}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-xs text-slate-500 italic">
                        {isGenerating ? 'Synthesizing STFT...' : 'Awaiting Generate'}
                      </div>
                    )}
                  </div>

                  {/* Stem Play / Pause Button */}
                  <button
                    disabled={!isReady}
                    onClick={() => stemData && togglePlay(stemData.audioUrl, stem.id)}
                    className={`w-11 h-11 rounded-full border flex items-center justify-center text-base transition cursor-pointer ${
                      isReady
                        ? playingTrack === stem.id
                          ? 'bg-white text-black font-bold border-white'
                          : 'border-white/30 hover:border-cyan-400 text-white'
                        : 'border-white/10 text-white/20 cursor-not-allowed'
                    }`}
                  >
                    {playingTrack === stem.id ? '⏸' : '▶'}
                  </button>
                </div>
              );
            })}

            {/* Master WAV Download Link */}
            {masterUrl && (
              <div className="mt-2 pt-2">
                <a
                  href={masterUrl}
                  download="charlie_master_composition.wav"
                  className="block text-center py-3 neural-node hover:border-emerald-400 text-emerald-300 text-lg transition font-semibold"
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
