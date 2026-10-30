import React, { useEffect, useRef, useState, useCallback } from 'react';

interface Props {
  matrix?: number[][];
  width?: number;
  height?: number;
  isEmpty?: boolean;
  emptyText?: string;
  isMini?: boolean;
  duration?: number;
  currentTime?: number;
  isPlaying?: boolean;
  onSeek?: (seekTime: number) => void;
}

// STFT Heatmap color calculation
function getAudioColor(val: number): string {
  const v = Math.max(0, Math.min(1, val));
  if (v < 0.04) return '#090a12';
  if (v < 0.28) {
    const t = v / 0.28;
    const r = Math.floor(18 + t * 85);
    const g = Math.floor(8 + t * 15);
    const b = Math.floor(45 + t * 90);
    return `rgb(${r},${g},${b})`;
  } else if (v < 0.6) {
    const t = (v - 0.28) / 0.32;
    const r = Math.floor(103 + t * 122);
    const g = Math.floor(23 + t * 47);
    const b = Math.floor(135 - t * 65);
    return `rgb(${r},${g},${b})`;
  } else if (v < 0.85) {
    const t = (v - 0.6) / 0.25;
    const r = Math.floor(225 + t * 25);
    const g = Math.floor(70 + t * 115);
    const b = Math.floor(70 - t * 45);
    return `rgb(${r},${g},${b})`;
  } else {
    const t = (v - 0.85) / 0.15;
    const r = 250;
    const g = Math.floor(185 + t * 65);
    const b = Math.floor(25 + t * 190);
    return `rgb(${r},${g},${b})`;
  }
}

// Generate realistic ambient idle spectrogram
function generateIdleSpectrogram(bins = 32, frames = 100): number[][] {
  const data: number[][] = [];
  for (let b = 0; b < bins; b++) {
    const row: number[] = [];
    const freqFactor = Math.exp(-b / 12);
    for (let f = 0; f < frames; f++) {
      const noise = (Math.sin(f * 0.18 + b * 0.4) + Math.cos(f * 0.09 - b * 0.2)) * 0.12;
      const pulse = (f > 25 && f < 40) || (f > 65 && f < 80) ? 0.4 * freqFactor : 0.06 * freqFactor;
      row.push(Math.max(0.02, Math.min(0.7, pulse + noise * freqFactor + 0.05)));
    }
    data.push(row);
  }
  return data;
}

export default function SpectrogramCanvas({
  matrix,
  width = 600,
  height = 180,
  isEmpty = false,
  emptyText,
  isMini = false,
  duration = 0,
  currentTime = 0,
  isPlaying = false,
  onSeek,
}: Props) {
  const baseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverFraction, setHoverFraction] = useState<number | null>(null);

  // 1. Render Base STFT Spectrogram Matrix (Only when matrix or dimensions change)
  useEffect(() => {
    const canvas = baseCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    let activeMatrix = matrix;
    if (isEmpty || !matrix || matrix.length === 0) {
      activeMatrix = generateIdleSpectrogram(isMini ? 24 : 48, isMini ? 80 : 160);
    }

    const numFreqBins = activeMatrix.length;
    const numTimeFrames = activeMatrix[0]?.length || 0;
    if (numTimeFrames === 0) return;

    const cellW = width / numTimeFrames;
    const cellH = height / numFreqBins;

    for (let f = 0; f < numFreqBins; f++) {
      const row = activeMatrix[f];
      const y = height - (f + 1) * cellH;

      for (let t = 0; t < numTimeFrames; t++) {
        const val = row[t] || 0;
        const displayVal = isEmpty ? val * 0.55 : val;
        ctx.fillStyle = getAudioColor(displayVal);
        ctx.fillRect(t * cellW, y, cellW + 0.5, cellH + 0.5);
      }
    }

    // Centered label on empty state
    if (isEmpty && emptyText) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(0, height / 2 - 20, width, 40);

      ctx.fillStyle = '#f8fafc';
      ctx.font = isMini ? 'italic 12px serif' : 'italic 18px serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(emptyText, width / 2, height / 2);
    }
  }, [matrix, width, height, isEmpty, emptyText, isMini]);

  // 2. Render Overlay (Playhead & Hover Seek Cursor - 60fps lightweight)
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    // Playhead Line
    if (duration > 0 && currentTime >= 0 && (isPlaying || currentTime > 0)) {
      const progress = Math.max(0, Math.min(1, currentTime / duration));
      const playX = progress * width;

      // Glow beam
      ctx.beginPath();
      ctx.moveTo(playX, 0);
      ctx.lineTo(playX, height);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#38bdf8';
      ctx.shadowColor = '#0284c7';
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Scrubber head marker at top
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(playX, 4, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Hover Seek Cursor
    if (hoverFraction !== null && !isEmpty && onSeek) {
      const hX = hoverFraction * width;
      ctx.beginPath();
      ctx.moveTo(hX, 0);
      ctx.lineTo(hX, height);
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [width, height, duration, currentTime, isPlaying, hoverFraction, isEmpty, onSeek]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isEmpty || !onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setHoverFraction(Math.max(0, Math.min(1, x / rect.width)));
  };

  const handleMouseLeave = () => {
    setHoverFraction(null);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isEmpty || !onSeek || duration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const fraction = Math.max(0, Math.min(1, x / rect.width));
    onSeek(fraction * duration);
  };

  return (
    <div className="relative w-full h-full select-none">
      {/* Base Canvas: STFT Spectrogram Heatmap */}
      <canvas
        ref={baseCanvasRef}
        width={width}
        height={height}
        className="w-full h-full rounded object-cover block"
      />

      {/* Overlay Canvas: Dynamic Playhead & Hover Scrub Line */}
      <canvas
        ref={overlayCanvasRef}
        width={width}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        className={`absolute inset-0 w-full h-full rounded object-cover ${
          !isEmpty && onSeek ? 'cursor-pointer' : ''
        }`}
      />
    </div>
  );
}
