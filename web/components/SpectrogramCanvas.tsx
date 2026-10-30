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

// Heatmap: 0.0 (dark purple/black) -> crimson -> neon orange -> bright yellow/white
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

// Generate realistic idle/empty spectrogram data for visual completeness
function generateIdleSpectrogram(bins = 40, frames = 120): number[][] {
  const data: number[][] = [];
  for (let b = 0; b < bins; b++) {
    const row: number[] = [];
    const freqFactor = Math.exp(-b / 14);
    for (let f = 0; f < frames; f++) {
      const noise = (Math.sin(f * 0.18 + b * 0.4) + Math.cos(f * 0.09 - b * 0.2)) * 0.12;
      const pulse = (f > 30 && f < 45) || (f > 75 && f < 95) ? 0.35 * freqFactor : 0.05 * freqFactor;
      row.push(Math.max(0.02, Math.min(0.7, pulse + noise * freqFactor + 0.05)));
    }
    data.push(row);
  }
  return data;
}

export default function SpectrogramCanvas({
  matrix,
  width = 600,
  height = 200,
  isEmpty = false,
  emptyText,
  isMini = false,
  duration = 0,
  currentTime = 0,
  isPlaying = false,
  onSeek,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  // Render spectrogram to canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    let activeMatrix = matrix;
    if (isEmpty || !matrix || matrix.length === 0) {
      // Use synthesized ambient spectrogram pattern matching the Excalidraw wireframe
      activeMatrix = generateIdleSpectrogram(isMini ? 24 : 48, isMini ? 80 : 160);
    }

    const numFreqBins = activeMatrix.length;
    const numTimeFrames = activeMatrix[0]?.length || 0;
    if (numTimeFrames === 0) return;

    const cellW = width / numTimeFrames;
    const cellH = height / numFreqBins;

    // Draw spectrogram bottom-up
    for (let f = 0; f < numFreqBins; f++) {
      const row = activeMatrix[f];
      const y = height - (f + 1) * cellH;

      for (let t = 0; t < numTimeFrames; t++) {
        const val = row[t] || 0;
        // In empty state, slightly dim the ambient spectrogram
        const displayVal = isEmpty ? val * 0.6 : val;
        ctx.fillStyle = getAudioColor(displayVal);
        ctx.fillRect(t * cellW, y, cellW + 0.5, cellH + 0.5);
      }
    }

    // If empty text specified, render clean serif label over the ambient spectrogram
    if (isEmpty && emptyText) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
      ctx.fillRect(0, height / 2 - 20, width, 40);

      ctx.fillStyle = '#f1f5f9';
      ctx.font = isMini ? 'italic 13px serif' : 'italic 20px serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(emptyText, width / 2, height / 2);
    }

    // Draw Playhead line if playing or currentTime > 0
    if (duration > 0 && currentTime >= 0) {
      const playX = (currentTime / duration) * width;
      ctx.beginPath();
      ctx.moveTo(playX, 0);
      ctx.lineTo(playX, height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#38bdf8';
      ctx.shadowColor = '#0284c7';
      ctx.shadowBlur = 6;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // Draw Hover seek line
    if (hoverX !== null && !isEmpty) {
      ctx.beginPath();
      ctx.moveTo(hoverX, 0);
      ctx.lineTo(hoverX, height);
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [matrix, width, height, isEmpty, emptyText, isMini, duration, currentTime, isPlaying, hoverX]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isEmpty || !onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setHoverX(Math.max(0, Math.min(width, (x / rect.width) * width)));
  };

  const handleMouseLeave = () => {
    setHoverX(null);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isEmpty || !onSeek || duration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const fraction = Math.max(0, Math.min(1, clickX / rect.width));
    onSeek(fraction * duration);
  };

  return (
    <div className="relative w-full h-full group">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        className={`w-full h-full rounded object-cover ${
          !isEmpty && onSeek ? 'cursor-pointer' : ''
        }`}
      />
      {!isEmpty && onSeek && (
        <div className="absolute bottom-1 right-2 opacity-0 group-hover:opacity-100 transition-opacity text-[10px] text-white/70 bg-black/60 px-1.5 py-0.5 rounded font-serif pointer-events-none">
          click to seek
        </div>
      )}
    </div>
  );
}
