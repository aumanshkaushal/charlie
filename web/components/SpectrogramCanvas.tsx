import React, { useEffect, useRef } from 'react';

interface Props {
  matrix?: number[][];
  width?: number;
  height?: number;
  isEmpty?: boolean;
  emptyText?: string;
  isMini?: boolean;
}

// Real Heatmap Palette: 0.0 (near black) -> deep purple -> crimson -> neon orange -> bright yellow/white
function getAudioColor(val: number): string {
  const v = Math.max(0, Math.min(1, val));
  if (v < 0.05) return '#090a10';
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

export default function SpectrogramCanvas({
  matrix,
  width = 600,
  height = 240,
  isEmpty = false,
  emptyText = 'Empty Spectrogram',
  isMini = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (isEmpty || !matrix || matrix.length === 0) {
      // Neural grid placeholder
      ctx.fillStyle = '#0b0c13';
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      const stepX = isMini ? 25 : 50;
      const stepY = isMini ? 15 : 30;

      for (let x = 0; x < width; x += stepX) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += stepY) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.font = isMini ? 'italic 12px serif' : 'italic 18px serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(emptyText, width / 2, height / 2);
      return;
    }

    const numFreqBins = matrix.length;
    const numTimeFrames = matrix[0]?.length || 0;
    if (numTimeFrames === 0) return;

    const cellW = width / numTimeFrames;
    const cellH = height / numFreqBins;

    // Draw real STFT spectrogram bottom-up
    for (let f = 0; f < numFreqBins; f++) {
      const row = matrix[f];
      const y = height - (f + 1) * cellH;

      for (let t = 0; t < numTimeFrames; t++) {
        const val = row[t] || 0;
        ctx.fillStyle = getAudioColor(val);
        ctx.fillRect(t * cellW, y, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [matrix, width, height, isEmpty, emptyText, isMini]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="w-full h-full rounded-lg object-cover"
    />
  );
}
