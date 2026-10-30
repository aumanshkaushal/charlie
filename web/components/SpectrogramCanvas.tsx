import React, { useEffect, useRef } from 'react';

interface Props {
  matrix?: number[][];
  width?: number;
  height?: number;
  isEmpty?: boolean;
  emptyText?: string;
}

// Magma / Inferno Colormap: 0.0 (dark purple) -> 0.5 (magenta/orange) -> 1.0 (bright yellow/white)
function getColor(val: number): string {
  const v = Math.max(0, Math.min(1, val));
  if (v < 0.02) return '#0d0d12';
  if (v < 0.25) {
    const t = v / 0.25;
    const r = Math.floor(20 + t * 90);
    const g = Math.floor(10 + t * 20);
    const b = Math.floor(50 + t * 80);
    return `rgb(${r},${g},${b})`;
  } else if (v < 0.6) {
    const t = (v - 0.25) / 0.35;
    const r = Math.floor(110 + t * 115);
    const g = Math.floor(30 + t * 45);
    const b = Math.floor(130 - t * 50);
    return `rgb(${r},${g},${b})`;
  } else if (v < 0.85) {
    const t = (v - 0.6) / 0.25;
    const r = Math.floor(225 + t * 25);
    const g = Math.floor(75 + t * 110);
    const b = Math.floor(80 - t * 40);
    return `rgb(${r},${g},${b})`;
  } else {
    const t = (v - 0.85) / 0.15;
    const r = 250;
    const g = Math.floor(185 + t * 65);
    const b = Math.floor(40 + t * 180);
    return `rgb(${r},${g},${b})`;
  }
}

export default function SpectrogramCanvas({
  matrix,
  width = 500,
  height = 240,
  isEmpty = false,
  emptyText = 'Empty spectrogram',
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (isEmpty || !matrix || matrix.length === 0) {
      // Draw minimal dark empty container with sketch grid lines
      ctx.fillStyle = '#111116';
      ctx.fillRect(0, 0, width, height);

      // Faint grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = '24px Gaegu, sans-serif';
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

    // Draw bottom-up: low frequencies at the bottom, high at top
    for (let f = 0; f < numFreqBins; f++) {
      const row = matrix[f];
      // Invert row index so frequency 0 is at bottom
      const y = height - (f + 1) * cellH;

      for (let t = 0; t < numTimeFrames; t++) {
        const val = row[t] || 0;
        ctx.fillStyle = getColor(val);
        ctx.fillRect(t * cellW, y, cellW + 0.6, cellH + 0.6);
      }
    }
  }, [matrix, width, height, isEmpty, emptyText]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="w-full h-full rounded-xl object-cover"
    />
  );
}
