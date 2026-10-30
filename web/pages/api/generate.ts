import type { NextApiRequest, NextApiResponse } from 'next';
import path from 'path';
import fs from 'fs';
import { execFile } from 'child_process';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { filePath, genre, musical, seed, bars } = req.body;
  if (!filePath) {
    return res.status(400).json({ error: 'Missing filePath' });
  }

  const charlieRoot = path.resolve(process.cwd(), '..');
  const fullAudioPath = path.resolve(charlieRoot, filePath);

  if (!fs.existsSync(fullAudioPath)) {
    return res.status(404).json({ error: 'Audio file not found: ' + filePath });
  }

  const pyPath = path.join(charlieRoot, '.venv', 'Scripts', 'python.exe');
  const composeScript = path.join(charlieRoot, 'compose.py');
  const analyzerScript = path.join(charlieRoot, 'analyze_cli.py');

  const targetSeed = seed ? parseInt(String(seed), 10) : Math.floor(1000 + Math.random() * 900000);
  const targetBars = bars ? parseInt(String(bars), 10) : 8;

  const args = [
    composeScript,
    fullAudioPath,
    '--bars', String(targetBars),
    '--seed', String(targetSeed),
  ];

  if (genre && genre !== 'random' && genre !== 'auto') {
    args.push('--genre', String(genre).toLowerCase());
  }

  if (musical) {
    args.push('--musical');
  }

  return new Promise<void>((resolve) => {
    execFile(pyPath, args, { cwd: charlieRoot }, (err, stdout, stderr) => {
      if (err) {
        console.error('Compose error:', stderr || err.message);
        res.status(500).json({ error: 'Composition failed: ' + (stderr || err.message) });
        return resolve();
      }

      // Locate stems directory
      const fileBase = path.basename(filePath, path.extname(filePath));
      const stemsDir = path.join(charlieRoot, 'outputs', 'generated', `stems_${fileBase}`);

      // Call analyzer on stemsDir to get real numerical STFT spectrograms for all stems
      execFile(pyPath, [analyzerScript, stemsDir], (stemErr, stemStdout) => {
        let stemsSpectrograms: Record<string, number[][]> = {};
        let stemsDurations: Record<string, number> = {};
        try {
          if (stemStdout) {
            const parsed = JSON.parse(stemStdout.trim());
            stemsSpectrograms = parsed.stems_spectrograms || {};
            stemsDurations = parsed.stems_durations || {};
          }
        } catch (parseE) {
          console.error('Failed to parse stem spectrograms:', parseE);
        }

        // Collect audio URLs & real spectrograms for each stem
        const stems: Record<string, { audioUrl: string; spectrogram: number[][]; duration: number }> = {};
        const stemNames = ['original', 'melody', 'drums', 'bass', 'chords'];

        for (const name of stemNames) {
          const stemFile = path.join(stemsDir, `${name}.wav`);
          if (fs.existsSync(stemFile)) {
            const rel = path.relative(charlieRoot, stemFile);
            stems[name] = {
              audioUrl: `/api/audio?file=${encodeURIComponent(rel)}`,
              spectrogram: stemsSpectrograms[name] || [],
              duration: stemsDurations[name] || 14.0,
            };
          }
        }

        // Master audio URL & Spectrogram
        const latestMaster = path.join(charlieRoot, 'outputs', 'generated', 'latest_composition.wav');
        const masterRel = path.relative(charlieRoot, latestMaster);

        // Analyze master
        execFile(pyPath, [analyzerScript, latestMaster], (masterErr, masterStdout) => {
          let masterSpectrogram: number[][] = [];
          let masterDuration = 14.0;
          try {
            if (masterStdout) {
              const mParsed = JSON.parse(masterStdout.trim());
              masterSpectrogram = mParsed.spectrogram || [];
              masterDuration = mParsed.duration || 14.0;
            }
          } catch {}

          res.status(200).json({
            success: true,
            seed: targetSeed,
            genre: genre || 'auto',
            musical: Boolean(musical),
            masterUrl: `/api/audio?file=${encodeURIComponent(masterRel)}`,
            masterSpectrogram,
            masterDuration,
            stems,
            outputLog: stdout,
          });
          resolve();
        });
      });
    });
  });
}
