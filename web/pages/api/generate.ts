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

  const targetSeed = seed ? parseInt(String(seed), 10) : Math.floor(1000 + Math.random() * 900000);
  const targetBars = bars ? parseInt(String(bars), 10) : 8;

  const args = [
    composeScript,
    fullAudioPath,
    '--bars', String(targetBars),
    '--seed', String(targetSeed),
  ];

  if (genre) {
    args.push('--genre', String(genre).toLowerCase());
  }

  if (musical) {
    args.push('--musical');
  }

  execFile(pyPath, args, { cwd: charlieRoot }, (err, stdout, stderr) => {
    if (err) {
      console.error('Compose error:', stderr || err.message);
      return res.status(500).json({ error: 'Composition failed: ' + (stderr || err.message) });
    }

    // Parse the output files
    const fileBase = path.basename(filePath, path.extname(filePath));
    const stemsDir = path.join(charlieRoot, 'outputs', 'generated', `stems_${fileBase}`);

    // Stems
    const stems: Record<string, string> = {};
    const stemNames = ['original', 'melody', 'drums', 'bass', 'chords'];

    for (const name of stemNames) {
      const stemFile = path.join(stemsDir, `${name}.wav`);
      if (fs.existsSync(stemFile)) {
        const rel = path.relative(charlieRoot, stemFile);
        stems[name] = `/api/audio?file=${encodeURIComponent(rel)}`;
      }
    }

    // Master file
    const latestMaster = path.join(charlieRoot, 'outputs', 'generated', 'latest_composition.wav');
    const masterRel = path.relative(charlieRoot, latestMaster);

    return res.status(200).json({
      success: true,
      seed: targetSeed,
      genre: genre || 'auto',
      musical: Boolean(musical),
      masterUrl: `/api/audio?file=${encodeURIComponent(masterRel)}`,
      stems,
      outputLog: stdout,
    });
  });
}
