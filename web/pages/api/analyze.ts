import type { NextApiRequest, NextApiResponse } from 'next';
import path from 'path';
import fs from 'fs';
import { execFile } from 'child_process';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { file } = req.query;
  if (!file || typeof file !== 'string') {
    return res.status(400).json({ error: 'Missing file param' });
  }

  const charlieRoot = path.resolve(process.cwd(), '..');
  const fullPath = path.resolve(charlieRoot, file);

  if (!fs.existsSync(fullPath)) {
    return res.status(404).json({ error: 'File not found: ' + file });
  }

  const pyPath = path.join(charlieRoot, '.venv', 'Scripts', 'python.exe');
  const analyzerScript = path.join(charlieRoot, 'analyze_cli.py');

  return new Promise<void>((resolve) => {
    execFile(pyPath, [analyzerScript, fullPath], (err, stdout) => {
      if (err || !stdout) {
        res.status(500).json({ error: 'Analysis failed' });
        return resolve();
      }
      try {
        const data = JSON.parse(stdout.trim());
        res.status(200).json(data);
      } catch (e) {
        res.status(500).json({ error: 'Invalid analyzer JSON' });
      }
      resolve();
    });
  });
}
