import type { NextApiRequest, NextApiResponse } from 'next';
import formidable from 'formidable';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const charlieRoot = path.resolve(process.cwd(), '..');
  const uploadDir = path.join(charlieRoot, 'outputs', 'uploads');
  if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
  }

  const form = formidable({
    uploadDir,
    keepExtensions: true,
    maxFileSize: 25 * 1024 * 1024,
  });

  form.parse(req, async (err, fields, files) => {
    if (err) {
      return res.status(500).json({ error: 'Upload failed: ' + err.message });
    }

    const file = Array.isArray(files.file) ? files.file[0] : files.file;
    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const originalName = file.originalFilename || 'upload.wav';
    const filePath = file.filepath;
    const relPath = path.relative(charlieRoot, filePath);

    // Analyze using python analyze_cli.py
    const pyPath = path.join(charlieRoot, '.venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(charlieRoot, 'analyze_cli.py');

    execFile(pyPath, [scriptPath, filePath], (pyErr, stdout, stderr) => {
      let analysis = {
        bpm: 120,
        key: 'C',
        duration: 0,
        spectrogram: [],
      };

      try {
        if (stdout) {
          analysis = JSON.parse(stdout.trim());
        }
      } catch (parseErr) {
        console.error('Failed to parse analyzer JSON:', stdout);
      }

      return res.status(200).json({
        success: true,
        fileName: originalName,
        filePath: relPath,
        audioUrl: `/api/audio?file=${encodeURIComponent(relPath)}`,
        analysis,
      });
    });
  });
}
