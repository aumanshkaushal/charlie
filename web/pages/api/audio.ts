import type { NextApiRequest, NextApiResponse } from 'next';
import fs from 'fs';
import path from 'path';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const { file } = req.query;
  if (!file || typeof file !== 'string') {
    return res.status(400).json({ error: 'Missing file query param' });
  }

  // Resolve file safely relative to charlie root
  const charlieRoot = path.resolve(process.cwd(), '..');
  const safePath = path.resolve(charlieRoot, file);

  if (!safePath.startsWith(charlieRoot) || !fs.existsSync(safePath)) {
    return res.status(404).json({ error: 'Audio file not found' });
  }

  const stat = fs.statSync(safePath);
  const total = stat.size;

  const range = req.headers.range;
  if (range) {
    const parts = range.replace(/bytes=/, '').split('-');
    const partialStart = parts[0];
    const partialEnd = parts[1];

    const start = parseInt(partialStart, 10);
    const end = partialEnd ? parseInt(partialEnd, 10) : total - 1;
    const chunkSize = end - start + 1;

    const fileStream = fs.createReadStream(safePath, { start, end });
    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${total}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunkSize,
      'Content-Type': 'audio/wav',
    });
    fileStream.pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Length': total,
      'Content-Type': 'audio/wav',
    });
    fs.createReadStream(safePath).pipe(res);
  }
}
