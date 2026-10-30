# CHARLIE — AI-Powered Sound-to-Music Generation

> **Turn Anything Into Music.**  
> Transform ordinary environmental sounds (tapping a glass, knocking on a table, clapping, humming) into structured, progressive musical compositions.

---

## 1. Project Overview & Architecture

CHARLIE transforms real-world audio inputs into multi-instrument musical arrangements with live progressive spectrogram rendering.

```
+-------------------------------------------------------------+
|                      User Sound Input                       |
|          (Tapping a cup, knocking, clapping, etc.)          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                A. Audio Preprocessing & Features            |
| - Onset detection & tempo estimation (BPM)                  |
| - STFT Spectrogram & Mel bands                              |
| - Spectral centroid (timbre/brightness) & RMS energy        |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|           B. Custom Trained Neural Network Adapter          |
|                 (checkpoints/charlie_adapter.pt)            |
| - AudioFeatureEncoder (Mel + Conv1D + Temporal pooling)     |
| - Bidirectional GRU Temporal Sequence Modeling             |
| - Multi-Head Prediction:                                    |
|     * Rhythmic Grid Head (16-step grid per bar)             |
|     * Harmony & Key Head (12 chromatic keys)                |
|     * Instrument Progression Head (Piano/Bass/Drums/Guitar) |
|     * Dynamic Velocity & Mood Descriptors                   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|           C. Progressive Arrangement & Synthesis            |
| - Bar 1: Original audio + Rhythmic Piano motifs             |
| - Bar 2: Bass accompaniment enters on root notes            |
| - Bar 3: Drums enter locked to the 16th-step groove         |
| - Bar 4: Guitar arpeggios & full harmonic arrangement       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              D. Incremental Spectrogram & Playback          |
| - Real-time STFT spectrogram growth across the timeline     |
| - Audio chunk streaming & WAV composition download          |
+-------------------------------------------------------------+
```

---

## 2. Model Training & Evaluation Results

The custom Charlie Neural Network Adapter was trained on the paired **ESC-50** environmental audio dataset aligned with rhythmic groove and harmonic targets on NVIDIA GeForce RTX 4050 Laptop GPU (CUDA).

### Quantitative Evaluation vs. Untrained Baseline:

| Metric | Untrained Baseline | Trained Charlie Adapter | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Rhythm BCE Loss** | `0.6907` | **`0.3932`** | **+43.1%** |
| **Instrument Entry BCE** | `0.7013` | **`0.1739`** | **+75.2%** |
| **Harmony / Key CE** | `2.4789` | **`1.9866`** | **+19.9%** |

Checkpoints are saved at `checkpoints/charlie_adapter.pt`.

---

## 3. Repository Structure

```
charlie/
├── ai/
│   ├── encoder.py              # Pretrained environmental audio feature encoder
│   ├── adapter.py              # Custom Charlie Neural Network Adapter
│   ├── dataset.py              # Paired environmental-to-music dataset loader
│   ├── train.py                # Supervised multi-task training script
│   ├── inference.py            # Neural inference pipeline
│   └── evaluate.py             # Evaluation comparing against baseline
├── audio/
│   ├── preprocess.py           # Waveform loading, normalization, pad/trim
│   ├── features.py             # Onset, tempo, chroma, pitch, brightness extraction
│   ├── beat_detection.py       # Beat grid tracking & 16-step quantization
│   ├── spectrogram.py          # STFT magnitude spectrogram generation & downsampling
│   ├── midi_generator.py       # Multi-track PrettyMIDI generator
│   ├── synthesizer.py          # Dual-mode synthesizer (Piano, Bass, Drums, Guitar)
│   └── mixer.py                # Track mixing, soft-limiting, volume balancing
├── generation/
│   ├── arranger.py             # Progressive 4-bar arrangement controller
│   ├── chunk_generator.py      # Incremental chunk generator for live spectrogram
│   └── musicgen_backend.py     # Optional facebook/musicgen-small backend integration
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── routes.py               # REST API endpoints
│   └── jobs.py                 # Asynchronous generation job coordinator
├── checkpoints/
│   ├── charlie_adapter.pt      # Saved best model checkpoint
│   └── charlie_adapter_history.json
├── dataset/                    # ESC-50, Groove, and MAESTRO datasets
├── models/                     # musicgen-small local weights
├── outputs/                    # Uploads and generated audio mix WAVs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Hardware Configuration & Optimization

- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)
- **CPU**: AMD Ryzen 7 7840HS
- **RAM**: 16 GB
- **PyTorch**: 2.11.0+cu128
- **Memory Footprint**:
  - The custom Charlie adapter consumes **< 150 MB VRAM**, making training and inference instantaneous (~30ms per sound).
  - MusicGen Small operates in **FP16** with lazy loading, keeping peak VRAM below **3.0 GB**.

---

## 5. Quickstart & Usage

### 1. Universal Composition CLI (Process Any Audio File)
You can now pass any audio file (`.wav`, `.mp3`, `.flac`) directly:
```powershell
.\.venv\Scripts\python.exe compose.py path/to/your_sound.mp3
```
Or specify bar length and mood:
```powershell
.\.venv\Scripts\python.exe compose.py mug-beat.mp3 --bars 8
.\.venv\Scripts\python.exe compose.py sample-2.mp3 --bars 8
```
Or simply run without arguments to get an interactive prompt:
```powershell
.\.venv\Scripts\python.exe compose.py
```

### 2. Run Pipeline Verification Test
```powershell
.\.venv\Scripts\python.exe test_pipeline.py
```

### 2. Train the Charlie Adapter
```powershell
.\.venv\Scripts\python.exe -m ai.train
```

### 3. Evaluate Model Checkpoint
```powershell
.\.venv\Scripts\python.exe -m ai.evaluate
```

### 4. Start the Backend API Server
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at: `http://localhost:8000/docs`

---

## 6. API Endpoints

- `POST /api/audio/upload`: Upload environmental sound recording (WAV/MP3).
- `POST /api/audio/analyze`: Extract onset times, BPM, key, brightness, and STFT spectrogram.
- `POST /api/generate`: Launch progressive composition generation in background.
- `GET /api/generate/{job_id}`: Poll generation progress, active instruments, and streaming chunks.
- `GET /api/audio/{job_id}`: Stream or download final mixed composition WAV.
- `GET /api/spectrogram/{job_id}`: Retrieve real-time STFT spectrogram matrix for canvas visualizer.
