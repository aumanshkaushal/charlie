"""FastAPI Route endpoints for Charlie."""
import os
import shutil
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from audio.preprocess import load_audio, normalize_audio, save_audio
from audio.features import extract_features
from audio.spectrogram import compute_stft_spectrogram, spectrogram_to_grid
from ai.inference import CharlieInferenceEngine
from generation.arranger import ArrangementEngine
from generation.chunk_generator import IncrementalChunkGenerator
from .jobs import JobManager, JobStatus


router = APIRouter()
job_manager = JobManager()

# Lazy loaded singletons
_inference_engine: Optional[CharlieInferenceEngine] = None
_arranger_engine: Optional[ArrangementEngine] = None
_chunk_gen: Optional[IncrementalChunkGenerator] = None


def get_inference_engine() -> CharlieInferenceEngine:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = CharlieInferenceEngine()
    return _inference_engine


def get_arranger_engine() -> ArrangementEngine:
    global _arranger_engine
    if _arranger_engine is None:
        _arranger_engine = ArrangementEngine()
    return _arranger_engine


def get_chunk_generator() -> IncrementalChunkGenerator:
    global _chunk_gen
    if _chunk_gen is None:
        _chunk_gen = IncrementalChunkGenerator()
    return _chunk_gen


class GenerateRequest(BaseModel):
    audio_path: str
    mood: str = "Chill"  # Chill, Energetic, Cinematic, Experimental
    duration: float = 16.0  # 10, 15, 20
    instruments: list[str] = ["piano", "bass", "drums", "guitar"]
    creativity: str = "Medium"  # Low, Medium, High


@router.post("/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Uploads recorded or selected environmental sound file."""
    upload_dir = "outputs/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    ext = os.path.splitext(file.filename)[1] or ".wav"
    file_id = str(uuid.uuid4())[:8]
    saved_path = os.path.join(upload_dir, f"{file_id}{ext}")

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "success", "file_path": saved_path, "file_id": file_id}


@router.post("/audio/analyze")
async def analyze_audio(audio_path: str = Form(...)):
    """Extracts spectrogram and musical features from audio."""
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    waveform, sr = load_audio(audio_path)
    waveform = normalize_audio(waveform)

    # 1. Acoustic features
    features, _ = extract_features(waveform, sr=sr)

    # 2. Spectrogram for original sound
    spec = compute_stft_spectrogram(waveform, sr=sr)
    spec_grid = spectrogram_to_grid(spec, max_freq_bins=64, max_time_steps=150)

    return {
        "status": "success",
        "features": {
            "bpm": features.bpm,
            "detected_key": features.detected_key,
            "duration": features.duration,
            "onset_times": features.onset_times,
            "rhythmic_density": features.rhythmic_density,
            "timbre_brightness": features.timbre_brightness,
            "spectral_centroid_hz": features.spectral_centroid_mean,
        },
        "spectrogram": spec_grid,
    }


def run_generation_task(job_id: str, req: GenerateRequest):
    """Background worker executing the complete Charlie AI generation pipeline."""
    try:
        job_manager.update_job(job_id, status=JobStatus.ANALYZING, progress_pct=15, message="Analyzing sound features...")

        # 1. Load audio
        waveform, sr = load_audio(req.audio_path)
        waveform = normalize_audio(waveform)

        # 2. Neural Adapter inference
        job_manager.update_job(job_id, status=JobStatus.ADAPTING, progress_pct=35, message="Translating timbre & rhythm with Charlie adapter...")
        inference_engine = get_inference_engine()
        predictions = inference_engine.predict(waveform, sr=sr)

        # 3. Progressive Arrangement
        job_manager.update_job(job_id, status=JobStatus.ARRANGING, progress_pct=60, message="Arranging musical composition...")
        arranger = get_arranger_engine()
        arrangement = arranger.create_arrangement(
            original_waveform=waveform,
            predictions=predictions,
            target_duration=req.duration,
            mood=req.mood,
            selected_instruments=req.instruments,
        )

        # 4. Incremental Chunks & Spectrogram computation
        job_manager.update_job(job_id, status=JobStatus.PROGRESSIVE_RENDERING, progress_pct=80, message="Rendering progressive tracks & spectrogram...")
        chunk_gen = get_chunk_generator()
        chunks = chunk_gen.generate_progressive_chunks(arrangement)

        # 5. Save final mix to outputs
        output_dir = "outputs/generated"
        os.makedirs(output_dir, exist_ok=True)
        final_wav_path = os.path.join(output_dir, f"charlie_mix_{job_id}.wav")
        save_audio(final_wav_path, arrangement["final_mix"], sr=arranger.sample_rate)

        # Compute full spectrogram
        full_spec = compute_stft_spectrogram(arrangement["final_mix"], sr=arranger.sample_rate)
        full_grid = spectrogram_to_grid(full_spec, max_freq_bins=64, max_time_steps=200)

        # 6. Complete Job
        job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress_pct=100,
            message="Composition complete!",
            audio_path=final_wav_path,
            spectrogram_data=full_grid,
            chunks=chunks,
            predictions=predictions,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            progress_pct=100,
            error=str(e),
            message=f"Generation failed: {str(e)}",
        )


@router.post("/generate")
async def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks):
    """Starts background music generation job."""
    if not os.path.exists(req.audio_path):
        raise HTTPException(status_code=404, detail="Source audio file not found")

    job = job_manager.create_job(original_audio_path=req.audio_path)
    background_tasks.add_task(run_generation_task, job.job_id, req)

    return {"status": "started", "job_id": job.job_id}


@router.get("/generate/{job_id}")
async def get_generation_status(job_id: str):
    """Retrieve generation progress, chunks, and metadata."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "message": job.message,
        "chunks_count": len(job.chunks),
        "chunks": job.chunks,
        "predictions": job.predictions,
        "has_audio": job.audio_path is not None,
        "error": job.error,
    }


@router.get("/audio/{job_id}")
async def get_audio_file(job_id: str):
    """Stream or download completed audio composition."""
    job = job_manager.get_job(job_id)
    if not job or not job.audio_path or not os.path.exists(job.audio_path):
        raise HTTPException(status_code=404, detail="Audio file not ready or not found")

    return FileResponse(
        job.audio_path,
        media_type="audio/wav",
        filename=f"charlie_composition_{job_id}.wav",
    )


@router.get("/spectrogram/{job_id}")
async def get_spectrogram_data(job_id: str):
    """Retrieve the generated STFT spectrogram grid."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.job_id,
        "spectrogram": job.spectrogram_data or [],
    }
