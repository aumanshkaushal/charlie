"""Background job tracker for Charlie music generation tasks."""
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    ADAPTING = "adapting"
    ARRANGING = "arranging"
    PROGRESSIVE_RENDERING = "progressive_rendering"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationJob:
    job_id: str
    status: JobStatus
    created_at: float = field(default_factory=time.time)
    progress_pct: int = 0
    message: str = "Initialized"
    audio_path: Optional[str] = None
    original_audio_path: Optional[str] = None
    spectrogram_data: Optional[list] = None
    chunks: list = field(default_factory=list)
    predictions: Optional[dict] = None
    error: Optional[str] = None


class JobManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.jobs = {}
        return cls._instance

    def create_job(self, original_audio_path: str = None) -> GenerationJob:
        job_id = str(uuid.uuid4())[:8]
        job = GenerationJob(
            job_id=job_id,
            status=JobStatus.QUEUED,
            original_audio_path=original_audio_path,
        )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        return self.jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress_pct: Optional[int] = None,
        message: Optional[str] = None,
        audio_path: Optional[str] = None,
        spectrogram_data: Optional[list] = None,
        chunks: Optional[list] = None,
        predictions: Optional[dict] = None,
        error: Optional[str] = None,
    ):
        job = self.get_job(job_id)
        if not job:
            return
        if status is not None:
            job.status = status
        if progress_pct is not None:
            job.progress_pct = progress_pct
        if message is not None:
            job.message = message
        if audio_path is not None:
            job.audio_path = audio_path
        if spectrogram_data is not None:
            job.spectrogram_data = spectrogram_data
        if chunks is not None:
            job.chunks = chunks
        if predictions is not None:
            job.predictions = predictions
        if error is not None:
            job.error = error
