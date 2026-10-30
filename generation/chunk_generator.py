"""Incremental progressive chunk generator for Charlie."""
import numpy as np
from audio.spectrogram import compute_stft_spectrogram, spectrogram_to_grid


class IncrementalChunkGenerator:
    """
    Manages progressive music generation and real-time growing spectrogram frames.
    """
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def generate_progressive_chunks(
        self,
        full_arrangement: dict,
        chunk_duration_bars: int = 1,
    ) -> list[dict]:
        """
        Subdivides the arranged composition into sequential chunks,
        computing spectrogram slices for each chunk to simulate streaming generation.
        """
        final_mix = full_arrangement["final_mix"]
        total_duration = full_arrangement["duration"]
        seconds_per_bar = full_arrangement["seconds_per_bar"]
        total_bars = full_arrangement["total_bars"]

        chunk_seconds = seconds_per_bar * chunk_duration_bars
        num_chunks = int(np.ceil(total_duration / chunk_seconds))

        chunks = []
        cumulative_audio = np.zeros(0, dtype=np.float32)

        for i in range(num_chunks):
            start_time = round(i * chunk_seconds, 2)
            end_time = round(min((i + 1) * chunk_seconds, total_duration), 2)

            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)

            chunk_audio = final_mix[start_sample:end_sample]
            cumulative_audio = np.concatenate([cumulative_audio, chunk_audio])

            # Active instruments in this section
            current_bar = i * chunk_duration_bars
            active_insts = ["Original Mug Beat"]
            if current_bar >= 1:
                active_insts.append("Tuned Mug Melody")
            if current_bar >= 2:
                active_insts.append("Mug Sub-Bass")
            if current_bar >= 3:
                active_insts.append("Mug Drums (Kick/Snare/Hi-Hat)")

            # Compute spectrogram of this specific chunk
            chunk_spec = compute_stft_spectrogram(chunk_audio, sr=self.sample_rate)
            chunk_grid = spectrogram_to_grid(chunk_spec, max_freq_bins=64, max_time_steps=80)

            # Cumulative growing spectrogram
            full_spec = compute_stft_spectrogram(cumulative_audio, sr=self.sample_rate)
            full_grid = spectrogram_to_grid(full_spec, max_freq_bins=64, max_time_steps=180)

            chunks.append({
                "chunk_index": i,
                "total_chunks": num_chunks,
                "start_time": start_time,
                "end_time": end_time,
                "duration": round(end_time - start_time, 2),
                "active_instruments": active_insts,
                "chunk_spectrogram": chunk_grid,
                "cumulative_spectrogram": full_grid,
                "audio_samples_len": len(chunk_audio),
            })

        return chunks
