from __future__ import annotations
import gc
import time
from pathlib import Path
from typing import Any
import torch
import whisper
from config import configure_ffmpeg, WHISPER_MODEL
from logger_config import logger, log_step

# ============================================================
# DEVICE SELECTION
# ============================================================

def get_processing_device(device: str) -> str:
    """Resolve and validate the requested processing device."""

    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            f"Unsupported processing device: {device}"
        )

    if device == "cpu":
        logger.info("CPU processing selected.")
        return "cpu"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

        logger.info("CUDA available: %s", gpu_name)

        return "cuda"

    if device == "cuda":
        raise RuntimeError(
            "CUDA was explicitly requested but is unavailable. "
            "Check your NVIDIA drivers and CUDA-enabled PyTorch."
        )

    logger.warning(
        "CUDA is unavailable. Using CPU because device='auto'."
    )

    return "cpu"


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio_with_whisper(
    file_path: str,
    device: str = "cuda",
) -> tuple[list[dict[str, Any]], str]:
    """
    Transcribe video audio using OpenAI Whisper.

    Model:
        base (fixed)

    Device:
        cuda by default
        auto for automatic CPU fallback
        cpu for CPU-only processing
    """

    log_step(
        "WHISPER",
        "Starting OpenAI Whisper transcription",
    )

    model = None

    try:
        # ----------------------------------------------------
        # FFMPEG VALIDATION
        # ----------------------------------------------------

        configure_ffmpeg()

        # ----------------------------------------------------
        # INPUT VALIDATION
        # ----------------------------------------------------

        video_path = Path(file_path)

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Input video not found: {video_path}"
            )

        logger.info("Input file: %s", video_path)
        logger.info("Whisper model: %s", WHISPER_MODEL)
        logger.info("Requested device: %s", device)

        # ----------------------------------------------------
        # DEVICE SELECTION
        # ----------------------------------------------------

        selected_device = get_processing_device(device)

        if selected_device == "cuda":

            gpu_properties = torch.cuda.get_device_properties(0)

            logger.info(
                "GPU: %s",
                gpu_properties.name,
            )

            logger.info(
                "GPU VRAM: %.2f GB",
                gpu_properties.total_memory / (1024 ** 3),
            )

            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # LOAD WHISPER MODEL
        # ----------------------------------------------------

        logger.info(
            "Loading Whisper base model on %s...",
            selected_device,
        )

        load_start = time.perf_counter()

        model = whisper.load_model(
            WHISPER_MODEL,
            device=selected_device,
        )

        load_time = time.perf_counter() - load_start

        logger.info(
            "Whisper model loaded successfully in %.2f seconds.",
            load_time,
        )

        # ----------------------------------------------------
        # TRANSCRIBE VIDEO
        # ----------------------------------------------------

        logger.info(
            "Starting transcription on %s...",
            selected_device,
        )

        transcription_start = time.perf_counter()

        transcript = model.transcribe(
            str(video_path),
            fp16=(selected_device == "cuda"),
            verbose=False,
            word_timestamps=False,
        )

        transcription_time = (
            time.perf_counter() - transcription_start
        )

        logger.info(
            "Transcription completed in %.2f seconds.",
            transcription_time,
        )

        # ----------------------------------------------------
        # EXTRACT TIMESTAMPED SEGMENTS
        # ----------------------------------------------------

        rows: list[dict[str, Any]] = []

        segments = transcript.get("segments", [])

        for segment in segments:

            if not isinstance(segment, dict):
                continue

            try:
                text = str(
                    segment.get("text") or ""
                ).strip()

                start = float(segment["start"])
                end = float(segment["end"])

            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Skipping malformed transcript segment."
                )
                continue

            if not text:
                continue

            if start < 0 or end <= start:
                logger.warning(
                    "Skipping invalid segment: %.2f - %.2f",
                    start,
                    end,
                )
                continue

            rows.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

        # ----------------------------------------------------
        # EXTRACT METADATA
        # ----------------------------------------------------

        language = str(
            transcript.get("language") or "unknown"
        )

        transcript_text = str(
            transcript.get("text") or ""
        )

        logger.info(
            "Detected language: %s",
            language,
        )

        logger.info(
            "Transcript segments: %d",
            len(rows),
        )

        logger.info(
            "Transcript characters: %d",
            len(transcript_text),
        )

        logger.info(
            "Active transcription device: %s",
            selected_device,
        )

        if not rows:
            raise RuntimeError(
                "No speech detected in the video."
            )

        # ----------------------------------------------------
        # GPU MEMORY INFORMATION
        # ----------------------------------------------------

        if selected_device == "cuda":

            allocated_memory = (
                torch.cuda.memory_allocated(0)
                / (1024 ** 3)
            )

            reserved_memory = (
                torch.cuda.memory_reserved(0)
                / (1024 ** 3)
            )

            logger.info(
                "GPU allocated memory: %.2f GB",
                allocated_memory,
            )

            logger.info(
                "GPU reserved memory: %.2f GB",
                reserved_memory,
            )

        logger.info(
            "Whisper transcription completed successfully."
        )

        return rows, language

    except Exception:

        logger.exception(
            "Whisper transcription failed."
        )

        raise

    finally:

        if model is not None:
            del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
