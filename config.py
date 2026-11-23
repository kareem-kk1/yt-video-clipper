
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = APP_DIR / "outputs"
WORK_DIR = APP_DIR / "work"
LOG_DIR = APP_DIR / "logs"

COOKIE_FILE = APP_DIR / "cookies.txt"


# ============================================================
# FFMPEG CONFIGURATION
# ============================================================

FFMPEG_DIR = Path(
    r"C:\ffmpeg\ffmpeg-7.1.1-full_build\bin"
)

FFMPEG_PATH = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE_PATH = FFMPEG_DIR / "ffprobe.exe"


def configure_ffmpeg() -> str:
    """
    Add FFmpeg directory to the current Python PATH.

    Returns the detected FFmpeg executable path.
    """

    if not FFMPEG_PATH.is_file():
        raise FileNotFoundError(
            f"FFmpeg not found: {FFMPEG_PATH}"
        )

    if not FFPROBE_PATH.is_file():
        raise FileNotFoundError(
            f"FFprobe not found: {FFPROBE_PATH}"
        )

    ffmpeg_directory = str(FFMPEG_DIR.resolve())

    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep)

    normalized_entries = {
        os.path.normcase(os.path.normpath(entry))
        for entry in path_entries
        if entry
    }

    normalized_directory = os.path.normcase(
        os.path.normpath(ffmpeg_directory)
    )

    if normalized_directory not in normalized_entries:
        os.environ["PATH"] = (
            ffmpeg_directory
            + os.pathsep
            + current_path
        )

    detected_ffmpeg = shutil.which("ffmpeg")

    if not detected_ffmpeg:
        raise RuntimeError(
            "FFmpeg is not accessible through Python PATH."
        )

    return detected_ffmpeg


def check_ffmpeg() -> bool:
    """Check whether FFmpeg and FFprobe are working."""

    try:
        configure_ffmpeg()

        for executable in (FFMPEG_PATH, FFPROBE_PATH):
            subprocess.run(
                [str(executable), "-version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )

        return True

    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ):
        return False


# ============================================================
# WHISPER CONFIGURATION
# ============================================================

WHISPER_MODEL = "base"

PROCESSING_DEVICES = [
    "cuda",
    "auto",
    "cpu",
]

DEFAULT_PROCESSING_DEVICE = "cuda"


# ============================================================
# VIDEO GENERATION SETTINGS
# ============================================================

CLIP_COUNTS = [1, 3, 5, 7]
DEFAULT_CLIP_COUNT = 1

CLIP_LENGTHS = [30, 45, 60]
DEFAULT_CLIP_LENGTH = 45

SUBTITLE_STYLES = [
    "Bold white",
    "Yellow highlight",
    "Minimal",
]

DEFAULT_SUBTITLE_STYLE = "Bold white"


# ============================================================
# VIDEO RENDERING SETTINGS
# ============================================================

VIDEO_CODEC = "h264_nvenc"
VIDEO_PRESET = "p4"

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30

VIDEO_AUDIO_CODEC = "aac"
VIDEO_AUDIO_BITRATE = "128k"

VIDEO_CRF = "23"


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

def create_project_directories() -> None:
    """Create required project directories."""

    for directory in (
        OUTPUT_DIR,
        WORK_DIR,
        LOG_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def initialize_config() -> None:
    """
    Initialize project directories and validate FFmpeg.
    Call this once when starting the application.
    """

    create_project_directories()

    detected_ffmpeg = configure_ffmpeg()

    print(f"FFmpeg configured: {detected_ffmpeg}")


# ============================================================
# MODULE INITIALIZATION
# ============================================================

# Create folders when config is imported.
# FFmpeg is configured explicitly through initialize_config()
# or check_ffmpeg(), not automatically on every import.

create_project_directories()
