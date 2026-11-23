
from __future__ import annotations
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any
from logger_config import log_step, logger
from config import (
    FFMPEG_PATH,
    VIDEO_CODEC,
    VIDEO_PRESET,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
)


# ============================================================
# CANDIDATE WINDOWS
# ============================================================

def candidate_windows(
    segments: list[dict[str, Any]],
    duration: float,
    clip_len: int,
) -> list[dict[str, Any]]:

    log_step(
        "CANDIDATES",
        "Generating candidate clip windows",
    )

    start_time = time.perf_counter()

    if duration <= 0 or clip_len <= 0:
        raise ValueError(
            "Video duration and clip length must be positive."
        )

    logger.info("Video duration: %.2f seconds", duration)
    logger.info("Clip length: %d seconds", clip_len)
    logger.info("Transcript segments: %d", len(segments))

    if duration < clip_len:
        raise ValueError(
            "Video duration is shorter than the requested clip length."
        )

    starts: set[float] = set()

    step = max(8, clip_len // 3)
    current_time = 0

    while current_time + clip_len <= duration:
        starts.add(float(current_time))
        current_time += step

    for segment in segments:

        segment_start = float(segment["start"])

        if segment_start + clip_len <= duration:

            starts.add(
                max(
                    0.0,
                    min(
                        segment_start,
                        duration - clip_len,
                    ),
                )
            )

    hooks = [
        "how",
        "why",
        "secret",
        "mistake",
        "important",
        "never",
        "best",
        "worst",
        "problem",
        "solution",
        "learn",
        "surprising",
        "actually",
        "because",
        "first",
    ]

    result: list[dict[str, Any]] = []

    for start in sorted(starts):

        end = start + clip_len

        selected = [
            segment
            for segment in segments
            if (
                segment["end"] > start
                and segment["start"] < end
            )
        ]

        text = " ".join(
            segment["text"]
            for segment in selected
        ).strip()

        words = re.findall(
            r"\w+",
            text.lower(),
        )

        if len(words) < 18:
            continue

        density = len(words) / clip_len

        hook_score = sum(
            1
            for word in hooks
            if re.search(
                r"\b" + re.escape(word) + r"\b",
                text.lower(),
            )
        )

        last = (
            selected[-1]["text"]
            if selected
            else ""
        )

        completeness = (
            1.0
            if re.search(r"[.!?]$", last)
            else 0.25
        )

        score = (
            density
            + hook_score * 0.18
            + completeness * 0.35
        )

        result.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "score": score,
            }
        )

    result.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Candidate windows generated: %d",
        len(result),
    )

    logger.info(
        "Top candidate score: %.4f",
        result[0]["score"] if result else 0,
    )

    logger.info(
        "Candidate generation time: %.2f seconds",
        elapsed,
    )

    return result


# ============================================================
# SUBTITLE TIME FORMATTING
# ============================================================

def ass_time(seconds: float) -> str:

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    return f"{h}:{m:02d}:{s:05.2f}"


def srt_time(seconds: float) -> str:

    ms = int(round(seconds * 1000))

    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)

    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# ============================================================
# ASS SUBTITLE GENERATION
# ============================================================

def make_ass(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
    path: Path,
    style: str,
) -> None:

    path = Path(path)

    log_step(
        "SUBTITLES",
        f"Creating ASS subtitles: {path.name}",
    )

    subtitle_start = time.perf_counter()

    if style == "Yellow highlight":

        primary = "&H0000D7FF"
        outline = "&H00000000"
        size = 58

    elif style == "Minimal":

        primary = "&H00FFFFFF"
        outline = "&H80000000"
        size = 46

    else:

        primary = "&H00FFFFFF"
        outline = "&H00000000"
        size = 58

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{size},{primary},&H000000FF,{outline},&H80000000,1,0,1,3,1,2,80,80,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    subtitle_count = 0

    for segment in segments:

        a = max(
            0.0,
            segment["start"] - start,
        )

        b = min(
            end - start,
            segment["end"] - start,
        )

        if b <= a:
            continue

        txt = (
            segment["text"]
            .replace("\\", "")
            .replace("{", "")
            .replace("}", "")
            .replace("\n", " ")
        )

        words = txt.split()

        chunks = [
            " ".join(words[i:i + 9])
            for i in range(0, len(words), 9)
        ]

        if not chunks:
            continue

        chunk_duration = (b - a) / len(chunks)

        for index, chunk in enumerate(chunks):

            lines.append(
                f"Dialogue: 0,"
                f"{ass_time(a + index * chunk_duration)},"
                f"{ass_time(a + (index + 1) * chunk_duration)},"
                f"Default,,0,0,0,,{chunk}\n"
            )

            subtitle_count += 1

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "".join(lines),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - subtitle_start

    logger.info(
        "ASS subtitle file created: %s",
        path,
    )

    logger.info(
        "Subtitle entries: %d",
        subtitle_count,
    )

    logger.info(
        "Subtitle generation time: %.2f seconds",
        elapsed,
    )


# ============================================================
# SRT SUBTITLE GENERATION
# ============================================================

def write_srt(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
    path: Path,
) -> None:

    path = Path(path)

    log_step(
        "SRT",
        f"Writing subtitle file: {path.name}",
    )

    lines: list[str] = []
    number = 1

    for segment in segments:

        a = max(
            0.0,
            segment["start"] - start,
        )

        b = min(
            end - start,
            segment["end"] - start,
        )

        if b <= a:
            continue

        lines.extend(
            [
                str(number),
                f"{srt_time(a)} --> {srt_time(b)}",
                segment["text"],
                "",
            ]
        )

        number += 1

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info(
        "SRT file created: %s",
        path,
    )

    logger.info(
        "SRT subtitle entries: %d",
        number - 1,
    )


# ============================================================
# FFMPEG VIDEO RENDERING
# ============================================================

def run_ffmpeg(
    video: Path,
    start: float,
    length: float,
    ass_file: Path,
    out_file: Path,
) -> None:
    """Render a vertical Shorts clip using configured encoder."""

    log_step(
        "FFMPEG",
        f"Rendering {Path(out_file).name}",
    )

    render_start = time.perf_counter()

    video = Path(video).resolve()
    ass_file = Path(ass_file).resolve()
    out_file = Path(out_file).resolve()

    logger.info(
        "FFmpeg executable: %s",
        FFMPEG_PATH,
    )

    logger.info("Input video: %s", video)
    logger.info("Start time: %.2f seconds", start)
    logger.info("Clip length: %.2f seconds", length)
    logger.info("Subtitle file: %s", ass_file)
    logger.info("Output file: %s", out_file)

    # --------------------------------------------------------
    # VALIDATE FILES
    # --------------------------------------------------------

    if not FFMPEG_PATH.is_file():

        raise FileNotFoundError(
            f"FFmpeg executable not found: {FFMPEG_PATH}"
        )

    if not video.is_file():

        raise FileNotFoundError(
            f"Input video not found: {video}"
        )

    if not ass_file.is_file():

        raise FileNotFoundError(
            f"ASS subtitle file not found: {ass_file}"
        )

    if start < 0 or length <= 0:

        raise ValueError(
            "Start time must be non-negative and "
            "clip length must be positive."
        )

    out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # ESCAPE WINDOWS PATH FOR ASS FILTER
    # --------------------------------------------------------

    ass_path = (
        str(ass_file)
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )

    width = VIDEO_WIDTH
    height = VIDEO_HEIGHT

    video_filter = (
        f"split=2[bg][fg];"
        f"[bg]scale={width}:{height}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"boxblur=30:10[bg2];"
        f"[fg]scale={width}:{height}:"
        f"force_original_aspect_ratio=decrease[fg2];"
        f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2,"
        f"ass='{ass_path}'"
    )

    # --------------------------------------------------------
    # FFMPEG COMMAND
    # --------------------------------------------------------

    command = [
        str(FFMPEG_PATH),
        "-hide_banner",
        "-y",
        "-nostdin",
        "-ss",
        str(start),
        "-i",
        str(video),
        "-t",
        str(length),
        "-vf",
        video_filter,

        # Video encoding.
        "-c:v",
        VIDEO_CODEC,
        "-preset",
        VIDEO_PRESET,
        "-rc",
        "vbr",
        "-cq",
        "23",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(VIDEO_FPS),

        # Audio encoding.
        "-c:a",
        "aac",
        "-b:a",
        "128k",

        "-movflags",
        "+faststart",
        str(out_file),
    ]

    # --------------------------------------------------------
    # EXECUTE FFMPEG
    # --------------------------------------------------------

    try:

        logger.info(
            "Starting FFmpeg rendering with encoder: %s",
            VIDEO_CODEC,
        )

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:

            error_message = result.stderr[-3000:]

            logger.error(
                "FFmpeg return code: %d",
                result.returncode,
            )

            logger.error(
                "FFmpeg stderr: %s",
                error_message,
            )

            raise RuntimeError(
                "FFmpeg rendering failed. "
                "Check the configured video encoder and "
                "your FFmpeg build.\n"
                f"Details: {error_message[-1800:]}"
            )

        if (
            not out_file.is_file()
            or out_file.stat().st_size == 0
        ):

            raise RuntimeError(
                "FFmpeg completed but output video "
                "was not created or is empty."
            )

        elapsed = time.perf_counter() - render_start

        output_size = (
            out_file.stat().st_size / (1024 * 1024)
        )

        logger.info(
            "Video rendered successfully: %s",
            out_file,
        )

        logger.info(
            "Encoder used: %s",
            VIDEO_CODEC,
        )

        logger.info(
            "Output size: %.2f MB",
            output_size,
        )

        logger.info(
            "Rendering time: %.2f seconds",
            elapsed,
        )

    except Exception:

        logger.exception(
            "FFmpeg rendering failed."
        )

        raise


# ============================================================
# ZIP BUNDLE
# ============================================================

def create_zip_bundle(
    output_job: Path,
) -> Path:

    output_job = Path(output_job)

    log_step(
        "ZIP",
        "Creating Shorts ZIP bundle",
    )

    start_time = time.perf_counter()

    zip_path = (
        output_job / "youtube_shorts_bundle.zip"
    )

    try:

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as archive:

            for file in output_job.iterdir():

                if (
                    file.is_file()
                    and file.name != zip_path.name
                ):

                    archive.write(
                        file,
                        file.name,
                    )

                    logger.info(
                        "Added to ZIP: %s",
                        file.name,
                    )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "ZIP bundle created: %s",
            zip_path,
        )

        logger.info(
            "ZIP size: %.2f MB",
            zip_path.stat().st_size / (1024 * 1024),
        )

        logger.info(
            "ZIP creation time: %.2f seconds",
            elapsed,
        )

        return zip_path

    except Exception:

        logger.exception(
            "ZIP creation failed."
        )

        raise
