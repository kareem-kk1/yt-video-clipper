
from logger_config import log_step, logger
import time
import re
import zipfile
import subprocess
from pathlib import Path

FFMPEG_PATH = Path(
    r"C:\ffmpeg\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
)

def candidate_windows(segments, duration, clip_len):
    log_step("CANDIDATES", "Generating candidate clip windows")
    start_time = time.perf_counter()
    logger.info("Video duration: %.2f seconds", duration)
    logger.info("Clip length: %d seconds", clip_len)
    logger.info("Transcript segments: %d", len(segments))
    starts = set()
    step = max(8, clip_len // 3)
    t = 0

    while t + clip_len <= duration:
        starts.add(float(t))
        t += step
        
    for segment in segments:
        if segment["start"] + clip_len <= duration:
            starts.add(
                max(
                    0.0,
                    min(
                        float(segment["start"]),
                        duration - clip_len
                    )
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
        "first"
    ]

    result = []

    for start in sorted(starts):
        end = start + clip_len
        selected = [
            s for s in segments
            if s["end"] > start and s["start"] < end
        ]

        text = " ".join(s["text"] for s in selected).strip()

        words = re.findall(r"\w+",text.lower())

        if len(words) < 18:
            continue

        density = len(words) / clip_len

        hook_score = sum(
            1 for word in hooks
            if re.search(
                r"\b" + re.escape(word) + r"\b",
                text.lower()
            )
        )

        last = selected[-1]["text"] if selected else ""

        completeness = (
            1.0 if re.search(r"[.!?]$", last) else 0.25
        )

        score = (
            density
            + hook_score * 0.18
            + completeness * 0.35
        )

        result.append({
            "start": start,
            "end": end,
            "text": text,
            "score": score
        })

    result = sorted(
        result,
        key=lambda x: x["score"],
        reverse=True
    )

    elapsed = time.perf_counter() - start_time

    logger.info("Candidate windows generated: %d", len(result))
    logger.info("Top candidate score: %.4f", result[0]["score"] if result else 0)
    logger.info("Candidate generation time: %.2f seconds", elapsed)

    return result


def ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def make_ass(segments, start, end, path, style):
    log_step("SUBTITLES", f"Creating ASS subtitles: {path.name}")
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
PlayResX: 1080
PlayResY: 1920
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
            0,
            segment["start"] - start
        )

        b = min(
            end - start,
            segment["end"] - start
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

        duration = (b - a) / len(chunks)

        for j, chunk in enumerate(chunks):
            lines.append(
                f"Dialogue: 0,"
                f"{ass_time(a + j * duration)},"
                f"{ass_time(a + (j + 1) * duration)},"
                f"Default,,0,0,0,,{chunk}\n"
            )

            subtitle_count += 1
            
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(lines),
        encoding="utf-8"
    )

    elapsed = time.perf_counter() - subtitle_start

    logger.info("ASS subtitle file created: %s", path)
    logger.info("Subtitle entries: %d", subtitle_count)
    logger.info("Subtitle generation time: %.2f seconds", elapsed)

def srt_time(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments, start, end, path):
    log_step("SRT", f"Writing subtitle file: {path.name}")
    lines = []
    n = 1

    for segment in segments:
        a = max(0, segment["start"] - start)
        b = min(end - start,segment["end"] - start)

        if b <= a:
            continue

        lines.extend([
            str(n),
            f"{srt_time(a)} --> {srt_time(b)}",
            segment["text"],
            ""
        ])

        n += 1

    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("SRT file created: %s", path)
    logger.info("SRT subtitle entries: %d", n - 1)



def run_ffmpeg(video, start, length, ass_file, out_file):
    """Render a vertical Shorts clip using NVIDIA NVENC."""

    log_step("FFMPEG", f"Rendering {out_file.name}")
    start_time = time.perf_counter()

    video = Path(video).resolve()
    ass_file = Path(ass_file).resolve()
    out_file = Path(out_file).resolve()

    logger.info("FFmpeg executable: %s", FFMPEG_PATH)
    logger.info("Input video: %s", video)
    logger.info("Start time: %.2f seconds", start)
    logger.info("Clip length: %.2f seconds", length)
    logger.info("Subtitle file: %s", ass_file)
    logger.info("Output file: %s", out_file)

    # Validate required files and paths.
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
            "Start time must be non-negative and clip length must be positive."
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Escape Windows paths for FFmpeg's ASS filter.
    ass_path = (
        str(ass_file)
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )

    video_filter = (
        "split=2[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=30:10[bg2];"
        "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fg2];"
        "[bg2][fg2]overlay=(W-w)/2:(H-h)/2,"
        f"ass='{ass_path}'"
    )

    command = [
        str(FFMPEG_PATH),
        "-hide_banner",
        "-y",
        "-nostdin",
        "-ss", str(start),
        "-i", str(video),
        "-t", str(length),
        "-vf", video_filter,

        # NVIDIA GPU video encoding.
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "vbr",
        "-cq", "23",
        "-b:v", "0",
        "-pix_fmt", "yuv420p",

        # Audio encoding.
        "-c:a", "aac",
        "-b:a", "128k",

        "-movflags", "+faststart",
        str(out_file)
    ]

    try:
        logger.info("Starting FFmpeg NVIDIA NVENC rendering")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            error_message = result.stderr[-3000:]

            logger.error(
                "FFmpeg return code: %d",
                result.returncode
            )
            logger.error("FFmpeg stderr: %s", error_message)

            raise RuntimeError(
                "FFmpeg NVENC rendering failed. "
                f"Check NVIDIA driver and encoder compatibility.\n"
                f"Details: {error_message[-1800:]}"
            )

        if not out_file.is_file():
            raise RuntimeError(
                "FFmpeg completed but output video was not created."
            )

        elapsed = time.perf_counter() - start_time
        output_size = out_file.stat().st_size / (1024 * 1024)

        logger.info(
            "Video rendered successfully: %s",
            out_file
        )
        logger.info("Encoder used: h264_nvenc")
        logger.info("Output size: %.2f MB", output_size)
        logger.info("Rendering time: %.2f seconds", elapsed)

    except Exception:
        logger.exception("FFmpeg rendering failed")
        raise

    
def create_zip_bundle(output_job):
    log_step("ZIP", "Creating Shorts ZIP bundle")
    start_time = time.perf_counter()
    zip_path = output_job / "youtube_shorts_bundle.zip"

    try:
        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED
        ) as archive:

            for file in output_job.iterdir():
                if (
                    file.is_file()
                    and file.name != zip_path.name
                ):

                    archive.write(
                        file,
                        file.name
                    )

                    logger.info("Added to ZIP: %s",file.name)

        elapsed = time.perf_counter() - start_time
        logger.info("ZIP bundle created: %s", zip_path)
        logger.info("ZIP size: %.2f MB", zip_path.stat().st_size / (1024 * 1024))
        logger.info("ZIP creation time: %.2f seconds", elapsed)

        return zip_path

    except Exception:
        logger.exception("ZIP creation failed")
        raise
