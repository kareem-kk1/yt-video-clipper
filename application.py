
from __future__ import annotations

import json
import shutil
import uuid
import zipfile

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from yt_downloader import download_video
from transcript import transcribe_audio_with_whisper
from llama_call import choose_moments

from clip_window import (
    candidate_windows,
    make_ass,
    write_srt,
    run_ffmpeg,
)

from logger_config import logger

from config import (
    OUTPUT_DIR,
    WORK_DIR,
    WHISPER_MODEL,
    PROCESSING_DEVICES,
    DEFAULT_PROCESSING_DEVICE,
    CLIP_COUNTS,
    DEFAULT_CLIP_COUNT,
    CLIP_LENGTHS,
    DEFAULT_CLIP_LENGTH,
    SUBTITLE_STYLES,
    DEFAULT_SUBTITLE_STYLE,
    check_ffmpeg,
    create_project_directories,
)


# ============================================================
# INITIALIZATION
# ============================================================

st.set_page_config(
    page_title="AI YouTube Shorts Generator",
    page_icon="🎬",
    layout="wide",
)

create_project_directories()

st.title("🎬 AI YouTube Shorts Generator")
st.caption(
    "Turn long YouTube videos into engaging short clips using AI."
)


# ============================================================
# SESSION STATE
# ============================================================

SESSION_DEFAULTS: dict[str, Any] = {
    "cached_url": None,
    "cached_job": None,
    "cached_video": None,
    "cached_info": None,
    "cached_transcription_config": None,
    "cached_transcript_path": None,
    "latest_clips": [],
    "latest_zip": None,
    "latest_title": None,
    "latest_language": None,
    "latest_duration": 0.0,
}

for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_processing_device(device: str) -> None:
    """Validate the selected processing device."""

    if device not in {"cuda", "auto", "cpu"}:
        raise ValueError(
            f"Unsupported processing device: {device}"
        )

    if device == "cpu":
        logger.info("CPU processing selected.")
        return

    try:
        import torch

    except ImportError as exc:

        if device == "cuda":
            raise RuntimeError(
                "PyTorch is not installed. Install CUDA-enabled "
                "PyTorch to use GPU processing."
            ) from exc

        logger.warning(
            "PyTorch is unavailable. Automatic device selection "
            "will be handled by the transcription module."
        )
        return

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        logger.info(
            "CUDA available. GPU: %s",
            gpu_name,
        )

    elif device == "cuda":

        raise RuntimeError(
            "CUDA is unavailable. Check your NVIDIA drivers "
            "and CUDA-enabled PyTorch installation."
        )

    else:

        logger.warning(
            "CUDA unavailable. Automatic device selection "
            "will use CPU."
        )


def update_progress(
    progress_bar: Any,
    percentage: float,
    message: str,
) -> None:
    """Update the single overall progress bar."""

    percentage = max(
        0.0,
        min(100.0, percentage),
    )

    progress_bar.progress(
        int(percentage),
        text=message,
    )


def validate_video_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize downloaded video metadata."""

    title = str(
        metadata.get("title") or "YouTube video"
    ).strip()

    try:
        duration = float(
            metadata.get("duration") or 0
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid video duration returned by YouTube."
        ) from exc

    if duration <= 0:
        raise ValueError(
            "YouTube returned an invalid video duration."
        )

    return {
        "title": title,
        "duration": duration,
    }


# ============================================================
# VIDEO CACHE
# ============================================================

def get_cached_video(
    video_url: str,
    download_progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Reuse an existing downloaded video when the URL matches.

    Otherwise, download a new video into a unique working directory.
    """

    cached_url = st.session_state.get("cached_url")
    cached_video = st.session_state.get("cached_video")
    cached_job = st.session_state.get("cached_job")
    cached_info = st.session_state.get("cached_info")

    # --------------------------------------------------------
    # REUSE CACHED VIDEO
    # --------------------------------------------------------

    if cached_url == video_url and cached_video and cached_job:

        video_path = Path(cached_video)
        job_path = Path(cached_job)

        if (
            video_path.is_file()
            and job_path.is_dir()
            and isinstance(cached_info, dict)
        ):

            logger.info(
                "Reusing cached video: %s",
                video_path,
            )

            if download_progress_callback:
                download_progress_callback(100.0)

            return video_path, job_path, cached_info

    # --------------------------------------------------------
    # DOWNLOAD NEW VIDEO
    # --------------------------------------------------------

    logger.info("Downloading a new YouTube video.")

    job_path = WORK_DIR / uuid.uuid4().hex

    job_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        video_path, download_info = download_video(
            video_url=video_url,
            folder=job_path,
            progress_callback=download_progress_callback,
        )

        video_path = Path(video_path)

        if not video_path.is_file():

            raise FileNotFoundError(
                f"Downloaded video was not found: {video_path}"
            )

        if not isinstance(download_info, Mapping):

            raise TypeError(
                "Video metadata must be a valid mapping."
            )

        info = validate_video_metadata(
            download_info
        )

        # ----------------------------------------------------
        # SAVE VIDEO CACHE
        # ----------------------------------------------------

        st.session_state["cached_url"] = video_url
        st.session_state["cached_job"] = str(job_path)
        st.session_state["cached_video"] = str(video_path)
        st.session_state["cached_info"] = info

        # Reset transcript cache for the new video.
        st.session_state["cached_transcription_config"] = None
        st.session_state["cached_transcript_path"] = None

        # Clear previous generated results.
        st.session_state["latest_clips"] = []
        st.session_state["latest_zip"] = None
        st.session_state["latest_title"] = None
        st.session_state["latest_language"] = None
        st.session_state["latest_duration"] = 0.0

        logger.info(
            "Video downloaded and cached: %s",
            video_path,
        )

        return video_path, job_path, info

    except Exception:

        # Remove only the incomplete new download directory.
        shutil.rmtree(
            job_path,
            ignore_errors=True,
        )

        raise


# ============================================================
# TRANSCRIPT VALIDATION
# ============================================================

def validate_transcript(
    transcript: Any,
) -> list[dict[str, Any]]:
    """Validate transcript segments and timestamps."""

    if not isinstance(transcript, list):

        raise ValueError(
            "Transcript must be a list of segments."
        )

    valid_segments: list[dict[str, Any]] = []

    for segment in transcript:

        if not isinstance(segment, dict):
            continue

        try:

            start = float(segment["start"])
            end = float(segment["end"])

            text = str(
                segment.get("text") or ""
            ).strip()

        except (KeyError, TypeError, ValueError):
            continue

        if (
            start < 0
            or end <= start
            or not text
        ):
            continue

        valid_segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    if not valid_segments:

        raise ValueError(
            "No valid transcript segments were found."
        )

    return valid_segments


# ============================================================
# TRANSCRIPT CACHE
# ============================================================

def get_cached_transcript(
    video: Path,
    job: Path,
    device: str,
) -> tuple[list[dict[str, Any]], str]:
    """
    Reuse transcript when the model and processing device match.

    Whisper model is always fixed to base.
    """

    transcript_path = job / "transcript.json"
    metadata_path = job / "transcript_metadata.json"

    transcription_config = (
        WHISPER_MODEL,
        device,
    )

    cached_config = st.session_state.get(
        "cached_transcription_config"
    )

    cached_path = st.session_state.get(
        "cached_transcript_path"
    )

    # --------------------------------------------------------
    # REUSE CACHED TRANSCRIPT
    # --------------------------------------------------------

    if (
        cached_config == transcription_config
        and cached_path == str(transcript_path)
        and transcript_path.is_file()
        and metadata_path.is_file()
    ):

        try:

            with transcript_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                transcript_data = json.load(file)

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(file)

            if not isinstance(metadata, Mapping):

                raise ValueError(
                    "Invalid transcript metadata."
                )

            if (
                metadata.get("model_size") != WHISPER_MODEL
                or metadata.get("device") != device
            ):

                raise ValueError(
                    "Cached transcript configuration mismatch."
                )

            valid_segments = validate_transcript(
                transcript_data
            )

            language = str(
                metadata.get("language") or "unknown"
            )

            logger.info(
                "Reusing cached transcript."
            )

            return valid_segments, language

        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):

            logger.warning(
                "Cached transcript is invalid. Re-transcribing.",
                exc_info=True,
            )

    # --------------------------------------------------------
    # TRANSCRIBE VIDEO
    # --------------------------------------------------------

    logger.info(
        "Starting Whisper transcription. Model=%s Device=%s",
        WHISPER_MODEL,
        device,
    )

    transcript, language = transcribe_audio_with_whisper(
        file_path=str(video),
        device=device,
    )

    valid_segments = validate_transcript(
        transcript
    )

    language = str(
        language or "unknown"
    )

    # --------------------------------------------------------
    # SAVE TRANSCRIPT
    # --------------------------------------------------------

    with transcript_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            valid_segments,
            file,
            ensure_ascii=False,
            indent=2,
        )

    metadata = {
        "language": language,
        "model_size": WHISPER_MODEL,
        "device": device,
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # UPDATE CACHE
    # --------------------------------------------------------

    st.session_state["cached_transcription_config"] = (
        transcription_config
    )

    st.session_state["cached_transcript_path"] = str(
        transcript_path
    )

    logger.info(
        "Transcript saved: %s",
        transcript_path,
    )

    return valid_segments, language


# ============================================================
# OUTPUT HELPERS
# ============================================================

def create_output_zip(
    output_directory: Path,
    zip_path: Path,
) -> Path:
    """Create a ZIP archive of generated clips and subtitles."""

    zip_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for file_path in output_directory.iterdir():

            if (
                file_path.is_file()
                and file_path != zip_path
            ):

                archive.write(
                    file_path,
                    arcname=file_path.name,
                )

    logger.info(
        "ZIP bundle created: %s",
        zip_path,
    )

    return zip_path


def create_unique_output_directory(
    clip_count: int,
    clip_length: int,
) -> Path:
    """Create a uniquely named output directory."""

    timestamp = datetime.now().strftime(
        "%d_%m_%Y_%H_%M_%S"
    )

    base_name = (
        f"{timestamp}_clips{clip_count}_time{clip_length}"
    )

    output_job = OUTPUT_DIR / base_name

    counter = 1

    while output_job.exists():

        output_job = OUTPUT_DIR / (
            f"{base_name}_{counter}"
        )

        counter += 1

    output_job.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_job


# ============================================================
# CLEAR CACHE
# ============================================================

def clear_cache() -> None:
    """Delete the active cached video and transcript."""

    cached_job = st.session_state.get(
        "cached_job"
    )

    if cached_job:

        job_path = Path(cached_job)

        try:

            resolved_job = job_path.resolve()
            resolved_work = WORK_DIR.resolve()

            # Delete only direct child directories of WORK_DIR.
            if resolved_job.parent == resolved_work:

                shutil.rmtree(
                    resolved_job,
                    ignore_errors=True,
                )

        except OSError:

            logger.exception(
                "Unable to remove cached working directory."
            )

    for key, default_value in SESSION_DEFAULTS.items():

        st.session_state[key] = default_value

    logger.info(
        "Active video and transcript cache cleared."
    )


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header(
    "⚙️ Generation Settings"
)

clip_count = st.sidebar.selectbox(
    "Number of clips",
    options=CLIP_COUNTS,
    index=CLIP_COUNTS.index(
        DEFAULT_CLIP_COUNT
    ),
)

clip_length = st.sidebar.selectbox(
    "Clip duration",
    options=CLIP_LENGTHS,
    index=CLIP_LENGTHS.index(
        DEFAULT_CLIP_LENGTH
    ),
)

device_choice = st.sidebar.selectbox(
    "Processing device",
    options=PROCESSING_DEVICES,
    index=PROCESSING_DEVICES.index(
        DEFAULT_PROCESSING_DEVICE
    ),
)

subtitle_style = st.sidebar.selectbox(
    "Subtitle style",
    options=SUBTITLE_STYLES,
    index=SUBTITLE_STYLES.index(
        DEFAULT_SUBTITLE_STYLE
    ),
)

st.sidebar.divider()

if st.sidebar.button(
    "🗑️ Clear cached video and transcript",
    use_container_width=True,
):

    clear_cache()

    st.sidebar.success(
        "Cache cleared."
    )

    st.rerun()


# ============================================================
# MAIN INPUT
# ============================================================

url = st.text_input(
    "YouTube video URL",
    placeholder="Paste your YouTube video URL here...",
)

generate_button = st.button(
    "🚀 Generate Shorts",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESSING PIPELINE
# ============================================================

if generate_button:

    video_url = url.strip()

    if not video_url:

        st.error(
            "Please enter a YouTube URL."
        )

    elif not check_ffmpeg():

        st.error(
            "FFmpeg was not found or could not be executed. "
            "Check the FFmpeg path in config.py."
        )

    else:

        # Only one progress bar for the entire pipeline.
        progress = st.progress(
            0,
            text="Preparing your project...",
        )

        status = st.empty()

        # Clear previous displayed results.
        st.session_state["latest_clips"] = []
        st.session_state["latest_zip"] = None
        st.session_state["latest_title"] = None
        st.session_state["latest_language"] = None
        st.session_state["latest_duration"] = 0.0

        try:

            # ------------------------------------------------
            # DEVICE VALIDATION
            # ------------------------------------------------

            status.info(
                "Checking processing device..."
            )

            validate_processing_device(
                device_choice
            )

            logger.info(
                "Selected processing device: %s",
                device_choice,
            )

            update_progress(
                progress,
                0,
                "Processing device ready: 0%",
            )

            # ------------------------------------------------
            # STEP 1: DOWNLOAD OR REUSE VIDEO (0-20%)
            # ------------------------------------------------

            status.info(
                "Step 1/5: Downloading YouTube video..."
            )

            last_download_percentage = [0.0]

            def download_progress_callback(
                percentage: float,
            ) -> None:
                """Update the single overall progress bar."""

                current_percentage = max(
                    last_download_percentage[0],
                    min(100.0, float(percentage)),
                )

                last_download_percentage[0] = (
                    current_percentage
                )

                overall_percentage = (
                    current_percentage * 0.20
                )

                update_progress(
                    progress,
                    overall_percentage,
                    (
                        f"Downloading video: "
                        f"{current_percentage:.1f}% "
                        f"| Overall: {overall_percentage:.1f}%"
                    ),
                )

            with st.spinner(
                "Downloading or checking cached video..."
            ):

                video, job, info = get_cached_video(
                    video_url=video_url,
                    download_progress_callback=(
                        download_progress_callback
                    ),
                )

            title = str(
                info.get("title") or "YouTube video"
            )

            duration = float(
                info.get("duration") or 0
            )

            update_progress(
                progress,
                20,
                "Video ready: 20%",
            )

            st.success(
                f"Video ready: {title}"
            )

            # ------------------------------------------------
            # STEP 2: TRANSCRIPTION (20-45%)
            # ------------------------------------------------

            status.info(
                "Step 2/5: Transcribing audio using Whisper..."
            )

            update_progress(
                progress,
                20,
                "Transcription started: 20%",
            )

            with st.spinner(
                f"🎙️ Transcribing with Whisper base "
                f"on {device_choice.upper()}..."
            ):

                transcript, language = get_cached_transcript(
                    video=video,
                    job=job,
                    device=device_choice,
                )

            update_progress(
                progress,
                45,
                "Transcription completed: 45%",
            )

            st.success(
                f"Transcript ready. Language: {language}"
            )

            # ------------------------------------------------
            # STEP 3: CANDIDATE MOMENTS (45-55%)
            # ------------------------------------------------

            status.info(
                "Step 3/5: Finding interesting moments..."
            )

            update_progress(
                progress,
                45,
                "Analyzing transcript: 45%",
            )

            with st.spinner(
                "Analyzing transcript and finding candidate moments..."
            ):

                candidates = candidate_windows(
                    transcript,
                    duration,
                    clip_length,
                )

            if not candidates:

                raise ValueError(
                    "No suitable candidate moments were found."
                )

            update_progress(
                progress,
                55,
                "Candidate moments generated: 55%",
            )

            # ------------------------------------------------
            # STEP 4: MOMENT SELECTION (55-65%)
            # ------------------------------------------------

            status.info(
                "Step 4/5: Selecting the best moments..."
            )

            with st.spinner(
                "🧠 Selecting the best moments..."
            ):

                # Local selection logic; no Ollama.
                selected_moments = choose_moments(
                    candidates,
                    clip_count,
                    clip_length,
                )

            if not selected_moments:

                raise ValueError(
                    "No moments were selected for rendering."
                )

            update_progress(
                progress,
                65,
                "Moment selection completed: 65%",
            )

            # ------------------------------------------------
            # STEP 5: RENDER CLIPS (65-95%)
            # ------------------------------------------------

            status.info(
                "Step 5/5: Generating video clips..."
            )

            output_job = create_unique_output_directory(
                clip_count=clip_count,
                clip_length=clip_length,
            )

            generated_clips: list[dict[str, Any]] = []

            total_moments = len(
                selected_moments
            )

            for index, moment in enumerate(
                selected_moments,
                start=1,
            ):

                start_time = float(
                    moment["start"]
                )

                end_time = float(
                    moment["end"]
                )

                if (
                    start_time < 0
                    or end_time <= start_time
                    or end_time > duration + 0.1
                ):

                    raise ValueError(
                        f"Invalid time range for clip {index}: "
                        f"{start_time:.2f} - {end_time:.2f}"
                    )

                actual_clip_length = (
                    end_time - start_time
                )

                if actual_clip_length <= 0:

                    raise ValueError(
                        f"Invalid clip duration for clip {index}."
                    )

                ass_path = (
                    output_job / f"short_{index:02d}.ass"
                )

                srt_path = (
                    output_job / f"short_{index:02d}.srt"
                )

                mp4_path = (
                    output_job / f"short_{index:02d}.mp4"
                )

                logger.info(
                    "Rendering clip %d: %.2f - %.2f (%.2f seconds)",
                    index,
                    start_time,
                    end_time,
                    actual_clip_length,
                )

                with st.spinner(
                    f"🎬 Rendering clip {index}/{total_moments}..."
                ):

                    make_ass(
                        transcript,
                        start_time,
                        end_time,
                        ass_path,
                        subtitle_style,
                    )

                    write_srt(
                        transcript,
                        start_time,
                        end_time,
                        srt_path,
                    )

                    run_ffmpeg(
                        video,
                        start_time,
                        actual_clip_length,
                        ass_path,
                        mp4_path,
                    )

                if (
                    not mp4_path.is_file()
                    or mp4_path.stat().st_size == 0
                ):

                    raise FileNotFoundError(
                        f"Rendered clip is missing or empty: {mp4_path}"
                    )

                generated_clips.append(
                    {
                        "index": index,
                        "start": start_time,
                        "end": end_time,
                        "video": str(mp4_path),
                        "subtitle": str(srt_path),
                    }
                )

                render_percentage = (
                    65 + (index / total_moments) * 30
                )

                update_progress(
                    progress,
                    render_percentage,
                    (
                        f"Rendering clips: "
                        f"{index}/{total_moments} "
                        f"| Overall: {render_percentage:.1f}%"
                    ),
                )

            # ------------------------------------------------
            # CREATE ZIP BUNDLE (95-100%)
            # ------------------------------------------------

            status.info(
                "Creating ZIP bundle..."
            )

            update_progress(
                progress,
                95,
                "Packaging generated clips: 95%",
            )

            zip_path = (
                output_job / "shorts_bundle.zip"
            )

            with st.spinner(
                "📦 Packaging your generated clips..."
            ):

                create_output_zip(
                    output_directory=output_job,
                    zip_path=zip_path,
                )

            if (
                not zip_path.is_file()
                or zip_path.stat().st_size == 0
            ):

                raise FileNotFoundError(
                    "ZIP bundle was not created successfully."
                )

            # ------------------------------------------------
            # SAVE RESULTS
            # ------------------------------------------------

            st.session_state["latest_clips"] = (
                generated_clips
            )

            st.session_state["latest_zip"] = str(
                zip_path
            )

            st.session_state["latest_title"] = title

            st.session_state["latest_language"] = language

            st.session_state["latest_duration"] = duration

            update_progress(
                progress,
                100,
                "Generation completed: 100%",
            )

            status.success(
                "🎉 All Shorts generated successfully!"
            )

            logger.info(
                "Successfully generated %d clips.",
                len(generated_clips),
            )

        except Exception as exc:

            logger.exception(
                "Shorts generation failed."
            )

            status.error(
                "Shorts generation failed."
            )

            st.error(
                str(exc)
            )

        finally:

            logger.info(
                "Processing finished. Cached working files retained."
            )


# ============================================================
# DISPLAY GENERATED CLIPS
# ============================================================

latest_clips = st.session_state.get(
    "latest_clips",
    [],
)

if latest_clips:

    st.divider()

    st.header(
        "🎉 Generated Shorts"
    )

    latest_title = st.session_state.get(
        "latest_title"
    )

    latest_language = st.session_state.get(
        "latest_language"
    )

    if latest_title:

        st.caption(
            f"Source video: {latest_title}"
        )

    if latest_language:

        st.caption(
            f"Transcript language: {latest_language}"
        )

    # Display 3 clips per row.
    for row_start in range(
        0,
        len(latest_clips),
        3,
    ):

        columns = st.columns(
            3,
            gap="small",
        )

        row_clips = latest_clips[
            row_start:row_start + 3
        ]

        for column, clip in zip(
            columns,
            row_clips,
        ):

            clip_number = clip["index"]

            video_path = Path(
                clip["video"]
            )

            subtitle_path = Path(
                clip["subtitle"]
            )

            with column:

                st.markdown(
                    f"**Short {clip_number}**"
                )

                st.caption(
                    f"{clip['start']:.1f}s – {clip['end']:.1f}s"
                )

                if video_path.is_file():

                    st.video(
                        str(video_path),
                        loop=True,
                    )

                    with video_path.open("rb") as video_file:

                        st.download_button(
                            label="⬇️ Video",
                            data=video_file,
                            file_name=video_path.name,
                            mime="video/mp4",
                            key=f"download_video_{clip_number}",
                            use_container_width=True,
                        )

                else:

                    st.warning(
                        "Video file not found."
                    )

                if subtitle_path.is_file():

                    with subtitle_path.open("rb") as subtitle_file:

                        st.download_button(
                            label="📝 Subtitles",
                            data=subtitle_file,
                            file_name=subtitle_path.name,
                            mime="text/plain",
                            key=f"download_subtitle_{clip_number}",
                            use_container_width=True,
                        )

        st.divider()

    # --------------------------------------------------------
    # DOWNLOAD ALL CLIPS
    # --------------------------------------------------------

    latest_zip = st.session_state.get(
        "latest_zip"
    )

    if latest_zip and Path(latest_zip).is_file():

        zip_path = Path(latest_zip)

        with zip_path.open("rb") as zip_file:

            st.download_button(
                label="📦 Download All Shorts (ZIP)",
                data=zip_file,
                file_name="shorts_bundle.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
