
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import yt_dlp

from config import (
    COOKIE_FILE,
    FFMPEG_DIR,
    configure_ffmpeg,
)

from logger_config import logger, log_step


# ============================================================
# DOWNLOAD PROGRESS
# ============================================================

def create_progress_hook(
    progress_callback: Callable[[float], None] | None = None,
) -> Callable[[dict[str, Any]], None]:
    """Create a yt-dlp progress hook for logging and Streamlit."""

    last_percent = -1

    def progress_hook(data: dict[str, Any]) -> None:
        nonlocal last_percent

        status = data.get("status")

        if status == "downloading":

            try:
                downloaded = float(
                    data.get("downloaded_bytes") or 0
                )

                total = float(
                    data.get("total_bytes")
                    or data.get("total_bytes_estimate")
                    or 0
                )

            except (TypeError, ValueError, OverflowError):
                return

            if total <= 0:
                return

            percentage = min(
                100.0,
                (downloaded / total) * 100,
            )

            if progress_callback is not None:
                progress_callback(percentage)

            current_percent = int(percentage)

            if (
                current_percent != last_percent
                and current_percent % 10 == 0
            ):
                logger.info(
                    "Download progress: %d%%",
                    current_percent,
                )

                last_percent = current_percent

        elif status == "finished":

            logger.info(
                "Download finished. Processing video and audio..."
            )

            if progress_callback is not None:
                progress_callback(100.0)

    return progress_hook


# ============================================================
# COOKIE VALIDATION
# ============================================================

def has_valid_netscape_cookies(
    cookie_content: str,
) -> bool:
    """
    Check for recognizable Netscape-format cookie entries.

    Supports regular and #HttpOnly_ cookie entries.

    This checks file structure only, not whether the
    cookies are currently authenticated.
    """

    for line in cookie_content.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#") and not line.startswith(
            "#HttpOnly_"
        ):
            continue

        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]

        fields = line.split("\t")

        if len(fields) == 7:
            return True

    return False


def cookies_file_available() -> bool:
    """Check whether the project cookies.txt file is usable."""

    if not COOKIE_FILE.is_file():
        logger.info("cookies.txt not found.")
        return False

    try:
        content = COOKIE_FILE.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        if has_valid_netscape_cookies(content):
            logger.info(
                "Recognizable Netscape cookie entries detected."
            )
            return True

        logger.warning(
            "cookies.txt is empty or has an invalid format."
        )

    except OSError:
        logger.warning(
            "Unable to read cookies.txt.",
            exc_info=True,
        )

    return False


# ============================================================
# ERROR HANDLING
# ============================================================

def is_authentication_error(
    error: Exception | str,
) -> bool:
    """Identify common YouTube authentication errors."""

    message = str(error).lower()

    authentication_errors = (
        "sign in to confirm",
        "not a bot",
        "cookies are no longer valid",
        "login required",
        "authentication",
        "confirm you're not a bot",
        "use --cookies-from-browser",
        "members-only",
        "private video",
        "age-restricted",
    )

    return any(
        item in message
        for item in authentication_errors
    )


def is_browser_cookie_error(
    error: Exception | str,
) -> bool:
    """Identify errors accessing Chrome's cookie database."""

    message = str(error).lower()

    cookie_errors = (
        "could not copy chrome cookie database",
        "failed to load cookies",
        "cookie database",
        "could not decrypt",
        "failed to decrypt",
    )

    return any(
        item in message
        for item in cookie_errors
    )


# ============================================================
# DOWNLOADED VIDEO DETECTION
# ============================================================

def find_downloaded_video(
    folder: Path,
    prepared_path: Path,
) -> Path:
    """Find the final downloaded video, preferring MP4."""

    # Check the expected merged MP4 file first.
    expected_mp4 = folder / "source.mp4"

    if expected_mp4.is_file():
        return expected_mp4

    # Check the filename returned by yt-dlp.
    if prepared_path.is_file():
        return prepared_path

    merged_path = prepared_path.with_suffix(".mp4")

    if merged_path.is_file():
        return merged_path

    # Search for remaining media files.
    ignored_extensions = {
        ".part",
        ".ytdl",
        ".temp",
        ".json",
        ".description",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".vtt",
        ".srt",
    }

    candidates = [
        item
        for item in folder.glob("source.*")
        if item.is_file()
        and item.suffix.lower() not in ignored_extensions
    ]

    if not candidates:
        raise RuntimeError(
            "Downloaded video file was not found."
        )

    mp4_candidates = [
        item
        for item in candidates
        if item.suffix.lower() == ".mp4"
    ]

    if mp4_candidates:
        return mp4_candidates[0]

    return max(
        candidates,
        key=lambda item: item.stat().st_size,
    )


# ============================================================
# YT-DLP DOWNLOAD HELPER
# ============================================================

def download_with_options(
    video_url: str,
    options: dict[str, Any],
    folder_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Download a video using the supplied yt-dlp options."""

    with yt_dlp.YoutubeDL(cast(Any, options)) as ydl:

        info_result = ydl.extract_info(
            video_url,
            download=True,
        )

        if not isinstance(info_result, Mapping):
            raise RuntimeError(
                "yt-dlp returned invalid video metadata."
            )

        info = info_result

        prepared_path = Path(
            ydl.prepare_filename(info)
        )

    video_path = find_downloaded_video(
        folder_path,
        prepared_path,
    )

    if not video_path.is_file():
        raise RuntimeError(
            f"Downloaded video does not exist: {video_path}"
        )

    if video_path.stat().st_size <= 0:
        raise RuntimeError(
            "Downloaded video file is empty."
        )

    return video_path, dict(info)


# ============================================================
# MAIN DOWNLOAD FUNCTION
# ============================================================

def download_video(
    video_url: str,
    folder: Path,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """
    Download a YouTube video using yt-dlp.

    Authentication priority:
        1. cookies.txt, if available.
        2. Public download.
        3. Chrome browser cookies if authentication is required.

    Returns:
        Downloaded video path and video metadata.
    """

    log_step(
        "DOWNLOAD",
        "Starting YouTube video download",
    )

    start_time = time.perf_counter()

    folder_path = Path(folder)

    authentication_method = "unknown"

    try:

        # ----------------------------------------------------
        # DIRECTORY AND FFMPEG SETUP
        # ----------------------------------------------------

        folder_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        configure_ffmpeg()

        if not video_url.strip():
            raise ValueError("YouTube URL cannot be empty.")

        # ----------------------------------------------------
        # BASE DOWNLOAD OPTIONS
        # ----------------------------------------------------

        progress_hook = create_progress_hook(
            progress_callback
        )

        base_options: dict[str, Any] = {
            "format": (
                "bestvideo[height<=1080]+bestaudio/"
                "best[height<=1080]/best"
            ),
            "ffmpeg_location": str(FFMPEG_DIR),
            "merge_output_format": "mp4",
            "outtmpl": str(
                folder_path / "source.%(ext)s"
            ),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [progress_hook],
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
        }

        logger.info(
            "YouTube URL: %s",
            video_url,
        )

        # ----------------------------------------------------
        # FIRST ATTEMPT
        # ----------------------------------------------------

        if cookies_file_available():

            authentication_method = "cookies.txt"

            logger.info(
                "Attempting download using cookies.txt."
            )

            download_options = {
                **base_options,
                "cookiefile": str(COOKIE_FILE),
            }

        else:

            authentication_method = "public"

            logger.info(
                "Attempting public video download."
            )

            download_options = base_options

        try:

            video_path, info = download_with_options(
                video_url=video_url,
                options=download_options,
                folder_path=folder_path,
            )

        except Exception as first_error:

            # -----------------------------------------------
            # CHROME COOKIE FALLBACK
            # -----------------------------------------------

            if not is_authentication_error(first_error):
                raise

            logger.warning(
                "Authentication required. "
                "Retrying with Chrome browser cookies."
            )

            authentication_method = "chrome"

            chrome_options = {
                **base_options,
                "cookiesfrombrowser": ("chrome",),
            }

            try:

                video_path, info = download_with_options(
                    video_url=video_url,
                    options=chrome_options,
                    folder_path=folder_path,
                )

            except Exception as chrome_error:

                if is_browser_cookie_error(chrome_error):
                    raise RuntimeError(
                        "Unable to access Chrome cookies. "
                        "Close Chrome completely and try again, "
                        "or export YouTube cookies in Netscape "
                        "format and save them as cookies.txt."
                    ) from chrome_error

                if is_authentication_error(chrome_error):
                    raise RuntimeError(
                        "YouTube authentication failed. "
                        "Export fresh YouTube cookies and save "
                        "them as cookies.txt in the project folder."
                    ) from chrome_error

                raise

        # ----------------------------------------------------
        # DOWNLOAD SUMMARY
        # ----------------------------------------------------

        file_size = video_path.stat().st_size

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Authentication method: %s",
            authentication_method,
        )

        logger.info(
            "Video title: %s",
            info.get("title", "Unknown"),
        )

        logger.info(
            "Video duration: %s seconds",
            info.get("duration", "Unknown"),
        )

        logger.info(
            "Downloaded file: %s",
            video_path,
        )

        logger.info(
            "File size: %.2f MB",
            file_size / (1024 * 1024),
        )

        logger.info(
            "Download completed in %.2f seconds",
            elapsed,
        )

        log_step(
            "DOWNLOAD",
            "YouTube video download completed successfully",
        )

        return video_path, info

    except Exception as error:

        logger.exception(
            "YouTube video download failed."
        )

        if is_browser_cookie_error(error):
            raise RuntimeError(
                "Unable to access Chrome browser cookies. "
                "Close Chrome completely or export YouTube "
                "cookies to cookies.txt."
            ) from error

        if is_authentication_error(error):
            raise RuntimeError(
                "YouTube requires authentication for this video. "
                "Please export fresh YouTube cookies and save "
                "them as cookies.txt in the project folder."
            ) from error

        raise RuntimeError(
            f"YouTube video download failed: {error}"
        ) from error
