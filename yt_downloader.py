from __future__ import annotations
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
import yt_dlp
from yt_dlp.utils import DownloadError
from logger_config import logger, log_step
from collections.abc import Callable


def create_progress_hook(
    progress_callback: Callable[[float], None] | None = None,
) -> Callable[[dict[str, Any]], None]:
    """Create a yt-dlp progress hook for Streamlit."""

    def progress_hook(data: dict[str, Any]) -> None:

        if progress_callback is None:
            return

        status = data.get("status")

        if status == "downloading":

            downloaded = float(
                data.get("downloaded_bytes") or 0
            )

            total = float(
                data.get("total_bytes")
                or data.get("total_bytes_estimate")
                or 0
            )

            if total > 0:

                percentage = min(
                    100.0,
                    (downloaded / total) * 100,
                )

                progress_callback(percentage)

        elif status == "finished":

            progress_callback(100.0)

    return progress_hook


# ============================================================
# COOKIE VALIDATION
# ============================================================

def has_valid_netscape_cookies(cookie_content: str) -> bool:
    """
    Check whether cookie content contains recognizable
    Netscape-format cookie entries.

    Supports regular cookies and #HttpOnly_ entries.

    Note:
        This checks the file structure only. It cannot verify
        whether the cookies are still authenticated or valid.
    """
    for line in cookie_content.splitlines():

        line = line.strip()

        if not line:
            continue

        # Ignore comments except HttpOnly cookie entries.
        if line.startswith("#") and not line.startswith("#HttpOnly_"):
            continue

        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]

        fields = line.split("\t")

        # Netscape format contains seven tab-separated fields.
        if len(fields) >= 7:
            return True

    return False


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def is_authentication_error(error: Exception | str) -> bool:
    """Identify common YouTube authentication errors."""

    error_message = str(error).lower()

    authentication_errors = (
        "sign in to confirm",
        "not a bot",
        "cookies are no longer valid",
        "login required",
        "authentication",
        "confirm you're not a bot",
        "use --cookies-from-browser",
        "members-only",
    )

    return any(
        message in error_message
        for message in authentication_errors
    )


def find_downloaded_video(
    folder: Path,
    prepared_path: Path,
) -> Path:
    """
    Locate the downloaded video, preferring the merged MP4 file.
    """

    # yt-dlp may return the pre-merge filename.
    if prepared_path.suffix.lower() != ".mp4":

        merged_path = prepared_path.with_suffix(".mp4")

        if merged_path.is_file():
            return merged_path

    # Prefer the expected output filename.
    expected_mp4 = folder / "source.mp4"

    if expected_mp4.is_file():
        return expected_mp4

    # Use the prepared filename if it exists.
    if prepared_path.is_file():
        return prepared_path

    # Ignore temporary files and metadata.
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

    # Prefer MP4 candidates.
    mp4_candidates = [
        item
        for item in candidates
        if item.suffix.lower() == ".mp4"
    ]

    if mp4_candidates:
        return mp4_candidates[0]

    # Otherwise, select the largest available media file.
    return max(
        candidates,
        key=lambda item: item.stat().st_size,
    )


def download_with_options(
    video_url: str,
    options: dict[str, Any],
    folder_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """
    Execute yt-dlp using the supplied authentication options.
    """

    with yt_dlp.YoutubeDL(cast(Any, options)) as ydl:

        info_result = ydl.extract_info(
            video_url,
            download=True,
        )

        if not isinstance(info_result, Mapping):
            raise RuntimeError(
                "yt-dlp returned invalid video metadata."
            )

        info: Mapping[str, Any] = info_result

        prepared_path = Path(
            ydl.prepare_filename(info)
        )

    # Find the actual downloaded/merged video.
    video_path = find_downloaded_video(
        folder_path,
        prepared_path,
    )

    if not video_path.is_file():
        raise RuntimeError(
            f"Downloaded video does not exist: {video_path}"
        )

    file_size = video_path.stat().st_size

    if file_size <= 0:
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
      APP_DIR: Path,
      progress_callback: Callable[[float], None] | None = None,
  ) -> tuple[Path, dict[str, Any]]:
    """
    Download a YouTube video using yt-dlp.

    Authentication priority:
        1. Use valid-looking cookies.txt if available.
        2. If missing/empty, use Chrome browser cookies.
        3. If cookies.txt is rejected for authentication,
           retry using Chrome browser cookies.

    Returns:
        tuple[Path, dict[str, Any]]:
            Downloaded video path and yt-dlp metadata.
    """

    log_step(
        "DOWNLOAD",
        "Starting YouTube video download",
    )

    start_time = time.perf_counter()

    last_percent = -1
    authentication_method = "unknown"

    try:

        # --------------------------------------------------
        # DIRECTORY SETUP
        # --------------------------------------------------

        folder_path = Path(folder)
        app_dir = Path(APP_DIR)

        folder_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        cookie_file = app_dir / "cookies.txt"

        # --------------------------------------------------
        # DOWNLOAD PROGRESS HOOK
        # --------------------------------------------------

        def download_progress_hook(
            data: dict[str, Any],
        ) -> None:

            nonlocal last_percent

            status = data.get("status")

            if status == "downloading":

                downloaded = data.get(
                    "downloaded_bytes",
                    0,
                )

                total = (
                    data.get("total_bytes")
                    or data.get("total_bytes_estimate")
                    or 0
                )

                try:
                    downloaded_bytes = int(downloaded or 0)
                    total_bytes = int(total or 0)

                except (
                    TypeError,
                    ValueError,
                    OverflowError,
                ):

                    logger.warning(
                        "Invalid download progress values received."
                    )
                    return

                if total_bytes > 0:

                    percent = min(
                        100,
                        int(
                            downloaded_bytes * 100 / total_bytes
                        ),
                    )

                    if (
                        percent != last_percent
                        and percent % 10 == 0
                    ):

                        logger.info(
                            "Download progress: %d%%",
                            percent,
                        )

                        last_percent = percent

            elif status == "finished":

                logger.info(
                    "Download completed. "
                    "Merging video if required."
                )

        # --------------------------------------------------
        # CHECK COOKIE FILE
        # --------------------------------------------------

        use_cookie_file = False

        if cookie_file.is_file():

            try:

                cookie_size = cookie_file.stat().st_size

                if cookie_size > 0:

                    cookie_content = cookie_file.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )

                    use_cookie_file = has_valid_netscape_cookies(
                        cookie_content,
                    )

                    if use_cookie_file:

                        logger.info(
                            "Valid-looking Netscape cookie entries detected."
                        )

                    else:

                        logger.warning(
                            "cookies.txt does not contain recognizable "
                            "Netscape cookie entries."
                        )

                else:

                    logger.warning(
                        "cookies.txt is empty."
                    )

            except OSError:

                logger.warning(
                    "Unable to read cookies.txt.",
                    exc_info=True,
                )

        else:

            logger.info(
                "cookies.txt not found."
            )

        # --------------------------------------------------
        # BASE YT-DLP OPTIONS
        # --------------------------------------------------

        outtmpl = str(
            folder_path / "source.%(ext)s"
        )

        base_options: dict[str, Any] = {
            "format": (
                "bestvideo[height<=1080]+bestaudio/"
                "best[height<=1080]/best"
            ),
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [
                download_progress_hook
            ],
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
        }
        base_options["progress_hooks"] = [create_progress_hook(progress_callback)
]
        logger.info(
            "YouTube URL: %s",
            video_url,
        )

        
        # --------------------------------------------------
        # AUTHENTICATION
        # --------------------------------------------------

        if use_cookie_file:
            authentication_method = "cookies.txt"

            logger.info(
                "Authentication: Using exported cookies.txt"
            )

            download_options = {
                **base_options,
                "cookiefile": str(cookie_file),
            }

        else:
            authentication_method = "public"

            logger.info(
                "Authentication: No cookies available. "
                "Attempting public video download."
            )

            download_options = base_options

        video_path, info = download_with_options(
            video_url=video_url,
            options=download_options,
            folder_path=folder_path,
)

   
        # --------------------------------------------------
        # DOWNLOAD SUMMARY
        # --------------------------------------------------

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
            "Video duration: %s",
            info.get("duration", "Unknown"),
        )

        logger.info(
            "Video file: %s",
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

        error_message = str(error).lower()

        if (
            "could not copy chrome cookie database"
            in error_message
            or "failed to load cookies" in error_message
            or "cookie database" in error_message
        ):
            raise RuntimeError(
                "Unable to access browser cookies. "
                "Export YouTube cookies in Netscape format "
                "and save them as cookies.txt in the project folder."
            ) from error

        if is_authentication_error(error):

            if authentication_method == "cookies.txt":
                raise RuntimeError(
                    "YouTube rejected cookies.txt. "
                    "Your session may have expired. "
                    "Export fresh YouTube cookies and replace the file."
                ) from error

            raise RuntimeError(
                "YouTube requires authentication for this video. "
                "Please export your signed-in YouTube session "
                "as cookies.txt."
            ) from error

        raise RuntimeError(
            f"YouTube video download failed: {error}"
        ) from error

    
    
    