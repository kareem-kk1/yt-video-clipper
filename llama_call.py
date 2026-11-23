
from __future__ import annotations

from typing import Any

from logger_config import log_step, logger


# ============================================================
# SPREAD SELECTED CLIPS
# ============================================================

def spread_out(
    candidates: list[dict[str, Any]],
    count: int,
    clip_len: int,
) -> list[dict[str, Any]]:
    """Select clips while avoiding overlapping or nearby moments."""

    if count <= 0:
        raise ValueError(
            "Clip count must be greater than zero."
        )

    if clip_len <= 0:
        raise ValueError(
            "Clip length must be greater than zero."
        )

    selected: list[dict[str, Any]] = []

    min_gap = clip_len * 0.65

    # First pass: select clips with sufficient separation.
    for candidate in candidates:

        if all(
            abs(
                candidate["start"] - item["start"]
            ) >= min_gap
            for item in selected
        ):

            selected.append(candidate)

        if len(selected) == count:
            break

    # Second pass: fill remaining slots if necessary.
    if len(selected) < count:

        for candidate in candidates:

            if candidate not in selected:
                selected.append(candidate)

            if len(selected) == count:
                break

    return sorted(
        selected[:count],
        key=lambda item: item["start"],
    )


# ============================================================
# SELECT BEST MOMENTS
# ============================================================

def choose_moments(
    candidates: list[dict[str, Any]],
    count: int,
    clip_len: int,
) -> list[dict[str, Any]]:
    """
    Select the best Shorts moments using local candidate scores.

    No Ollama, API, external AI service, or model is required.
    """

    log_step(
        "SELECTION",
        "Selecting best Shorts moments",
    )

    logger.info(
        "Available candidates: %d",
        len(candidates),
    )

    logger.info(
        "Requested clips: %d",
        count,
    )

    if not candidates:

        raise RuntimeError(
            "No suitable candidate moments found."
        )

    if count <= 0:

        raise ValueError(
            "Requested clip count must be greater than zero."
        )

    # Candidate windows are already ranked by score.
    ranked = sorted(
        candidates,
        key=lambda item: float(item.get("score", 0)),
        reverse=True,
    )

    # Use the best candidates for selection.
    ranked = ranked[:min(30, len(ranked))]

    logger.info(
        "Using top %d candidates for selection.",
        len(ranked),
    )

    selected = spread_out(
        ranked,
        count,
        clip_len,
    )

    if len(selected) < count:

        raise RuntimeError(
            f"Only {len(selected)} suitable clips found. "
            f"Requested {count}. Try a shorter clip length "
            "or fewer clips."
        )

    for index, moment in enumerate(
        selected,
        start=1,
    ):

        logger.info(
            "Selected clip %d | Start: %.2f | End: %.2f | Score: %.4f",
            index,
            moment["start"],
            moment["end"],
            moment.get("score", 0),
        )

    logger.info(
        "Successfully selected %d clips.",
        len(selected),
    )

    return selected
