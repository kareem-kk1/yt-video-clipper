from logger_config import log_step,logger
import time
import json
import requests

def call_llama(prompt, model):
    log_step("LLAMA", "Starting AI moment selection")
    start_time = time.perf_counter()

    try:
        logger.info("Llama model: %s", model)
        logger.info("Prompt size: %d characters", len(prompt))

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=180
        )

        response.raise_for_status()
        result = response.json().get(
            "response",
            ""
        )

        elapsed = time.perf_counter() - start_time

        logger.info("Llama response received")
        logger.info("Response length: %d", len(result))
        logger.info("Llama execution time: %.2f seconds", elapsed)
        return result

    except Exception:
        logger.exception("Llama API request failed")
        raise
    
def spread_out(candidates, count, clip_len):
    selected = []
    min_gap = clip_len * 0.65

    for candidate in candidates:
        if all(
            abs(candidate["start"] - item["start"]) >= min_gap
            for item in selected
        ):
            selected.append(candidate)

        if len(selected) == count:
            break

    if len(selected) < count:
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)

            if len(selected) == count:
                break

    return sorted(
        selected[:count],
        key=lambda x: x["start"]
    )
    
    
    
def choose_moments(
    candidates,
    count,
    clip_len,
    use_ai,
    model
):

    log_step("SELECTION", "Selecting best Shorts moments")

    logger.info("Available candidates: %d", len(candidates))
    logger.info("Requested clips: %d", count)
    logger.info("AI ranking enabled: %s", use_ai)

    if not candidates:
        raise RuntimeError(
            "No suitable candidate moments found."
        )

    ranked = candidates[:min(30, len(candidates))]

    selected = []

    if use_ai:
        try:
            compact = [
                {
                    "id": i,
                    "start": round(candidate["start"], 1),
                    "text": candidate["text"][:900]
                }
                for i, candidate in enumerate(ranked)
            ]

            prompt = (
                "You are an expert YouTube Shorts editor. "
                "Choose distinct, compelling, self-contained moments "
                "with a strong hook, useful insight, story payoff, "
                "or surprising point. Avoid repetitive sections. "
                'Return ONLY valid JSON: {"selected_ids":[integers]}. '
                f"Select exactly {count} unique IDs from: "
                f"{json.dumps(compact, ensure_ascii=False)}"
            )

            raw = call_llama(prompt, model)

            data = json.loads(raw)

            ids = data.get("selected_ids", [])

            if not isinstance(ids, list):
                raise ValueError("Ollama returned an invalid selected_ids list.")

            valid_ids = []

            for item in ids:
                if (
                    isinstance(item, int)
                    and not isinstance(item, bool)
                    and 0 <= item < len(ranked)
                    and item not in valid_ids
                ):

                    valid_ids.append(item)

            selected = [
                ranked[i]
                for i in valid_ids
            ]

            logger.info(
                "Ollama selected %d unique moments",
                len(selected)
            )

        except Exception:
            logger.exception(
                "AI moment selection failed. Using local ranking."
            )

            selected = []

    if len(selected) < count:
        logger.info(
            "Filling remaining clips using local candidate ranking."
        )

        combined = selected + [
            candidate
            for candidate in ranked
            if candidate not in selected
        ]

        selected = spread_out(
            combined,
            count,
            clip_len
        )

    else:
        selected = spread_out(
            selected,
            count,
            clip_len
        )

    if len(selected) < count:
        raise RuntimeError(
            f"Only {len(selected)} suitable clips found. "
            f"Requested {count}. Try a shorter clip length "
            "or fewer clips."
        )

    for index, moment in enumerate(selected, 1):
        logger.info(
            "Selected clip %d | Start: %.2f | End: %.2f | Score: %.4f",
            index,
            moment["start"],
            moment["end"],
            moment["score"]
        )

    return selected