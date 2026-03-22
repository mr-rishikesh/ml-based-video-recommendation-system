"""Whisper-based transcription with hallucination detection and windowed processing."""

import logging

import config as cfg

logger = logging.getLogger(__name__)

# lazy-loaded
_whisper_model = None


def _load_model(device: str):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("Loading Whisper model '%s' ...", cfg.WHISPER_MODEL)
        try:
            _whisper_model = whisper.load_model(cfg.WHISPER_MODEL, device=device)
        except Exception as exc:
            logger.warning("First Whisper load failed (%s), retrying ...", exc)
            _whisper_model = whisper.load_model(cfg.WHISPER_MODEL, device="cpu")
    return _whisper_model


def _deduplicate_hallucinations(segments: list[dict]) -> list[dict]:
    """Remove Whisper hallucinations: consecutive segments with identical text."""
    if len(segments) < 3:
        return segments

    cleaned = []
    repeat_count = 1
    for i, seg in enumerate(segments):
        if i > 0 and seg["text"].strip() == segments[i - 1]["text"].strip():
            repeat_count += 1
        else:
            repeat_count = 1

        if repeat_count <= 2:
            cleaned.append(seg)
        else:
            logger.warning(
                "Hallucination detected: '%s' repeated %d+ times at %.1fs — skipping",
                seg["text"].strip()[:60], repeat_count, seg["start"],
            )

    return cleaned


def _segments_from_result(result: dict) -> list[dict]:
    """Convert Whisper result to our segment format with confidence."""
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip(),
            "confidence": float(seg.get("avg_logprob", -1)),
        })
    return segments


def transcribe(
    audio_path: str,
    duration: float,
    device: str = "cpu",
) -> dict:
    """Transcribe audio and return structured result.

    Returns:
        {
            "segments": [{"start", "end", "text", "confidence"}, ...],
            "language": str,
            "has_transcript": bool,
            "low_confidence": bool,
        }
    """
    model = _load_model(device)
    empty = {
        "segments": [],
        "language": "unknown",
        "has_transcript": False,
        "low_confidence": True,
    }

    if audio_path is None:
        return empty

    try:
        # long audio → windowed processing
        if duration > cfg.WHISPER_SEGMENT_MAX_SECONDS:
            return _transcribe_windowed(model, audio_path, duration)

        result = model.transcribe(
            audio_path,
            word_timestamps=False,
            verbose=False,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.warning("Whisper OOM — falling back to windowed processing")
            return _transcribe_windowed(model, audio_path, duration)
        logger.error("Whisper error: %s", exc)
        return empty
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        return empty

    language = result.get("language", "unknown")
    segments = _segments_from_result(result)
    segments = _deduplicate_hallucinations(segments)

    # confidence check
    if segments:
        avg_conf = sum(s["confidence"] for s in segments) / len(segments)
        low_confidence = avg_conf < cfg.WHISPER_CONFIDENCE_THRESHOLD
    else:
        avg_conf = 0
        low_confidence = True

    has_transcript = len(segments) > 0 and not low_confidence

    if low_confidence and segments:
        logger.warning(
            "Low transcript confidence (%.2f) — may be music/noise only", avg_conf
        )

    if not segments:
        logger.warning("Whisper returned 0 segments — audio may be silent")

    logger.info(
        "Transcription: %d segments, language=%s, confidence=%.2f",
        len(segments), language, avg_conf,
    )

    return {
        "segments": segments,
        "language": language,
        "has_transcript": has_transcript,
        "low_confidence": low_confidence,
    }


def _transcribe_windowed(model, audio_path: str, duration: float) -> dict:
    """Process long audio in overlapping windows to avoid OOM."""
    import numpy as np
    import whisper

    logger.info("Using windowed transcription (%.0fs audio)", duration)

    audio = whisper.load_audio(audio_path)
    sr = 16000
    window = cfg.WHISPER_SEGMENT_MAX_SECONDS * sr
    overlap = cfg.WHISPER_OVERLAP_SECONDS * sr
    step = window - overlap

    all_segments = []
    language = "unknown"
    offset = 0

    while offset < len(audio):
        chunk = audio[offset: offset + window]
        # pad if too short
        if len(chunk) < sr:
            break

        try:
            result = model.transcribe(
                chunk,
                word_timestamps=False,
                verbose=False,
            )
        except RuntimeError:
            logger.warning("OOM on window at %.0fs — skipping", offset / sr)
            offset += step
            continue

        if language == "unknown":
            language = result.get("language", "unknown")

        time_offset = offset / sr
        for seg in result.get("segments", []):
            all_segments.append({
                "start": float(seg["start"]) + time_offset,
                "end": float(seg["end"]) + time_offset,
                "text": seg["text"].strip(),
                "confidence": float(seg.get("avg_logprob", -1)),
            })

        offset += step

    # deduplicate overlap regions — remove segments whose start falls in
    # the overlap zone AND whose text is similar to an existing segment
    all_segments.sort(key=lambda s: s["start"])
    deduped = []
    for seg in all_segments:
        if deduped and abs(seg["start"] - deduped[-1]["start"]) < cfg.WHISPER_OVERLAP_SECONDS:
            # keep the one with higher confidence
            if seg["confidence"] > deduped[-1]["confidence"]:
                deduped[-1] = seg
        else:
            deduped.append(seg)

    deduped = _deduplicate_hallucinations(deduped)

    if deduped:
        avg_conf = sum(s["confidence"] for s in deduped) / len(deduped)
        low_confidence = avg_conf < cfg.WHISPER_CONFIDENCE_THRESHOLD
    else:
        avg_conf = 0
        low_confidence = True

    logger.info(
        "Windowed transcription: %d segments, language=%s", len(deduped), language
    )

    return {
        "segments": deduped,
        "language": language,
        "has_transcript": len(deduped) > 0 and not low_confidence,
        "low_confidence": low_confidence,
    }
