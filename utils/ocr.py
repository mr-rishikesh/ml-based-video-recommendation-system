"""OCR extraction from keyframes with dedup and non-English translation."""

import logging

import numpy as np
from PIL import Image

import config as cfg

logger = logging.getLogger(__name__)

_ocr_reader = None
_translator = None


def _get_reader():
    """Lazy-load EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        import os
        import easyocr
        logger.info("Loading EasyOCR reader ...")
        # Fix Windows encoding issue: EasyOCR's download progress bar uses
        # Unicode block chars (█) that cp1252 can't encode.
        # Setting PYTHONIOENCODING won't help a running process, so we
        # monkey-patch the problematic function if needed.
        os.environ["PYTHONIOENCODING"] = "utf-8"
        _ocr_reader = easyocr.Reader(["en"], gpu=_has_gpu(), verbose=False)
    return _ocr_reader


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_translator():
    """Lazy-load Google Translator for non-English OCR text."""
    global _translator
    if _translator is None:
        from deep_translator import GoogleTranslator
        _translator = GoogleTranslator(source="auto", target="en")
    return _translator


def _is_blank_frame(img: np.ndarray) -> bool:
    """Detect blank (black/white) frames by mean pixel value."""
    mean_val = np.mean(img)
    return mean_val < cfg.BLANK_FRAME_THRESHOLD_LOW or mean_val > cfg.BLANK_FRAME_THRESHOLD_HIGH


def _text_overlap(a: str, b: str) -> float:
    """Rough overlap ratio between two strings using word sets."""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))


def _is_noisy_text(text: str) -> bool:
    """Detect garbled OCR output: >50% of words are <2 chars."""
    words = text.split()
    if not words:
        return True
    short = sum(1 for w in words if len(w) < 2)
    return (short / len(words)) > 0.5


def _translate_if_needed(text: str, detected_language: str) -> str:
    """Translate non-English OCR text to English for consistent embedding.

    We translate because the embedding model works best with English text,
    and mixing languages in the vector space reduces retrieval quality.
    """
    if not text or not text.strip():
        return text

    # simple heuristic: if most chars are ASCII, it's likely English
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio > 0.85:
        return text  # likely English already

    try:
        translator = _get_translator()
        translated = translator.translate(text)
        if translated and translated.strip():
            logger.info("Translated OCR text from non-English to English")
            return translated
    except Exception as exc:
        logger.warning("Translation failed: %s — using original text", exc)

    return text


def extract_ocr_for_chunk(
    keyframes: list[dict],
    detected_language: str = "en",
) -> dict:
    """Run OCR on keyframes belonging to a chunk.

    Args:
        keyframes: list of {"path": str, "timestamp": float}
        detected_language: language detected by Whisper (for translation decision)

    Returns:
        {"text": str, "has_ocr": bool, "frame_count": int}
    """
    if not keyframes:
        return {"text": "", "has_ocr": False, "frame_count": 0}

    reader = _get_reader()
    all_texts = []
    prev_text = ""
    frames_with_text = 0

    for kf in keyframes:
        try:
            img = np.array(Image.open(kf["path"]).convert("RGB"))
        except Exception as exc:
            logger.warning("Cannot open frame %s: %s", kf["path"], exc)
            continue

        # skip blank frames
        if _is_blank_frame(img):
            continue

        # run OCR
        try:
            results = reader.readtext(img)
        except Exception as exc:
            logger.warning("OCR failed on %s: %s", kf["path"], exc)
            continue

        # filter by confidence
        texts = [
            text for (_, text, conf) in results
            if conf >= cfg.OCR_CONFIDENCE_THRESHOLD
        ]
        frame_text = " ".join(texts).strip()

        if not frame_text:
            continue

        # filter garbled output
        if _is_noisy_text(frame_text):
            logger.warning("Noisy OCR output skipped at %.1fs", kf["timestamp"])
            continue

        # dedup: skip if too similar to previous frame's text
        if prev_text and _text_overlap(frame_text, prev_text) > cfg.OCR_DEDUP_THRESHOLD:
            continue

        all_texts.append(frame_text)
        prev_text = frame_text
        frames_with_text += 1

    combined = " ".join(all_texts).strip()

    # translate non-English OCR to English for embedding consistency
    if combined:
        combined = _translate_if_needed(combined, detected_language)

    has_ocr = len(combined) > 0

    if has_ocr:
        logger.debug(
            "OCR: %d chars from %d/%d frames", len(combined), frames_with_text, len(keyframes)
        )

    return {
        "text": combined,
        "has_ocr": has_ocr,
        "frame_count": frames_with_text,
    }
