"""Text embedding generation for chunks using SentenceTransformer."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

import config as cfg

logger = logging.getLogger(__name__)

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer '%s' ...", cfg.EMBEDDING_MODEL)
        _model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return _model


def generate_text_embeddings(chunks: list[dict]) -> list[np.ndarray | None]:
    """Generate text embeddings for each chunk.

    Input text = transcript_text + " " + ocr_text.
    Returns list parallel to chunks: np.ndarray(384,) or None if no text.
    """
    model = _get_model()

    texts = []
    indices_with_text = []

    for i, chunk in enumerate(chunks):
        transcript = chunk.get("text", "") or ""
        ocr = chunk.get("ocr_text", "") or ""
        combined = (transcript + " " + ocr).strip()

        if not combined:
            texts.append(None)
        else:
            # truncate to avoid silent model truncation
            if len(combined) > cfg.MAX_TEXT_LENGTH:
                combined = combined[: cfg.MAX_TEXT_LENGTH]
            texts.append(combined)
            indices_with_text.append(i)

    # batch-embed all non-None texts
    if indices_with_text:
        valid_texts = [texts[i] for i in indices_with_text]
        embeddings = model.encode(
            valid_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True,
        )
    else:
        embeddings = np.array([])

    # assemble results
    results: list[np.ndarray | None] = [None] * len(chunks)
    for j, idx in enumerate(indices_with_text):
        results[idx] = embeddings[j].astype(np.float32)

    n_embedded = len(indices_with_text)
    n_empty = len(chunks) - n_embedded
    logger.info("Text embeddings: %d generated, %d empty (no text)", n_embedded, n_empty)

    return results
