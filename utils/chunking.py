"""Semantic chunking with boundary detection and topic-density scoring."""

import logging
import re

import numpy as np
from sentence_transformers import SentenceTransformer

import config as cfg

logger = logging.getLogger(__name__)

_embed_model = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading SentenceTransformer '%s' ...", cfg.EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return _embed_model


# ── boundary detection ───────────────────────────────────────────────────────

def _detect_hard_boundaries(segments: list[dict]) -> set[int]:
    """Return indices where a HARD split should occur (after that index).

    Signals:
      1. Transitional phrases in the segment text
      2. Long pause (>PAUSE_THRESHOLD seconds gap)
      3. Dramatic topic shift (cosine sim < 0.3 between consecutive embeddings)
    """
    if len(segments) < 2:
        return set()

    boundaries = set()

    # precompute embeddings for shift detection
    model = _get_embed_model()
    texts = [s["text"] for s in segments]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    for i in range(len(segments) - 1):
        seg = segments[i]
        next_seg = segments[i + 1]

        # 1. transitional phrase in NEXT segment (the boundary is before it)
        lower_text = next_seg["text"].lower()
        if any(phrase in lower_text for phrase in cfg.BOUNDARY_PHRASES):
            boundaries.add(i)
            continue

        # 2. long pause
        gap = next_seg["start"] - seg["end"]
        if gap >= cfg.PAUSE_THRESHOLD:
            boundaries.add(i)
            continue

        # 3. dramatic topic shift
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        if sim < 0.3:
            boundaries.add(i)

    if boundaries:
        logger.info("Detected %d hard boundaries in transcript", len(boundaries))

    return boundaries


# ── similarity-based merging ─────────────────────────────────────────────────

def _merge_segments_into_chunks(
    segments: list[dict],
    hard_boundaries: set[int],
) -> list[dict]:
    """Merge transcript segments into semantic chunks.

    Each chunk: {"start", "end", "text", "segment_indices": [int, ...]}
    """
    if not segments:
        return []

    model = _get_embed_model()
    texts = [s["text"] for s in segments]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # compute consecutive similarities
    sims = []
    for i in range(len(embeddings) - 1):
        sims.append(float(np.dot(embeddings[i], embeddings[i + 1])))

    if sims:
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)
        threshold = mean_sim - cfg.SIMILARITY_STD_FACTOR * std_sim
    else:
        threshold = 0.5

    chunks = []
    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"],
        "segment_indices": [0],
    }

    for i in range(1, len(segments)):
        seg = segments[i]
        current_duration = seg["end"] - current["start"]

        # hard boundary — always split
        if (i - 1) in hard_boundaries:
            chunks.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "segment_indices": [i],
            }
            continue

        # soft split: duration + similarity
        should_split = (
            current_duration >= cfg.MIN_CHUNK_DURATION
            and (
                (i - 1 < len(sims) and sims[i - 1] < threshold)
                or current_duration >= cfg.MAX_CHUNK_DURATION
            )
        )

        if should_split:
            chunks.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "segment_indices": [i],
            }
        else:
            current["end"] = seg["end"]
            current["text"] += " " + seg["text"]
            current["segment_indices"].append(i)

    chunks.append(current)
    return chunks


# ── fallback: fixed-duration chunks (no transcript) ─────────────────────────

def create_fixed_chunks(duration: float, interval: float = 60.0) -> list[dict]:
    """Create chunks at fixed intervals when there's no transcript."""
    chunks = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + interval, duration)
        chunks.append({
            "start": start,
            "end": end,
            "text": "",
            "segment_indices": [],
        })
        start = end
        idx += 1

    logger.info("Created %d fixed-duration chunks (%.0fs each)", len(chunks), interval)
    return chunks


# ── overlap context ──────────────────────────────────────────────────────────

def _add_overlap_context(chunks: list[dict]) -> None:
    """Add prev_context / next_context from adjacent chunks."""
    for i, chunk in enumerate(chunks):
        # last sentence of previous chunk
        if i > 0 and chunks[i - 1]["text"]:
            sentences = re.split(r'[.!?]+', chunks[i - 1]["text"].strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            chunk["prev_context"] = sentences[-1] if sentences else ""
        else:
            chunk["prev_context"] = ""

        # first sentence of next chunk
        if i < len(chunks) - 1 and chunks[i + 1]["text"]:
            sentences = re.split(r'[.!?]+', chunks[i + 1]["text"].strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            chunk["next_context"] = sentences[0] if sentences else ""
        else:
            chunk["next_context"] = ""


# ── topic density (neighbor similarity) ──────────────────────────────────────

def compute_neighbor_density(chunks: list[dict]) -> None:
    """Compute neighbor_density for each chunk and store it in-place.

    For short topics (single chunk surrounded by different topics):
    - density will be low → signals a passing mention
    - BUT we also compute content_richness based on text length,
      so a dense 45-second explanation still gets a fair score.
    """
    if len(chunks) <= 1:
        for c in chunks:
            c["neighbor_density"] = 1.0
        return

    model = _get_embed_model()
    texts = [c["text"] if c["text"] else "empty" for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    window = 2  # look 2 chunks on each side

    for i, chunk in enumerate(chunks):
        neighbors = []
        for j in range(max(0, i - window), min(len(chunks), i + window + 1)):
            if j != i:
                sim = float(np.dot(embeddings[i], embeddings[j]))
                neighbors.append(sim)

        if neighbors:
            density = np.mean(neighbors)
        else:
            density = 1.0

        # content richness bonus: a chunk with substantial text (>200 chars)
        # that covers a short topic in one chunk shouldn't be penalised
        # just because its neighbors are on different topics.
        text_len = len(chunk["text"])
        if text_len > 200 and density < 0.5:
            # boost: a 45-sec chunk with 300+ chars of content is a real explanation
            richness_boost = min(0.3, text_len / 2000)
            density = min(1.0, density + richness_boost)

        chunk["neighbor_density"] = round(float(np.clip(density, 0, 1)), 4)


# ── main entry point ─────────────────────────────────────────────────────────

def create_chunks(
    segments: list[dict],
    duration: float,
) -> list[dict]:
    """Create semantic chunks from transcript segments.

    If no segments, falls back to fixed-duration chunking.

    Returns list of chunks, each with:
        start, end, text, segment_indices,
        prev_context, next_context, neighbor_density
    """
    if not segments:
        logger.warning("No transcript segments — using fixed-duration chunks")
        chunks = create_fixed_chunks(duration)
    elif len(segments) == 1:
        chunks = [{
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "text": segments[0]["text"],
            "segment_indices": [0],
        }]
    else:
        hard_bounds = _detect_hard_boundaries(segments)
        chunks = _merge_segments_into_chunks(segments, hard_bounds)

    # resolve overlapping timestamps from Whisper
    chunks.sort(key=lambda c: c["start"])

    _add_overlap_context(chunks)
    compute_neighbor_density(chunks)

    logger.info(
        "Chunking complete: %d chunks (%.0f–%.0fs each)",
        len(chunks),
        min((c["end"] - c["start"]) for c in chunks) if chunks else 0,
        max((c["end"] - c["start"]) for c in chunks) if chunks else 0,
    )
    return chunks
