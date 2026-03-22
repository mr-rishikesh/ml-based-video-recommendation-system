"""Query pipeline: multi-signal search with re-ranking, time decay, title boost."""

import logging
import re
from datetime import datetime, timezone

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    SearchParams,
)

import config as cfg
from utils.visual import encode_text as clip_encode_text
from utils.embeddings import _get_model as get_text_model

logger = logging.getLogger(__name__)


def _get_client() -> QdrantClient:
    return QdrantClient(host=cfg.QDRANT_HOST, port=cfg.QDRANT_PORT, timeout=10)


def _clean_query(query: str) -> str:
    """Strip, lowercase, limit length."""
    query = query.strip()
    if not query:
        return ""
    # truncate very long queries
    words = query.split()
    if len(words) > 50:
        query = " ".join(words[:50])
    return query


def _compute_time_factor(upload_date: str) -> float:
    """Time-decay factor based on video age."""
    if not upload_date:
        return 1.0
    try:
        dt = datetime.fromisoformat(upload_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).days
    except (ValueError, TypeError):
        return 1.0

    if age_days > cfg.TIME_DECAY_OLD_DAYS:
        return cfg.TIME_DECAY_OLD_FACTOR
    elif age_days < cfg.TIME_DECAY_NEW_DAYS:
        return cfg.TIME_DECAY_NEW_FACTOR
    return 1.0


def _compute_title_boost(query: str, title: str) -> float:
    """Boost if query words overlap with video title."""
    if not query or not title:
        return 1.0
    q_words = set(re.findall(r'\w+', query.lower()))
    t_words = set(re.findall(r'\w+', title.lower()))
    # ignore very common words
    stopwords = {"the", "a", "an", "is", "in", "to", "of", "and", "for", "how", "what"}
    q_words -= stopwords
    t_words -= stopwords
    if not q_words:
        return 1.0
    overlap = q_words & t_words
    if overlap:
        return cfg.TITLE_BOOST_FACTOR
    return 1.0


def search(
    query: str,
    device: str = "cpu",
    top_k: int = cfg.DEFAULT_TOP_K,
    video_id: str | None = None,
    language: str | None = None,
    require_transcript: bool = False,
) -> list[dict]:
    """Multi-modal search with re-ranking.

    Returns list of result dicts sorted by final_score descending.
    """
    query = _clean_query(query)
    if not query:
        logger.warning("Empty query — returning no results")
        return []

    client = _get_client()

    # ── build optional filter ──
    must_conditions = []
    if video_id:
        must_conditions.append(
            FieldCondition(key="video_id", match=MatchValue(value=video_id))
        )
    if language:
        must_conditions.append(
            FieldCondition(key="detected_language", match=MatchValue(value=language))
        )
    if require_transcript:
        must_conditions.append(
            FieldCondition(key="has_transcript", match=MatchValue(value=True))
        )

    q_filter = Filter(must=must_conditions) if must_conditions else None

    # ── generate query embeddings ──
    text_model = get_text_model()
    text_emb = text_model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False,
    )[0].astype(np.float32)

    clip_emb = clip_encode_text(query, device=device)

    # ── text search ──
    try:
        text_results = client.search(
            collection_name=cfg.COLLECTION_NAME,
            query_vector=NamedVector(name="text", vector=text_emb.tolist()),
            query_filter=q_filter,
            limit=cfg.SEARCH_CANDIDATES,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.error("Text search failed: %s", exc)
        text_results = []

    # ── visual search (if CLIP succeeded) ──
    visual_results = []
    if clip_emb is not None:
        try:
            visual_results = client.search(
                collection_name=cfg.COLLECTION_NAME,
                query_vector=NamedVector(name="visual", vector=clip_emb.tolist()),
                query_filter=q_filter,
                limit=cfg.SEARCH_CANDIDATES,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("Visual search failed: %s — text-only results", exc)

    # ── merge candidates (deduplicate by point ID) ──
    candidates = {}
    for hit in text_results:
        candidates[hit.id] = {
            "payload": hit.payload,
            "text_score": float(hit.score),
            "visual_score": 0.0,
        }

    for hit in visual_results:
        if hit.id in candidates:
            candidates[hit.id]["visual_score"] = float(hit.score)
        else:
            candidates[hit.id] = {
                "payload": hit.payload,
                "text_score": 0.0,
                "visual_score": float(hit.score),
            }

    if not candidates:
        logger.info("No results found for query: '%s'", query[:60])
        return []

    # ── re-rank ──
    results = []
    for point_id, data in candidates.items():
        p = data["payload"]
        text_s = data["text_score"]
        visual_s = data["visual_score"]
        density = p.get("neighbor_density", 1.0)

        # adjust weights if one modality is missing
        has_text = p.get("has_transcript", False) or p.get("has_ocr", False)
        has_vis = p.get("has_visual", False)

        if has_text and has_vis:
            base = (cfg.TEXT_WEIGHT * text_s + cfg.VISUAL_WEIGHT * visual_s
                    + cfg.DENSITY_WEIGHT * density)
        elif has_text:
            base = ((cfg.TEXT_WEIGHT + cfg.VISUAL_WEIGHT) * text_s
                    + cfg.DENSITY_WEIGHT * density)
        else:
            base = ((cfg.TEXT_WEIGHT + cfg.VISUAL_WEIGHT) * visual_s
                    + cfg.DENSITY_WEIGHT * density)

        time_factor = _compute_time_factor(p.get("video_upload_date", ""))
        title_boost = _compute_title_boost(query, p.get("video_title", ""))

        final = base * time_factor * title_boost

        # transcript snippet (~200 chars)
        full_text = p.get("transcript_text", "")
        snippet = full_text[:200] + "..." if len(full_text) > 200 else full_text

        results.append({
            "video_id": p.get("video_id"),
            "video_filename": p.get("video_filename"),
            "video_title": p.get("video_title"),
            "timestamp_start": p.get("timestamp_start"),
            "timestamp_end": p.get("timestamp_end"),
            "transcript_snippet": snippet,
            "ocr_text": (p.get("ocr_text", "") or "")[:200],
            "score": round(final, 4),
            "score_breakdown": {
                "text": round(text_s, 4),
                "visual": round(visual_s, 4),
                "density": round(density, 4),
                "time_factor": round(time_factor, 2),
                "title_boost": round(title_boost, 2),
            },
            "prev_context": p.get("prev_context", ""),
            "next_context": p.get("next_context", ""),
            "source": p.get("source", "unknown"),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    results = results[:top_k]

    # warn about low scores
    if results and results[0]["score"] < cfg.LOW_SCORE_THRESHOLD:
        logger.warning(
            "All results have low scores (max %.3f) — results may not be relevant",
            results[0]["score"],
        )

    logger.info("Query returned %d results (top score: %.3f)", len(results), results[0]["score"] if results else 0)
    return results


def format_results(results: list[dict]) -> str:
    """Pretty-print search results for CLI output."""
    if not results:
        return "No matching content found. Try broader search terms."

    lines = []
    for i, r in enumerate(results, 1):
        start = r["timestamp_start"]
        end = r["timestamp_end"]
        start_str = f"{int(start // 60)}:{int(start % 60):02d}"
        end_str = f"{int(end // 60)}:{int(end % 60):02d}"

        lines.append(f"\n{'=' * 60}")
        lines.append(f"  Result #{i}  |  Score: {r['score']:.3f}  |  [{start_str} - {end_str}]")
        lines.append(f"  Video: {r['video_title']}  ({r['video_filename']})")
        lines.append(f"  Source: {r['source']}")
        lines.append(f"  Scores: text={r['score_breakdown']['text']:.3f}"
                      f"  visual={r['score_breakdown']['visual']:.3f}"
                      f"  density={r['score_breakdown']['density']:.3f}"
                      f"  time={r['score_breakdown']['time_factor']}"
                      f"  title_boost={r['score_breakdown']['title_boost']}")
        if r["transcript_snippet"]:
            lines.append(f"  Text: {r['transcript_snippet']}")
        if r["prev_context"]:
            lines.append(f"  [prev: ...{r['prev_context']}]")
        if r["next_context"]:
            lines.append(f"  [next: {r['next_context']}...]")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)
