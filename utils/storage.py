"""Qdrant storage: collection management, video dedup, batch upsert."""

import hashlib
import logging
import os
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    PointStruct,
    VectorParams,
)

import config as cfg

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=cfg.QDRANT_HOST, port=cfg.QDRANT_PORT, timeout=10)
    return _client


# ── collection management ────────────────────────────────────────────────────

def ensure_collection() -> None:
    """Create the collection if it doesn't exist; validate config if it does."""
    client = _get_client()
    collections = [c.name for c in client.get_collections().collections]

    if cfg.COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=cfg.COLLECTION_NAME,
            vectors_config={
                "text": VectorParams(size=cfg.TEXT_EMBEDDING_DIM, distance=Distance.COSINE),
                "visual": VectorParams(size=cfg.VISUAL_EMBEDDING_DIM, distance=Distance.COSINE),
            },
        )
        logger.info("Created Qdrant collection '%s'", cfg.COLLECTION_NAME)
    else:
        # validate existing config
        info = client.get_collection(cfg.COLLECTION_NAME)
        vectors_config = info.config.params.vectors
        expected = {
            "text": cfg.TEXT_EMBEDDING_DIM,
            "visual": cfg.VISUAL_EMBEDDING_DIM,
        }
        for name, dim in expected.items():
            if name not in vectors_config:
                raise RuntimeError(
                    f"Collection '{cfg.COLLECTION_NAME}' exists but missing "
                    f"vector '{name}'. Delete and recreate the collection."
                )
            if vectors_config[name].size != dim:
                raise RuntimeError(
                    f"Collection '{cfg.COLLECTION_NAME}' has vector '{name}' "
                    f"with size {vectors_config[name].size}, expected {dim}. "
                    "Delete and recreate the collection."
                )
        logger.info("Collection '%s' exists with correct config", cfg.COLLECTION_NAME)


# ── video ID ─────────────────────────────────────────────────────────────────

def compute_video_id(video_path: str) -> str:
    """SHA256 hash of first 10 MB + file size → deterministic video ID."""
    h = hashlib.sha256()
    size = os.path.getsize(video_path)
    h.update(str(size).encode())
    with open(video_path, "rb") as f:
        h.update(f.read(10 * 1024 * 1024))
    return h.hexdigest()[:16]


def video_exists(video_id: str) -> bool:
    """Check if any points with this video_id exist in the collection."""
    client = _get_client()
    try:
        result = client.scroll(
            collection_name=cfg.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=1,
        )
        return len(result[0]) > 0
    except Exception:
        return False


def delete_video(video_id: str) -> int:
    """Delete all points for a video_id. Returns count deleted."""
    client = _get_client()
    # get all point IDs for this video
    points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=cfg.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=100,
            offset=offset,
        )
        batch, next_offset = result
        points.extend([p.id for p in batch])
        if next_offset is None:
            break
        offset = next_offset

    if points:
        client.delete(
            collection_name=cfg.COLLECTION_NAME,
            points_selector=points,
        )
    logger.info("Deleted %d points for video_id=%s", len(points), video_id)
    return len(points)


# ── upsert ───────────────────────────────────────────────────────────────────

def store_chunks(
    chunks: list[dict],
    text_embeddings: list[np.ndarray | None],
    visual_embeddings: list[dict],
    video_id: str,
    video_path: str,
    video_meta: dict,
    transcript_result: dict,
) -> int:
    """Batch-upsert chunk data into Qdrant.

    Returns number of points stored.
    """
    client = _get_client()

    filename = os.path.basename(video_path)
    title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip()
    upload_date = ""
    try:
        mtime = os.path.getmtime(video_path)
        from datetime import datetime, timezone
        upload_date = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass

    points = []
    skipped = 0

    for i, chunk in enumerate(chunks):
        text_emb = text_embeddings[i] if i < len(text_embeddings) else None
        vis_data = visual_embeddings[i] if i < len(visual_embeddings) else {}
        vis_emb = vis_data.get("embedding")

        # skip chunks with no vectors at all
        if text_emb is None and vis_emb is None:
            skipped += 1
            logger.warning(
                "Chunk %d has no text or visual embedding — skipping", i
            )
            continue

        # build named vectors (omit missing modalities)
        vectors = {}
        if text_emb is not None:
            vectors["text"] = text_emb.tolist()
        if vis_emb is not None:
            vectors["visual"] = vis_emb.tolist()

        # determine source
        has_text = text_emb is not None
        has_vis = vis_data.get("has_visual", False)
        has_ocr = chunk.get("has_ocr", False)
        has_transcript = bool(chunk.get("text", "").strip())

        if has_text and has_vis:
            source = "both"
        elif has_text:
            source = "transcript_only" if has_transcript else "ocr_only"
        else:
            source = "visual_only"

        # truncate long text for payload
        transcript_text = (chunk.get("text", "") or "")[:cfg.MAX_TEXT_LENGTH]
        ocr_text = (chunk.get("ocr_text", "") or "")[:cfg.MAX_TEXT_LENGTH]
        # strip null bytes
        transcript_text = transcript_text.replace("\x00", "")
        ocr_text = ocr_text.replace("\x00", "")

        payload = {
            "video_id": video_id,
            "video_filename": filename,
            "video_title": title,
            "video_duration": video_meta.get("duration", 0),
            "video_upload_date": upload_date,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "timestamp_start": chunk["start"],
            "timestamp_end": chunk["end"],
            "transcript_text": transcript_text,
            "ocr_text": ocr_text,
            "detected_language": transcript_result.get("language", "unknown"),
            "has_audio": video_meta.get("has_audio", False),
            "has_transcript": has_transcript,
            "has_visual": has_vis,
            "has_ocr": has_ocr,
            "source": source,
            "neighbor_density": chunk.get("neighbor_density", 1.0),
            "keyframe_count": vis_data.get("keyframe_count", 0),
            "keyframe_timestamps": [
                kf["timestamp"] for kf in chunk.get("keyframes", [])
            ],
            "prev_context": chunk.get("prev_context", ""),
            "next_context": chunk.get("next_context", ""),
            "low_confidence": transcript_result.get("low_confidence", True),
        }

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors,
            payload=payload,
        ))

    # batch upsert
    stored = 0
    for i in range(0, len(points), cfg.UPSERT_BATCH_SIZE):
        batch = points[i: i + cfg.UPSERT_BATCH_SIZE]
        try:
            client.upsert(collection_name=cfg.COLLECTION_NAME, points=batch)
            stored += len(batch)
        except Exception as exc:
            logger.warning("Upsert batch failed: %s — retrying once", exc)
            try:
                client.upsert(collection_name=cfg.COLLECTION_NAME, points=batch)
                stored += len(batch)
            except Exception as exc2:
                logger.error("Upsert retry failed: %s — %d points lost", exc2, len(batch))

    logger.info("Stored %d points in Qdrant (%d skipped)", stored, skipped)
    return stored
