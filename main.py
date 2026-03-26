"""Main pipeline: video → extract → transcribe → chunk → OCR → CLIP → store → query."""

import argparse
import logging
import sys
import time

# add project root to path so 'config' and 'utils' are importable
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding for Unicode characters in progress bars / logs
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config as cfg
from utils.validation import run_all_checks
from utils.storage import compute_video_id, ensure_collection, video_exists, delete_video, store_chunks
from utils.video import extract_audio, extract_keyframes, extract_fallback_frame_at
from utils.audio import transcribe
from utils.chunking import create_chunks
from utils.ocr import extract_ocr_for_chunk
from utils.visual import extract_visual_embedding
from utils.embeddings import generate_text_embeddings
from utils.query import search, format_results
from utils.cleanup import cleanup_video_files

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _timer():
    """Simple context-manager-style timer."""
    start = time.time()
    return lambda: round(time.time() - start, 1)


def run_pipeline(
    video_path: str,
    skip_ocr: bool = False,
    skip_visual: bool = False,
    reindex: bool = False,
    keep_temp: bool = False,
) -> dict:
    """Run the full ingestion pipeline.

    Returns summary dict with stats.
    """
    if keep_temp:
        cfg.KEEP_TEMP_FILES = True

    video_path = os.path.realpath(video_path)
    video_id = compute_video_id(video_path)
    logger.info("Video ID: %s", video_id)

    summary = {"video_id": video_id, "steps": {}}

    try:
        # ── 0. Validate ──
        t = _timer()
        video_meta, device = run_all_checks(video_path)
        summary["steps"]["validation"] = t()

        # ── duplicate check ──
        ensure_collection()
        if video_exists(video_id):
            if reindex:
                logger.info("Re-indexing: deleting existing data for this video")
                delete_video(video_id)
            else:
                logger.info("Video already indexed (use --reindex to overwrite)")
                return summary

        # ── 1. Extract audio ──
        t = _timer()
        audio_path = None
        if video_meta["has_audio"]:
            audio_path = extract_audio(video_path, video_id, video_meta["duration"])
        else:
            logger.info("No audio stream — skipping audio extraction")
        summary["steps"]["audio_extraction"] = t()

        # ── 2. Extract keyframes ──
        t = _timer()
        keyframes = extract_keyframes(
            video_path, video_id,
            video_meta["duration"], video_meta["height"],
        )
        summary["steps"]["keyframe_extraction"] = t()
        summary["keyframe_count"] = len(keyframes)
        logger.info("Keyframes: %d", len(keyframes))

        # ── 3. Transcribe ──
        t = _timer()
        transcript_result = transcribe(audio_path, video_meta["duration"], device=device)
        summary["steps"]["transcription"] = t()
        summary["segment_count"] = len(transcript_result["segments"])
        logger.info("Segments: %d", len(transcript_result["segments"]))

        # ── 4. Semantic chunking ──
        t = _timer()
        chunks = create_chunks(transcript_result["segments"], video_meta["duration"])
        summary["steps"]["chunking"] = t()
        summary["chunk_count"] = len(chunks)

        # ── 5. Map keyframes to chunks ──
        for chunk in chunks:
            chunk["keyframes"] = [
                kf for kf in keyframes
                if chunk["start"] <= kf["timestamp"] < chunk["end"]
            ]
            # Pass 2: fallback frame for chunks with no keyframes
            if not chunk["keyframes"]:
                midpoint = (chunk["start"] + chunk["end"]) / 2
                fallback = extract_fallback_frame_at(
                    video_path, video_id, midpoint, video_meta["height"],
                )
                if fallback:
                    chunk["keyframes"] = [fallback]

        # ── 6. OCR per chunk ──
        t = _timer()
        if not skip_ocr:
            ocr_count = 0
            for chunk in chunks:
                ocr_result = extract_ocr_for_chunk(
                    chunk["keyframes"],
                    detected_language=transcript_result.get("language", "en"),
                )
                chunk["ocr_text"] = ocr_result["text"]
                chunk["has_ocr"] = ocr_result["has_ocr"]
                if ocr_result["has_ocr"]:
                    ocr_count += 1
            summary["ocr_chunks"] = ocr_count
            logger.info("OCR: %d/%d chunks have text", ocr_count, len(chunks))
        else:
            for chunk in chunks:
                chunk["ocr_text"] = ""
                chunk["has_ocr"] = False
            logger.info("OCR: skipped (--skip-ocr)")
        summary["steps"]["ocr"] = t()

        # ── 7. Visual features per chunk ──
        t = _timer()
        visual_embeddings = []
        if not skip_visual:
            vis_count = 0
            for chunk in chunks:
                vis_result = extract_visual_embedding(chunk["keyframes"], device=device)
                visual_embeddings.append(vis_result)
                if vis_result["has_visual"]:
                    vis_count += 1
            summary["visual_chunks"] = vis_count
            logger.info("Visual: %d/%d chunks have embeddings", vis_count, len(chunks))
        else:
            visual_embeddings = [{"embedding": None, "has_visual": False, "keyframe_count": 0}] * len(chunks)
            logger.info("Visual: skipped (--skip-visual)")
        summary["steps"]["visual"] = t()

        # ── 8. Text embeddings ──
        t = _timer()
        text_embeddings = generate_text_embeddings(chunks)
        summary["steps"]["text_embeddings"] = t()

        # ── 9. Store in Qdrant ──
        t = _timer()
        stored = store_chunks(
            chunks, text_embeddings, visual_embeddings,
            video_id, video_path, video_meta, transcript_result,
        )
        summary["steps"]["storage"] = t()
        summary["stored_points"] = stored

        # ── summary ──
        logger.info("=" * 50)
        logger.info("Pipeline complete for: %s", os.path.basename(video_path))
        logger.info("  Chunks: %d", len(chunks))
        logger.info("  Stored: %d points", stored)
        logger.info("  Modalities: audio=%s, ocr=%s, visual=%s",
                     video_meta["has_audio"],
                     not skip_ocr,
                     not skip_visual)
        logger.info("  Timings: %s",
                     " | ".join(f"{k}={v}s" for k, v in summary["steps"].items()))
        logger.info("=" * 50)

    finally:
        # cleanup always runs
        cleanup_video_files(video_id)

    return summary


def _detect_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Video Retrieval Pipeline",
    )

    # ── two modes via subcommands ──
    subparsers = parser.add_subparsers(dest="command")

    # ── mode 1: ingest (upload video) ──
    ingest_parser = subparsers.add_parser("ingest", help="Index a video into the database")
    ingest_parser.add_argument("video", help="Path to video file")
    ingest_parser.add_argument("--keep-temp", action="store_true", help="Keep extracted frames/audio")
    ingest_parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR step")
    ingest_parser.add_argument("--skip-visual", action="store_true", help="Skip CLIP visual features")
    ingest_parser.add_argument("--reindex", action="store_true", help="Force re-index if video exists")

    # ── mode 2: query (search only) ──
    query_parser = subparsers.add_parser("query", help="Search across all indexed videos")
    query_parser.add_argument("text", help="Search query text")
    query_parser.add_argument("--top-k", type=int, default=cfg.DEFAULT_TOP_K, help="Number of results")
    query_parser.add_argument("--video-id", type=str, default=None, help="Search within a specific video")
    query_parser.add_argument("--language", type=str, default=None, help="Filter by language")

    args = parser.parse_args()

    if args.command == "ingest":
        run_pipeline(
            video_path=args.video,
            skip_ocr=args.skip_ocr,
            skip_visual=args.skip_visual,
            reindex=args.reindex,
            keep_temp=args.keep_temp,
        )

    elif args.command == "query":
        device = _detect_device()
        results = search(
            args.text,
            device=device,
            top_k=args.top_k,
            video_id=args.video_id,
            language=args.language,
        )
        print(format_results(results))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
