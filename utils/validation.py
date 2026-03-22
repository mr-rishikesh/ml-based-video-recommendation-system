"""Pre-pipeline validation: dependencies, input file, disk space, GPU."""

import logging
import os
import shutil
import subprocess
import json

import config as cfg

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )


# ── public API ───────────────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    """Ensure ffmpeg and ffprobe are on PATH."""
    for binary in ("ffmpeg", "ffprobe"):
        try:
            _run([binary, "-version"])
        except FileNotFoundError:
            raise RuntimeError(
                f"{binary} not found on PATH. "
                "Install it: https://ffmpeg.org/download.html"
            )
    logger.info("ffmpeg/ffprobe: OK")


def check_qdrant() -> None:
    """Verify Qdrant server is reachable."""
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(host=cfg.QDRANT_HOST, port=cfg.QDRANT_PORT, timeout=5)
        client.get_collections()
        logger.info("Qdrant: OK (%s:%s)", cfg.QDRANT_HOST, cfg.QDRANT_PORT)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Qdrant at {cfg.QDRANT_HOST}:{cfg.QDRANT_PORT} — {exc}. "
            "Make sure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)."
        )


def check_gpu() -> str:
    """Return 'cuda' if available, else 'cpu' with a warning."""
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.warning("No CUDA GPU detected — models will run on CPU (slower)")
    return device


def validate_video_file(path: str) -> dict:
    """Validate the input file and return ffprobe metadata.

    Returns dict with keys: duration, width, height, has_audio, codec, fps.
    Raises RuntimeError on invalid input.
    """
    # resolve symlinks / normalise
    path = os.path.realpath(path)

    if not os.path.exists(path):
        raise RuntimeError(f"File not found: {path}")

    if os.path.getsize(path) == 0:
        raise RuntimeError(f"File is empty (0 bytes): {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in cfg.VALID_VIDEO_EXTENSIONS:
        raise RuntimeError(
            f"Unsupported extension '{ext}'. "
            f"Allowed: {', '.join(sorted(cfg.VALID_VIDEO_EXTENSIONS))}"
        )

    # ── ffprobe validation ──
    try:
        result = _run([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            path,
        ], timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out on {path}")

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("ffprobe returned invalid JSON")

    streams = probe.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise RuntimeError(
            "No video stream found — this may be an audio-only file."
        )

    vs = video_streams[0]
    fmt = probe.get("format", {})

    duration = float(fmt.get("duration", 0) or 0)
    if duration <= 0:
        # fallback: try stream-level duration or nb_frames
        duration = float(vs.get("duration", 0) or 0)

    width = int(vs.get("width", 0))
    height = int(vs.get("height", 0))
    codec = vs.get("codec_name", "unknown")

    # fps
    r_frame_rate = vs.get("r_frame_rate", "0/1")
    try:
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) else 0
    except (ValueError, ZeroDivisionError):
        fps = 0

    has_audio = len(audio_streams) > 0

    meta = dict(
        duration=duration,
        width=width,
        height=height,
        has_audio=has_audio,
        codec=codec,
        fps=round(fps, 2),
    )

    if duration <= 0:
        logger.warning("Could not determine video duration from metadata")

    if not has_audio:
        logger.warning("Video has no audio stream — transcription will be skipped")

    size_gb = os.path.getsize(path) / (1024 ** 3)
    if size_gb > 10:
        logger.warning("Large file (%.1f GB) — processing may be slow", size_gb)

    logger.info(
        "Video validated: %s | %.1fs | %dx%d | %s | audio=%s",
        os.path.basename(path), duration, width, height, codec, has_audio,
    )
    return meta


def check_disk_space(video_duration: float) -> None:
    """Warn if free disk space is likely insufficient."""
    # rough estimate: keyframes are much smaller than 1-FPS, but be conservative
    estimated_mb = (video_duration / 10) * 0.05 + 50  # ~50 KB per keyframe + audio
    free_mb = shutil.disk_usage(cfg.DATA_DIR).free / (1024 ** 2)
    if free_mb < estimated_mb * 1.2:
        logger.warning(
            "Low disk space: %.0f MB free, estimated need ≈%.0f MB",
            free_mb, estimated_mb,
        )
    else:
        logger.info("Disk space: OK (%.0f MB free)", free_mb)


def run_all_checks(video_path: str) -> tuple[dict, str]:
    """Run every validation check.

    Returns (video_metadata_dict, device_string).
    """
    check_ffmpeg()
    check_qdrant()
    device = check_gpu()
    meta = validate_video_file(video_path)

    # ensure data dirs exist
    os.makedirs(cfg.FRAMES_DIR, exist_ok=True)
    os.makedirs(cfg.AUDIO_DIR, exist_ok=True)

    check_disk_space(meta["duration"])
    return meta, device
