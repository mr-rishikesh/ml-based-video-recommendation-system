"""Video processing: audio extraction + keyframe extraction via scene detection."""

import glob
import logging
import os
import re
import subprocess

import config as cfg

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ffmpeg_timeout(duration: float) -> int:
    """Generous timeout: 2× duration + 60 s (min 120 s)."""
    return max(120, int(duration * 2 + 60))


def _parse_showinfo_pts(stderr: str) -> list[float]:
    """Parse pts_time values from ffmpeg showinfo filter output."""
    # showinfo prints lines like:  n:   0 pts:   1234 pts_time:1.234 ...
    timestamps = []
    for match in re.finditer(r"pts_time:\s*([\d.]+)", stderr):
        timestamps.append(float(match.group(1)))
    return sorted(set(timestamps))


# ── public API ───────────────────────────────────────────────────────────────

def extract_audio(video_path: str, video_id: str, duration: float) -> str | None:
    """Extract mono 16 kHz WAV audio. Returns path or None if no audio."""
    out_path = os.path.join(cfg.AUDIO_DIR, f"{video_id}.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                          # no video
        "-ac", "1",                     # mono
        "-ar", str(cfg.AUDIO_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        out_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_ffmpeg_timeout(duration),
        )
    except subprocess.TimeoutExpired:
        logger.error("Audio extraction timed out")
        return None

    if result.returncode != 0:
        # likely "no audio stream"
        logger.warning("Audio extraction failed: %s", result.stderr[:200])
        return None

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        logger.warning("Audio file missing or too small — skipping")
        return None

    logger.info("Audio extracted: %s", out_path)
    return out_path


def extract_keyframes(
    video_path: str,
    video_id: str,
    duration: float,
    height: int,
) -> list[dict]:
    """Extract keyframes via scene-change detection.

    Returns list of {"path": str, "timestamp": float} sorted by timestamp.
    """
    frames_dir = os.path.join(cfg.FRAMES_DIR, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    # ── determine scale filter ──
    scale_filter = ""
    if height > cfg.MAX_FRAME_RESOLUTION:
        scale_filter = f",scale=-1:{cfg.MAX_FRAME_RESOLUTION}"

    # ── Pass 1: scene-change detection ──
    vf = f"select='gt(scene\\,{cfg.SCENE_CHANGE_THRESHOLD})',showinfo{scale_filter}"
    out_pattern = os.path.join(frames_dir, "kf_%05d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-vsync", "vfn",
        "-q:v", "2",
        out_pattern,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_ffmpeg_timeout(duration),
        )
    except subprocess.TimeoutExpired:
        logger.error("Keyframe extraction timed out")
        return _fallback_extraction(video_path, video_id, duration, height)

    if result.returncode != 0:
        logger.warning(
            "Scene detection failed (%s), falling back to interval sampling",
            result.stderr[:200],
        )
        return _fallback_extraction(video_path, video_id, duration, height)

    # parse timestamps from showinfo output (written to stderr by ffmpeg)
    timestamps = _parse_showinfo_pts(result.stderr)

    # map files to timestamps
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "kf_*.jpg")))
    keyframes = []
    for i, fpath in enumerate(frame_files):
        ts = timestamps[i] if i < len(timestamps) else i * cfg.FALLBACK_FRAME_INTERVAL
        # rename to encode timestamp
        new_name = f"frame_{ts:010.3f}.jpg"
        new_path = os.path.join(frames_dir, new_name)
        os.rename(fpath, new_path)
        keyframes.append({"path": new_path, "timestamp": ts})

    # ── cap excessive keyframes (fast-cut videos) ──
    max_total = int(max(1, (duration / 60) * cfg.MAX_KEYFRAMES_PER_MINUTE))
    if len(keyframes) > max_total:
        logger.warning(
            "Too many keyframes (%d), subsampling to %d",
            len(keyframes), max_total,
        )
        step = len(keyframes) / max_total
        indices = [int(i * step) for i in range(max_total)]
        # remove unselected files
        selected = set(indices)
        for i, kf in enumerate(keyframes):
            if i not in selected:
                try:
                    os.remove(kf["path"])
                except OSError:
                    pass
        keyframes = [keyframes[i] for i in indices]

    # ── Pass 2: if no keyframes at all, fallback ──
    if not keyframes:
        logger.warning("Scene detection found 0 keyframes — using interval fallback")
        return _fallback_extraction(video_path, video_id, duration, height)

    logger.info("Keyframes extracted: %d frames", len(keyframes))
    return keyframes


def _fallback_extraction(
    video_path: str,
    video_id: str,
    duration: float,
    height: int,
) -> list[dict]:
    """Extract frames at fixed intervals when scene detection fails/yields nothing."""
    frames_dir = os.path.join(cfg.FRAMES_DIR, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    interval = cfg.FALLBACK_FRAME_INTERVAL
    scale_filter = ""
    if height > cfg.MAX_FRAME_RESOLUTION:
        scale_filter = f",scale=-1:{cfg.MAX_FRAME_RESOLUTION}"

    vf = f"fps=1/{interval}{scale_filter}"
    out_pattern = os.path.join(frames_dir, "fb_%05d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-q:v", "2",
        out_pattern,
    ]

    try:
        subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_ffmpeg_timeout(duration),
        )
    except subprocess.TimeoutExpired:
        logger.error("Fallback frame extraction timed out")
        return []

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "fb_*.jpg")))
    keyframes = []
    for i, fpath in enumerate(frame_files):
        ts = i * interval
        new_name = f"frame_{ts:010.3f}.jpg"
        new_path = os.path.join(frames_dir, new_name)
        os.rename(fpath, new_path)
        keyframes.append({"path": new_path, "timestamp": ts})

    logger.info("Fallback frames extracted: %d frames (every %ds)", len(keyframes), interval)
    return keyframes


def extract_fallback_frame_at(
    video_path: str,
    video_id: str,
    timestamp: float,
    height: int,
) -> dict | None:
    """Extract a single frame at a specific timestamp (for chunks with no keyframes)."""
    frames_dir = os.path.join(cfg.FRAMES_DIR, video_id)
    os.makedirs(frames_dir, exist_ok=True)

    scale_filter = ""
    if height > cfg.MAX_FRAME_RESOLUTION:
        scale_filter = f",scale=-1:{cfg.MAX_FRAME_RESOLUTION}"

    out_path = os.path.join(frames_dir, f"frame_{timestamp:010.3f}.jpg")
    vf = f"select='eq(n\\,0)'{scale_filter}" if not scale_filter else f"scale=-1:{cfg.MAX_FRAME_RESOLUTION}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
    ]
    if height > cfg.MAX_FRAME_RESOLUTION:
        cmd += ["-vf", f"scale=-1:{cfg.MAX_FRAME_RESOLUTION}"]
    cmd.append(out_path)

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.warning("Failed to extract fallback frame at %.1fs: %s", timestamp, exc)
        return None

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return {"path": out_path, "timestamp": timestamp}
    return None
