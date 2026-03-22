"""Temp file cleanup for extracted frames and audio."""

import logging
import os
import shutil

import config as cfg

logger = logging.getLogger(__name__)


def cleanup_video_files(video_id: str) -> None:
    """Delete extracted frames and audio for a specific video."""
    if cfg.KEEP_TEMP_FILES:
        logger.info("KEEP_TEMP_FILES=True — skipping cleanup")
        return

    # frames directory
    frames_dir = os.path.join(cfg.FRAMES_DIR, video_id)
    if os.path.isdir(frames_dir):
        try:
            shutil.rmtree(frames_dir)
            logger.info("Cleaned up frames: %s", frames_dir)
        except OSError as exc:
            logger.warning("Failed to remove frames dir: %s", exc)

    # audio file
    audio_path = os.path.join(cfg.AUDIO_DIR, f"{video_id}.wav")
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            logger.info("Cleaned up audio: %s", audio_path)
        except OSError as exc:
            logger.warning("Failed to remove audio file: %s", exc)
