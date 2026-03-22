"""CLIP-based visual feature extraction from keyframes."""

import logging

import numpy as np
import torch
from PIL import Image

import config as cfg

logger = logging.getLogger(__name__)

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_device = "cpu"


def _load_clip(device: str = "cpu"):
    """Lazy-load CLIP model."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _device
    if _clip_model is not None:
        return

    import open_clip

    _device = device
    logger.info("Loading CLIP model '%s' ...", cfg.CLIP_MODEL_NAME)
    try:
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            cfg.CLIP_MODEL_NAME, pretrained=cfg.CLIP_PRETRAINED, device=device,
        )
        _clip_tokenizer = open_clip.get_tokenizer(cfg.CLIP_MODEL_NAME)
        _clip_model.eval()
    except Exception as exc:
        logger.warning("CLIP load failed on %s (%s), falling back to CPU", device, exc)
        _device = "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            cfg.CLIP_MODEL_NAME, pretrained=cfg.CLIP_PRETRAINED, device="cpu",
        )
        _clip_tokenizer = open_clip.get_tokenizer(cfg.CLIP_MODEL_NAME)
        _clip_model.eval()


def _is_blank_frame(img: np.ndarray) -> bool:
    mean_val = np.mean(img)
    return mean_val < cfg.BLANK_FRAME_THRESHOLD_LOW or mean_val > cfg.BLANK_FRAME_THRESHOLD_HIGH


def _embed_images(images: list[Image.Image]) -> np.ndarray:
    """Encode a batch of PIL images through CLIP. Returns (N, 512) array."""
    if not images:
        return np.array([])

    batch_size = cfg.CLIP_BATCH_SIZE
    all_embs = []

    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size]
        tensors = torch.stack([_clip_preprocess(img) for img in batch]).to(_device)
        try:
            with torch.no_grad():
                embs = _clip_model.encode_image(tensors)
            embs = embs.cpu().numpy()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning("CLIP OOM — processing on CPU for this batch")
                tensors = tensors.cpu()
                _clip_model.cpu()
                with torch.no_grad():
                    embs = _clip_model.encode_image(tensors).numpy()
                if _device == "cuda":
                    _clip_model.to(_device)
            else:
                raise
        all_embs.append(embs)

    result = np.vstack(all_embs).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return result / norms


def encode_text(text: str, device: str = "cpu") -> np.ndarray | None:
    """Encode a query string through CLIP text encoder. Returns (512,) or None."""
    _load_clip(device)
    try:
        tokens = _clip_tokenizer([text]).to(_device)
        with torch.no_grad():
            emb = _clip_model.encode_text(tokens)
        emb = emb.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb
    except Exception as exc:
        logger.error("CLIP text encoding failed: %s", exc)
        return None


def extract_visual_embedding(
    keyframes: list[dict],
    device: str = "cpu",
) -> dict:
    """Compute a single visual embedding for a chunk from its keyframes.

    Strategy:
      N=0: has_visual=False
      N=1: use that embedding directly
      N=2-5: average embeddings
      N>5: k-means cluster to 3, average centroids

    Returns: {"embedding": np.ndarray or None, "has_visual": bool, "keyframe_count": int}
    """
    _load_clip(device)

    # filter out blank/corrupt frames
    valid_images = []
    for kf in keyframes:
        try:
            img = Image.open(kf["path"]).convert("RGB")
            arr = np.array(img)
            if _is_blank_frame(arr):
                continue
            valid_images.append(img)
        except Exception as exc:
            logger.warning("Skipping frame %s: %s", kf["path"], exc)

    if not valid_images:
        return {"embedding": None, "has_visual": False, "keyframe_count": 0}

    embeddings = _embed_images(valid_images)

    if len(embeddings) == 0:
        return {"embedding": None, "has_visual": False, "keyframe_count": 0}

    if len(embeddings) == 1:
        final = embeddings[0]
    elif len(embeddings) <= cfg.CLIP_CLUSTER_THRESHOLD:
        final = np.mean(embeddings, axis=0)
    else:
        # cluster to 3 centroids, then average
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=3, n_init=3, random_state=42)
            km.fit(embeddings)
            final = np.mean(km.cluster_centers_, axis=0)
        except Exception:
            final = np.mean(embeddings, axis=0)

    # L2 normalize final
    norm = np.linalg.norm(final)
    if norm > 0:
        final = final / norm

    return {
        "embedding": final.astype(np.float32),
        "has_visual": True,
        "keyframe_count": len(valid_images),
    }
