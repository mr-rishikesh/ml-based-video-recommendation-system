import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}

# ──────────────────────────────────────────────
# Video processing — Keyframe extraction
# ──────────────────────────────────────────────
SCENE_CHANGE_THRESHOLD = 0.3        # ffmpeg scene detection sensitivity (0‑1, lower = more frames)
MAX_KEYFRAMES_PER_MINUTE = 10       # cap for fast‑cut videos
FALLBACK_FRAME_INTERVAL = 10        # seconds — used when scene detection yields 0 frames
AUDIO_SAMPLE_RATE = 16000           # Hz
MAX_FRAME_RESOLUTION = 720          # resize frames if height exceeds this

# ──────────────────────────────────────────────
# Whisper
# ──────────────────────────────────────────────
WHISPER_MODEL = "base"
WHISPER_CONFIDENCE_THRESHOLD = 0.3  # below this → treat as "no speech"
WHISPER_SEGMENT_MAX_SECONDS = 1800  # 30‑min windows for long audio
WHISPER_OVERLAP_SECONDS = 10        # overlap between windows

# ──────────────────────────────────────────────
# Semantic chunking
# ──────────────────────────────────────────────
MIN_CHUNK_DURATION = 30             # seconds
MAX_CHUNK_DURATION = 120            # seconds
SIMILARITY_STD_FACTOR = 0.5
BOUNDARY_PHRASES = [
    "now let's", "moving on", "next topic", "let's switch",
    "on the other hand", "in contrast", "compared to",
    "let's talk about", "another thing", "the next step",
    "let me show you", "switching to", "next we",
]
PAUSE_THRESHOLD = 2.0               # seconds gap → hard boundary

# ──────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.5
OCR_DEDUP_THRESHOLD = 0.8           # text‑overlap ratio for dedup
BLANK_FRAME_THRESHOLD_LOW = 10      # mean pixel value
BLANK_FRAME_THRESHOLD_HIGH = 245

# ──────────────────────────────────────────────
# CLIP
# ──────────────────────────────────────────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_BATCH_SIZE = 32
CLIP_CLUSTER_THRESHOLD = 5          # if chunk has > N keyframes, cluster to 3

# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM = 384
VISUAL_EMBEDDING_DIM = 512
MAX_TEXT_LENGTH = 10000              # chars — payload truncation

# ──────────────────────────────────────────────
# Qdrant
# ──────────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "video_chunks"
UPSERT_BATCH_SIZE = 100

# ──────────────────────────────────────────────
# Query / Re‑ranking
# ──────────────────────────────────────────────
DEFAULT_TOP_K = 5
SEARCH_CANDIDATES = 20              # fetch top‑N from each modality before re‑rank
TEXT_WEIGHT = 0.6
VISUAL_WEIGHT = 0.3
DENSITY_WEIGHT = 0.1
TIME_DECAY_OLD_DAYS = 730           # 2 years
TIME_DECAY_NEW_DAYS = 180           # 6 months
TIME_DECAY_OLD_FACTOR = 0.8
TIME_DECAY_NEW_FACTOR = 1.1
TITLE_BOOST_FACTOR = 1.5
LOW_SCORE_THRESHOLD = 0.3

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────
KEEP_TEMP_FILES = False
