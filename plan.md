You are a senior Python engineer. Build a clean, production-ready multimodal video retrieval pipeline.

The system takes a video file as input and creates a searchable vector database using audio, OCR, and visual features.

IMPORTANT CONSTRAINTS:

* Keep implementation SIMPLE and clean (avoid overengineering)
* Code must be modular and readable
* Handle ALL edge cases robustly — never crash, always degrade gracefully
* Use Python only
* Use functions and clear file structure
* No unnecessary abstractions

---

## GOAL

Given a video file (or batch of videos):

1. Validate input and check dependencies
2. Extract audio and frames
3. Generate timestamped transcript using Whisper
4. Create semantic chunks using similarity-based merging with boundary detection
5. Extract OCR and visual features per chunk
6. Compute topic density scores per chunk
7. Generate embeddings
8. Store in Qdrant with rich metadata
9. Provide an advanced query function with re-ranking

---

## PROJECT STRUCTURE

Create this structure:

```
project/
|
├── main.py
├── config.py
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── video.py
│   ├── audio.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── ocr.py
│   ├── visual.py
│   ├── storage.py
│   ├── query.py
│   ├── validation.py
│   └── cleanup.py
│
└── data/
    ├── frames/
    └── audio/
```

---

## 0. VALIDATION & DEPENDENCY CHECKS (validation.py)

Run these checks BEFORE the pipeline starts:

* Verify ffmpeg is installed and accessible on PATH
  - If missing: raise clear error with install instructions
* Verify Qdrant is reachable (connection test with timeout)
  - If down: raise clear error, do not silently fail
* Validate input file:
  - File exists on disk
  - File is not empty (0 bytes)
  - File extension is a known video format (.mp4, .avi, .mkv, .mov, .webm, .flv, .wmv)
  - File is actually a valid video (probe with ffprobe, check for video stream)
  - Reject non-video files (images, text, audio-only) with clear message
* Validate disk space:
  - Estimate required space: (video_duration_sec * 1 FPS * ~50KB/frame) + audio size
  - Warn if available disk space is below estimate + 20% buffer
* Check GPU availability (optional, for CLIP/Whisper):
  - If CUDA available, use it; otherwise fallback to CPU with a log warning

Edge cases:
* Symlinks pointing to deleted files -> resolve and validate the real path
* File path with unicode/special characters -> normalize path before processing
* Read-only file -> still works (we only read the video, never write to it)
* Network-mounted paths -> warn about potential slowness

---

## 1. VIDEO PROCESSING (video.py)

* Use ffmpeg via subprocess
* Extract:
  * Mono audio at 16kHz WAV format
  * **Keyframes ONLY — NOT fixed 1 FPS** (see below)

### WHY NOT 1 FPS?

1 FPS on a 1-hour video = 3,600 frames. In a typical lecture, the same slide stays
on screen for 30-60 seconds. That means ~30-60 nearly identical frames per slide,
wasting disk, compute, and producing redundant CLIP embeddings that add noise
when averaged.

### KEYFRAME EXTRACTION STRATEGY (TWO-PASS)

**Pass 1: Scene-change detection (ffmpeg)**
```
ffmpeg -i input.mp4 -vf "select='gt(scene,0.3)',showinfo" -vsync vfn frame_%05d.jpg
```
This extracts a frame only when the visual content changes significantly
(scene change threshold = 0.3). A 1-hour lecture with 20 slides produces ~25-40
keyframes instead of 3,600.

**Pass 2: Minimum sampling guarantee**
Scene detection can miss slow transitions or videos with no cuts (e.g., a talking head).
After Pass 1, check: if any chunk's time range has ZERO keyframes, extract one frame
at the midpoint of that chunk as a fallback.

Guaranteed: at least 1 frame per chunk, at most ~1 frame per 5 seconds of new content.

**Timestamp preservation:**
Each keyframe's filename encodes its timestamp: `frame_00042.300.jpg` = 42.3 seconds.
ffmpeg's showinfo filter outputs the timestamp; parse it and rename files accordingly.

### Benefits:
* 1-hour video: ~30-80 keyframes instead of 3,600 (50-100x reduction)
* Each keyframe represents a DISTINCT visual state (new slide, new scene, new diagram)
* CLIP embeddings per chunk are meaningful — each represents different visual content
* OCR runs on fewer frames with more unique text
* Disk usage drops from ~180MB to ~2-5MB for frames

* Probe video metadata FIRST using ffprobe:
  - Get duration, resolution, codec, frame count, audio stream presence
  - Store metadata for later use (payload enrichment)

* Handle:
  * Missing audio stream -> set has_audio=False, skip transcription, continue pipeline
  * Corrupted/unreadable frames -> skip frame, log warning, continue
  * Very short video (<5 sec) -> extract all frames, produce single chunk
  * Very long video (>2 hours) -> process in segments to limit memory:
    - Extract audio in 30-minute segments for Whisper
    - Keyframe extraction scales naturally (long video ≠ more frames if content is static)
  * Scene detection produces TOO MANY frames (fast cuts, music video) ->
    cap at MAX_KEYFRAMES_PER_MINUTE = 10; if exceeded, subsample uniformly
  * Scene detection produces ZERO frames (static video, e.g., screencast with no cuts) ->
    fallback to 1 frame every 10 seconds
  * Videos with multiple audio tracks -> use first audio track
  * Non-standard codecs -> let ffmpeg handle, catch decode errors
  * Variable frame rate video -> keyframe extraction handles this naturally
  * Interlaced video -> ffmpeg deinterlace filter
  * Zero-duration video (metadata says 0) -> use ffprobe frame count instead
  * Protected/DRM video -> catch error, log "cannot process protected content"
  * Extremely high resolution (4K+) -> resize frames to max 720p for processing
  * ffmpeg process hangs -> set subprocess timeout (duration * 2 + 60 seconds)
  * ffmpeg writes partial output then crashes -> validate output files exist and are non-empty

---

## 2. TRANSCRIPTION (audio.py)

* Use Whisper base model
* Return timestamped segments: [{start, end, text}]

* Language detection:
  - Let Whisper auto-detect language
  - Store detected language in metadata

* Handle:
  * Empty/silent audio -> return empty segment list, set has_audio=False
  * Audio with only music/noise, no speech -> Whisper returns garbled text
    - Detect: if average segment confidence < 0.3, treat as "no speech"
    - Set has_transcript=False
  * Very long audio (>2 hours) -> process in 30-minute overlapping windows
    - Overlap by 10 seconds to avoid cutting words at boundaries
    - Deduplicate segments from overlap regions (prefer higher confidence)
  * Whisper model download failure -> catch, retry once, then raise clear error
  * Whisper OOM on long audio -> fallback to processing in smaller windows
  * Non-English audio -> still works (Whisper is multilingual), store language tag
  * Audio with background noise -> Whisper handles this, but log warning if confidence is low
  * Segments with only filler words ("um", "uh", "like") -> keep them but flag low_content=True
  * Hallucinated repetitions (Whisper bug: repeating same phrase) ->
    detect: if same text appears in 3+ consecutive segments, deduplicate

---

## 3. SEMANTIC CHUNKING (chunking.py)

* Use SentenceTransformers (all-MiniLM-L6-v2) for segment embeddings
* Merge segments into chunks using TWO signals:

### A. Similarity-Based Merging (existing approach, improved)

CONDITIONS:
* min_duration = 30 sec
* max_duration = 120 sec
* dynamic similarity threshold:
  threshold = mean(similarity) - 0.5 * std

SPLIT WHEN:
* duration >= min_duration AND (similarity < threshold OR duration >= max_duration)

### B. Semantic Boundary Detection (NEW — from edgecases.md)

Before similarity merging, scan transcript for boundary signals:
* Transitional phrases: "now let's look at", "moving on to", "next topic",
  "let's switch to", "on the other hand", "in contrast", "compared to"
* Long pauses (>2 seconds gap between segments)
* Dramatic topic shift (cosine similarity between consecutive segments < 0.3)

These boundaries act as HARD SPLITS — the similarity merger will not merge across them.
This prevents the Information Splintering Problem where related content
(e.g., "React vs Angular" comparison) gets split across chunks.

### C. Chunk Overlap (NEW)

After chunking, create overlapping context:
* Store the LAST sentence of the previous chunk as "prev_context"
* Store the FIRST sentence of the next chunk as "next_context"
* These go into the payload (not the embedding) for context-aware retrieval

Edge cases:
* No transcript segments at all -> fallback to fixed-duration chunks (every 60 sec)
  using frame timestamps only
* Single segment -> single chunk
* Very short video (<30 sec) -> single chunk, skip similarity computation
* All segments nearly identical (e.g., looping video) -> single chunk
* Huge number of tiny segments (>1000) -> batch embedding computation, 64 at a time
* Segments with overlapping timestamps (Whisper bug) -> sort by start, resolve overlaps
* Gap in timestamps (e.g., segments jump from 60s to 180s) -> treat gap as boundary

---

## 4. TOPIC DENSITY SCORING (NEW — from edgecases.md)

### The Problem: "Passing Mention" vs. "Deep Explanation"

A chunk that briefly mentions "JWT authentication" should rank lower than a chunk
that deeply explains it, even if the brief mention contains the exact phrase.

### Implementation (per chunk):

1. For each chunk, look at the N surrounding chunks (window_size = 2 on each side)
2. Compute cosine similarity between the query embedding and each chunk in the window
3. topic_density = count of neighbors with similarity > 0.5 / window_size
4. A chunk surrounded by topically similar chunks = deep explanation (density ~ 1.0)
5. An isolated mention surrounded by unrelated chunks = passing mention (density ~ 0.0)

Store topic_density as a precomputed payload field:
* For each chunk, compute density relative to its neighbors at index time
* neighbor_similarity = avg cosine similarity to adjacent chunks
* Store as: chunk.payload.neighbor_density = float (0.0 to 1.0)

At query time:
* final_score = base_score * (0.7 + 0.3 * neighbor_density)
* This boosts chunks that are part of a sustained discussion on a topic

Edge cases:
* First/last chunk has fewer neighbors -> use available neighbors only
* Single chunk video -> density = 1.0 (only chunk, must be relevant)
* All chunks nearly identical -> density = 1.0 for all (fine, no penalty)
* Short but complete topic (instructor explains in 45s = 1 chunk, then moves on) ->
  low neighbor_density because neighbors are about different topics, BUT the chunk
  has substantial text content (>200 chars). Solution: content_richness_boost —
  if chunk text > 200 chars and density < 0.5, add boost = min(0.3, text_len/2000).
  This ensures a concise-but-complete explanation isn't unfairly penalised.

---

## 5. OCR (ocr.py)

* Use EasyOCR
* Extract text from frames belonging to each chunk
* Filter detections with confidence < 0.5

* Smart frame sampling for OCR:
  - With keyframe extraction, each frame already represents a distinct visual state
  - OCR ALL keyframes in the chunk (they are already sparse and unique)
  - No need for time-based sampling — keyframes ARE the meaningful frames

* Text deduplication:
  - Slides/code on screen often repeat across many frames
  - Deduplicate OCR text within a chunk: if >80% overlap with previous frame's text, skip
  - Use fuzzy matching (not exact) to handle minor OCR variations

* Handle:
  * No text detected in any frame -> set has_ocr=False, continue
  * OCR library crash/error on specific frame -> skip that frame, log warning, continue
  * Extremely noisy OCR output (random characters) ->
    filter: if >50% of detected "words" are <2 chars or non-dictionary, discard
  * Non-English text -> EasyOCR supports multiple languages, detect and store language
  * Watermarks/logos -> these repeat in every frame; deduplicate handles this
  * Code on screen (common in tutorials) -> preserve as-is, valuable for search
  * Entirely black/blank frames -> skip OCR (detect: mean pixel value < 10)
  * Entirely white/washed-out frames -> skip OCR (detect: mean pixel value > 245)
  * Transition frames (fade/dissolve) -> skip OCR (detect: high variance + low edge count)
  * Frames with only a small overlay/banner -> still OCR, but flag as partial_text
  * Non-English OCR text -> translate to English using deep-translator (GoogleTranslator)
    BEFORE embedding. Reason: embedding model works best with English; mixing languages
    in the vector space reduces retrieval quality. Heuristic: if <85% ASCII chars,
    auto-translate to English. If translation fails, use original text as fallback.

---

## 6. VISUAL FEATURES (visual.py)

* Use CLIP ViT-B/32
* Process ALL keyframes within each chunk (no subsampling needed — keyframes are already sparse)
* Produce per-chunk visual embedding

### HOW AUDIO (TEXT) AND VISUAL VECTORS ALIGN

This is a critical architectural decision. The pipeline produces TWO independent
vector spaces per chunk:

```
CHUNK (30-120 seconds of video)
├── TEXT vector (384-dim, SentenceTransformer)
│   └── Source: what was SAID (transcript) + what was SHOWN as text (OCR)
│
└── VISUAL vector (512-dim, CLIP)
    └── Source: what was VISUALLY on screen (keyframes)
```

These vectors live in DIFFERENT embedding spaces and are NOT directly comparable.
They are aligned at the CHUNK level — both describe the same time window but from
different modalities. This is how they work together at query time:

**Example: User searches "how to configure nginx"**
1. Text search finds chunks where the instructor SAYS "nginx configuration"
2. Visual search finds chunks where an nginx config file is SHOWN on screen
3. A chunk where the instructor explains nginx while showing the config file
   scores high on BOTH → ranked highest
4. A chunk where they only briefly say "nginx" but show a different topic →
   text score high, visual score low → ranked lower

**Why this works better than a single fused vector:**
- Text-only queries (conceptual questions) lean on the text vector (weight 0.6)
- Visual queries ("show me the diagram") lean on the visual vector (weight 0.3)
- The chunk is the alignment unit — same time window guarantees both vectors
  describe the same moment in the video

### KEYFRAME-TO-CHUNK MAPPING

* Each keyframe has a timestamp from extraction (e.g., frame_00042.300.jpg = 42.3s)
* Each chunk has [start, end] timestamps from the chunking step
* A keyframe at time T belongs to the chunk where start <= T < end
* Since keyframes represent DISTINCT visual states, each one gets full weight

### PER-CHUNK VISUAL EMBEDDING STRATEGY

If a chunk has N keyframes:
* N = 1: use that single CLIP embedding directly
* N = 2-5: average the CLIP embeddings (each represents a different visual state,
  so the average captures the chunk's visual diversity)
* N > 5 (rare, fast-cut segments): cluster keyframes using simple k-means (k=3),
  take the centroid of each cluster, average the 3 centroids.
  This prevents one dominant visual theme from being diluted.
* N = 0: fallback frame was extracted at chunk midpoint (from video.py Pass 2),
  use that. If truly no frame exists, set has_visual=False.

All visual embeddings are L2-normalized before storage.

* Handle:
  * Failed image load (corrupted JPEG) -> skip frame, log, continue
  * Blank/black frames (mean pixel < 10) -> skip, they add noise to the embedding
  * Blank/white frames (mean pixel > 245) -> skip
  * No valid frames in a chunk -> set has_visual=False for that chunk
  * CLIP model download failure -> catch, retry once, raise clear error
  * GPU OOM during CLIP inference -> fallback to CPU, or reduce batch size
  * Very large frames (4K) -> resize to 224x224 for CLIP (CLIP does this internally,
    but pre-resizing saves memory)
  * Batch processing -> process all keyframes in batches of 32 to limit memory
  * Chunk with only blank/transition keyframes -> has_visual=False

---

## 7. TEXT EMBEDDINGS (embeddings.py)

* Use SentenceTransformer (all-MiniLM-L6-v2) — same model as chunking to avoid loading two
* Input = chunk transcript + " " + OCR text (concatenated)
* L2-normalize all embeddings

* Handle:
  * Empty text (no transcript AND no OCR) -> do NOT generate a text embedding
    set has_text=False, rely on visual embedding only
  * Very long text (>512 tokens, model limit) -> truncate to first 512 tokens
    (SentenceTransformer does this internally, but be aware)
  * Text is only OCR (no transcript) -> still embed, but flag source=ocr_only
  * Text is only transcript (no OCR) -> still embed, flag source=transcript_only
  * Batch embedding -> process all chunks at once for efficiency

IMPORTANT — Vector dimensions:
* all-MiniLM-L6-v2 produces 384-dim vectors
* CLIP ViT-B/32 produces 512-dim vectors
* The Qdrant collection MUST be configured with separate named vector sizes

---

## 8. STORAGE (storage.py)

* Use Qdrant client (qdrant-client library)
* Collection: "video_chunks"

* Collection setup:
  - Create collection if it doesn't exist
  - Named vectors config:
    * "text": size=384, distance=Cosine
    * "visual": size=512, distance=Cosine
  - If collection exists with wrong config -> log error, do NOT silently recreate

* Video ID strategy:
  - video_id = SHA256 hash of file content (first 10MB) + file size
  - This ensures same video = same ID regardless of filename
  - Before ingestion: check if video_id already exists in Qdrant
    * If yes: ask user whether to skip or re-ingest (delete old + insert new)

Each point:
```
{
  id: chunk_uuid (UUID4),
  vector: {
    "text": [...] or OMITTED if no text,
    "visual": [...] or OMITTED if no visual
  },
  payload: {
    video_id: str,
    video_filename: str,
    video_title: str (from filename, cleaned),
    video_duration: float,
    video_upload_date: str (file modification date, ISO format),
    chunk_index: int,
    total_chunks: int,
    timestamp_start: float,
    timestamp_end: float,
    transcript_text: str,
    ocr_text: str,
    detected_language: str,
    has_audio: bool,
    has_transcript: bool,
    has_visual: bool,
    has_ocr: bool,
    source: str (transcript_only | ocr_only | both | visual_only),
    neighbor_density: float (0.0-1.0, topic density score),
    keyframe_count: int (number of keyframes in this chunk),
    keyframe_timestamps: list[float] (timestamps of each keyframe),
    prev_context: str (last sentence of previous chunk),
    next_context: str (first sentence of next chunk),
    low_confidence: bool (true if transcript confidence < 0.5)
  }
}
```

IMPORTANT:
* Do NOT insert points with ZERO vectors (no text AND no visual) -> skip with warning
* Do NOT insert null/None vectors — omit the named vector entirely if missing
* Batch upsert in groups of 100 points
* If Qdrant upsert fails mid-batch -> retry that batch once, then log and continue

Edge cases:
* Qdrant connection drops mid-ingestion -> retry with backoff, save progress
* Collection exists with different vector config -> raise clear error, don't corrupt
* Extremely large payloads (long transcript) -> truncate transcript_text to 10,000 chars
* Special characters in text fields -> sanitize (Qdrant handles UTF-8, but strip null bytes)

---

## 9. QUERY PIPELINE (query.py)

* Input: user query string + optional filters

### Query Processing:
1. Clean query: strip whitespace, lowercase, remove special chars
2. Generate text embedding using SentenceTransformer
3. Generate CLIP text embedding using CLIP's text encoder (for visual search)

### Search Strategy:

* Text search: query Qdrant "text" vector, get top 20 candidates
* Visual search: query Qdrant "visual" vector using CLIP text embedding, get top 20

### Re-Ranking (Multi-Signal Scoring):

For each candidate chunk, compute:

```
text_score    = cosine_similarity(query_emb, chunk.text_vector)    # 0-1
visual_score  = cosine_similarity(query_clip_emb, chunk.visual_vector) # 0-1
density_score = chunk.payload.neighbor_density                      # 0-1

# Time decay (from edgecases.md)
age_days = (today - chunk.payload.video_upload_date).days
if age_days > 730:        # older than 2 years
    time_factor = 0.8
elif age_days < 180:      # newer than 6 months
    time_factor = 1.1
else:
    time_factor = 1.0

# Title/tag boost (from edgecases.md)
title_boost = 1.5 if query tokens overlap with video_title words else 1.0

# Final score
base_score = (0.6 * text_score) + (0.3 * visual_score) + (0.1 * density_score)
final_score = base_score * time_factor * title_boost
```

### Return top K results (default K=5):
```
{
  video_id, video_filename, video_title,
  timestamp_start, timestamp_end,
  transcript_text (snippet, ~200 chars around best match),
  score, score_breakdown: {text, visual, density, time_factor, title_boost},
  prev_context, next_context
}
```

### Optional Filters:
* Filter by video_id (search within a specific video)
* Filter by has_transcript=True (only chunks with speech)
* Filter by detected_language
* Filter by date range

Edge cases:
* Empty query string -> return error message, don't search
* Query is very long (>200 words) -> truncate to first 50 words
* No results at all -> return fallback: "No matching content found. Try broader terms."
* All results have very low scores (<0.3) -> return results but with a warning
* Visual-only chunks (no text vector) -> only scored on visual + density
* Text-only chunks (no visual vector) -> only scored on text + density, re-weight accordingly
* CLIP text encoder fails -> fallback to text-only search
* Qdrant is unreachable during query -> raise clear connection error
* Special characters in query -> sanitize before embedding
* Query in different language than video -> SentenceTransformer is multilingual, still works
  but log that cross-language matching may be less accurate

---

## 10. CLEANUP (cleanup.py)

After pipeline completes:
* Delete extracted frames from data/frames/
* Delete extracted audio from data/audio/
* Only delete files for the current video (use video_id prefix in filenames)
* If cleanup fails (permission error) -> log warning, don't crash

Option to keep files:
* config.py: KEEP_TEMP_FILES = False (default)
* If True, skip cleanup

---

## 11. MAIN PIPELINE (main.py)

main.py should:

* Accept video path as CLI argument (argparse)
* Optional flags:
  - --keep-temp: keep extracted frames/audio
  - --skip-ocr: skip OCR step (faster)
  - --skip-visual: skip CLIP step (faster)
  - --query "search text": run a query after ingestion
  - --reindex: force re-index even if video exists
* Run validation checks first
* Run full pipeline with progress logging
* Print summary:
  - Number of chunks created
  - Modalities available (audio/ocr/visual)
  - Time taken per step
* If --query provided, run sample query and print results
* Clean up temp files at the end

Pipeline must be wrapped in try/finally:
* If any step crashes, cleanup still runs
* Partial results (e.g., some chunks stored before crash) are logged

---

## 12. EDGE CASE HANDLING (COMPREHENSIVE)

### Input Validation
* Non-existent file path -> clear error with the path shown
* Empty file (0 bytes) -> clear error
* Non-video file (.txt, .jpg, audio-only .mp3) -> reject with message
* File path with spaces/unicode -> handle correctly
* Extremely large file (>10GB) -> warn but allow processing
* Symlink to deleted file -> resolve and validate

### Video Processing
* Video without audio stream -> skip transcript, continue with OCR + visual
* Video without video stream (audio file) -> reject
* Corrupted video (partial download) -> process what's available, warn
* DRM-protected video -> clear error
* Very short video (<5 sec) -> single chunk
* Very long video (>2 hours) -> segment-based processing
* 4K+ resolution -> downscale frames for processing
* Variable frame rate -> force constant output
* Video with subtitles baked in -> OCR will capture them (feature, not bug)

### Transcription
* Silent audio -> empty transcript, has_transcript=False
* Music-only audio -> detect low confidence, has_transcript=False
* Whisper hallucinations (repeated text) -> deduplicate consecutive identical segments
* Very noisy audio -> low confidence flag
* Non-English audio -> multilingual support, store language

### Chunking
* No transcript AND no OCR -> chunk by fixed 60-second intervals using frames
* Single segment -> single chunk
* Information splintering -> semantic boundary detection prevents splitting related topics
* Passing mention vs deep explanation -> topic density scoring handles this

### OCR
* No text on screen anywhere -> has_ocr=False
* Watermarks/logos repeating -> deduplication
* Black/white/transition frames -> skip
* Code on screen -> preserve (valuable)
* Noisy/garbled OCR -> filter by word quality

### Visual & Audio-Visual Alignment
* All keyframes blank/black -> has_visual=False
* CLIP OOM -> reduce batch size or fallback to CPU
* No keyframes extracted -> fallback frame at chunk midpoint; if still fails, has_visual=False
* Scene detection extracts too many frames (music video, fast cuts) -> cap per minute
* Scene detection extracts zero frames (static screencast) -> fallback interval sampling
* Chunk with many keyframes (fast visual changes) but short transcript ->
  visual vector dominates the chunk's identity; this is correct behavior
* Chunk with long transcript but zero keyframes (talking head, no visual change) ->
  text vector dominates; visual uses fallback frame; this is correct behavior
* Audio says one thing, screen shows another (instructor talks about X while code for Y
  is on screen) -> both are captured independently; a query for X matches text,
  a query for Y matches visual; the chunk is findable by either modality

### Storage
* Qdrant not running -> clear error before pipeline starts
* Duplicate video -> detect by video_id hash, ask before re-index
* No vectors for a chunk (no text, no visual) -> skip chunk with warning
* Connection drop mid-ingestion -> retry with backoff

### Query
* Empty query -> error message
* No results -> fallback message
* Low quality results -> warning
* Stale results from old videos -> time decay factor
* Passing mentions ranking high -> density scoring penalizes them
* Title relevance -> title boost for matching videos
* Cross-language query -> works but with accuracy warning
* Visual-only query intent (e.g., "show me a red car") -> CLIP text encoding handles this

---

## 13. PERFORMANCE (KEEP SIMPLE)

* Batch frame processing: load and process frames in groups of 32
* Batch embedding computation: embed all chunks at once
* Batch Qdrant upsert: groups of 100
* Use numpy for all vector operations (averaging, normalization)
* Lazy model loading: only load Whisper if has_audio, only load CLIP if not --skip-visual
* Log time taken per pipeline step for profiling
* Memory management: delete large arrays (frames, embeddings) after use with explicit del

---

## 14. CONFIG (config.py)

Centralize all constants:

```python
# Paths
DATA_DIR = "data"
FRAMES_DIR = "data/frames"
AUDIO_DIR = "data/audio"

# Video processing — Keyframe extraction
SCENE_CHANGE_THRESHOLD = 0.3    # ffmpeg scene detection sensitivity (0-1, lower = more frames)
MAX_KEYFRAMES_PER_MINUTE = 6    # cap for fast-cut videos
FALLBACK_FRAME_INTERVAL = 10    # seconds, if scene detection yields 0 frames
AUDIO_SAMPLE_RATE = 16000       # Hz
MAX_FRAME_RESOLUTION = 720      # resize if larger

# Whisper
WHISPER_MODEL = "base"
WHISPER_CONFIDENCE_THRESHOLD = 0.3
WHISPER_SEGMENT_MAX_SECONDS = 1800  # 30 min windows for long audio

# Chunking
MIN_CHUNK_DURATION = 30         # seconds
MAX_CHUNK_DURATION = 120        # seconds
SIMILARITY_STD_FACTOR = 0.5
BOUNDARY_PHRASES = ["now let's", "moving on", "next topic", "let's switch",
                     "on the other hand", "in contrast", "compared to",
                     "let's talk about", "another thing", "the next step"]
PAUSE_THRESHOLD = 2.0           # seconds, hard boundary

# OCR
OCR_CONFIDENCE_THRESHOLD = 0.5
OCR_SAMPLE_INTERVAL = 5         # seconds between OCR frames
OCR_DEDUP_THRESHOLD = 0.8       # text overlap ratio for dedup
BLANK_FRAME_THRESHOLD_LOW = 10  # mean pixel value
BLANK_FRAME_THRESHOLD_HIGH = 245

# CLIP
CLIP_MODEL = "ViT-B/32"
CLIP_BATCH_SIZE = 32
CLIP_CLUSTER_THRESHOLD = 5      # if chunk has > this many keyframes, cluster to 3

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM = 384
VISUAL_EMBEDDING_DIM = 512
MAX_TEXT_LENGTH = 10000          # chars, for payload truncation

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "video_chunks"
UPSERT_BATCH_SIZE = 100

# Query
DEFAULT_TOP_K = 5
TEXT_WEIGHT = 0.6
VISUAL_WEIGHT = 0.3
DENSITY_WEIGHT = 0.1
TIME_DECAY_OLD_DAYS = 730       # 2 years
TIME_DECAY_NEW_DAYS = 180       # 6 months
TIME_DECAY_OLD_FACTOR = 0.8
TIME_DECAY_NEW_FACTOR = 1.1
TITLE_BOOST_FACTOR = 1.5
LOW_SCORE_THRESHOLD = 0.3

# Cleanup
KEEP_TEMP_FILES = False
```

---

## 15. LOGGING & OUTPUT

* Use Python logging module (not print)
* Log levels:
  - INFO: pipeline progress (step started/completed, counts)
  - WARNING: non-fatal issues (skipped frames, low confidence, fallbacks)
  - ERROR: failures that skip a step
* Progress format:
```
[INFO] Validating input: video.mp4
[INFO] Extracting audio... done (12.3s)
[INFO] Extracting frames... done (45.1s, 3600 frames)
[INFO] Transcribing audio... done (34.2s, 142 segments)
[WARNING] Low transcript confidence (0.28), flagging as low_confidence
[INFO] Creating semantic chunks... done (1.2s, 24 chunks)
[INFO] Running OCR... done (89.4s, 18/24 chunks have text)
[INFO] Extracting visual features... done (23.1s, 24/24 chunks)
[INFO] Generating embeddings... done (0.8s)
[INFO] Storing in Qdrant... done (1.1s, 24 points)
[INFO] Cleanup... done
[INFO] Pipeline complete: 24 chunks indexed for video.mp4
```

---

## 16. REQUIREMENTS (requirements.txt)

```
openai-whisper
sentence-transformers
torch
clip-model  # or use open_clip_torch
easyocr
qdrant-client
numpy
Pillow
tqdm
```

Note: ffmpeg must be installed separately (system package, not pip)

---

## 17. OPTIONAL ENHANCEMENTS (IF EASY)

* Similarity visualization function (matplotlib) — scatter plot of chunk embeddings
* Batch video ingestion — accept a directory of videos
* Export search results to JSON

---

FINAL REQUIREMENT:
Write COMPLETE WORKING CODE for all files.

Do NOT skip implementations.
Do NOT leave TODOs.
Everything must run end-to-end.
Every edge case listed above must be handled in code.

---

Now generate the full codebase.
