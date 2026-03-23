# Multimodal Video Retrieval Pipeline — Complete Flow Documentation

This document traces exactly how data flows through every phase of the pipeline,
what transformations happen at each step, what edge cases are handled, and how the
query system works end-to-end.

---

## HIGH-LEVEL PIPELINE OVERVIEW

```
INPUT                           OUTPUT
  |                               |
  v                               v
video.mp4  ──────────────────>  Qdrant DB with searchable chunks
                                  |
                                  v
              user query  ──>  ranked results with timestamps
```

### The 10-step pipeline:

```
video.mp4
    |
    ├──[0]── Validation (ffmpeg? Qdrant? GPU? valid file?)
    |
    ├──[1]── Extract Audio ──────────> audio.wav (16kHz mono)
    |
    ├──[2]── Extract Keyframes ──────> [frame_00042.300.jpg, frame_00098.100.jpg, ...]
    |
    ├──[3]── Whisper Transcription ──> [{start:0.0, end:4.2, text:"Today we'll..."}, ...]
    |
    ├──[4]── Semantic Chunking ──────> [{start:0, end:65, text:"...", neighbor_density:0.82}, ...]
    |
    ├──[5]── Map Keyframes to Chunks ─> each chunk gets its keyframe list
    |
    ├──[6]── OCR per Chunk ──────────> chunk.ocr_text = "def authenticate(user):"
    |
    ├──[7]── CLIP per Chunk ─────────> chunk visual embedding = [0.12, -0.34, ...] (512-dim)
    |
    ├──[8]── Text Embeddings ────────> chunk text embedding = [0.05, 0.78, ...] (384-dim)
    |
    ├──[9]── Store in Qdrant ────────> points with named vectors + rich payload
    |
    └──[cleanup]── Delete temp frames/audio
```

---

## PHASE 0: VALIDATION (validation.py)

### What happens:
Before any processing, the system runs 5 checks in order:

```
check_ffmpeg()      →  runs "ffmpeg -version" and "ffprobe -version"
check_qdrant()      →  tries QdrantClient.get_collections() with 5s timeout
check_gpu()         →  torch.cuda.is_available() → returns "cuda" or "cpu"
validate_video_file →  ffprobe JSON probe of the file
check_disk_space()  →  shutil.disk_usage vs estimated need
```

### Data flow:
```
Input:  "C:/videos/lecture.mp4"
        │
        ├── os.path.realpath() ── resolves symlinks
        ├── os.path.exists() ── file exists?
        ├── os.path.getsize() ── not 0 bytes?
        ├── extension check ── .mp4 in {.mp4, .avi, .mkv, .mov, .webm, .flv, .wmv}?
        │
        └── ffprobe -print_format json -show_format -show_streams lecture.mp4
                │
                ├── parse JSON response
                ├── find video streams (codec_type == "video")
                ├── find audio streams (codec_type == "audio")
                │
                └── Output: video_meta = {
                        duration: 3600.0,    # seconds
                        width: 1920,
                        height: 1080,
                        has_audio: True,
                        codec: "h264",
                        fps: 30.0
                    }
```

### Edge cases handled:
| Scenario | What happens |
|----------|-------------|
| ffmpeg not installed | RuntimeError with install URL |
| Qdrant not running | RuntimeError with docker run command |
| No CUDA GPU | Falls back to CPU with warning |
| File doesn't exist | RuntimeError with path shown |
| 0-byte file | RuntimeError |
| .txt / .jpg / .mp3 file | RuntimeError "unsupported extension" |
| Audio-only file (.mp4 with no video stream) | RuntimeError "no video stream found" |
| Symlink to deleted file | os.path.realpath resolves first, then existence check fails |
| 15GB video file | Warning logged, processing continues |
| Duration=0 in metadata | Tries stream-level duration as fallback |
| ffprobe times out (30s) | RuntimeError |
| Unicode characters in path | os.path.realpath normalizes |

### Output:
```python
video_meta = {"duration": 3600.0, "width": 1920, "height": 1080, "has_audio": True, "codec": "h264", "fps": 30.0}
device = "cuda"  # or "cpu"
```

---

## PHASE 1: AUDIO EXTRACTION (video.py → extract_audio)

### What happens:
ffmpeg strips the audio track into a mono 16kHz WAV file suitable for Whisper.

```
ffmpeg -y -i lecture.mp4 -vn -ac 1 -ar 16000 -acodec pcm_s16le data/audio/{video_id}.wav
```

### Data flow:
```
Input:  lecture.mp4 (1920x1080, h264, 1 hour, has audio)
        │
        ├── video_id = SHA256(first 10MB + filesize)[:16]  →  "a3f8b2c1d4e5f678"
        │
        └── ffmpeg extracts audio
                │
                └── Output: "data/audio/a3f8b2c1d4e5f678.wav"
                            (mono, 16kHz, PCM 16-bit, ~115 MB for 1 hour)
```

### Edge cases handled:
| Scenario | What happens |
|----------|-------------|
| No audio stream | video_meta.has_audio=False, skipped entirely, returns None |
| ffmpeg timeout | timeout = max(120, duration*2+60) seconds. Returns None |
| ffmpeg crash | Logs stderr[:200], returns None |
| Output file < 1KB | Treated as failed extraction, returns None |
| Multiple audio tracks | ffmpeg uses first track by default |

### Output:
```python
audio_path = "data/audio/a3f8b2c1d4e5f678.wav"  # or None
```

---

## PHASE 2: KEYFRAME EXTRACTION (video.py → extract_keyframes)

### Why keyframes, not 1 FPS:
```
1 FPS on 1-hour video = 3,600 frames
    └── 95% are identical (same slide for 30 sec = 30 identical frames)
    └── Wastes: disk (~180MB), CLIP compute, embedding quality (noise from averaging duplicates)

Keyframe extraction = ~30-80 frames for 1 hour
    └── Each frame is visually DISTINCT (new slide, scene change, camera cut)
    └── Saves: disk (~3MB), compute (50-100x less), embedding quality (each contributes uniquely)
```

### How it works (two-pass):

**Pass 1 — Scene-change detection:**
```
ffmpeg -i lecture.mp4 -vf "select='gt(scene,0.3)',showinfo,scale=-1:720" -vsync vfn kf_%05d.jpg
```

The `scene` filter computes a score (0-1) for how different each frame is from the previous.
When score > 0.3 (configurable), ffmpeg outputs that frame. The `showinfo` filter writes
`pts_time:42.300` to stderr, which we parse to get exact timestamps.

```
ffmpeg stderr output:
    [Parsed_showinfo_1 @ ...] n:   0 pts:    42300 pts_time:42.300  ...
    [Parsed_showinfo_1 @ ...] n:   1 pts:    98100 pts_time:98.100  ...
    [Parsed_showinfo_1 @ ...] n:   2 pts:   156700 pts_time:156.700 ...
        │
        ├── Parse pts_time with regex: r"pts_time:\s*([\d.]+)"
        ├── Result: timestamps = [42.3, 98.1, 156.7, ...]
        │
        └── Rename files: kf_00001.jpg → frame_00042.300.jpg
                          kf_00002.jpg → frame_00098.100.jpg
```

**Cap check — Too many keyframes:**
```
max_allowed = (duration_minutes) * MAX_KEYFRAMES_PER_MINUTE
            = (60) * 10 = 600

If extracted > 600 (e.g., music video with fast cuts):
    step = len(keyframes) / 600
    Keep every step-th frame, delete the rest
```

**Pass 2 — Fallback guarantee (happens later in main.py, Phase 5):**
```
After chunking, if a chunk has ZERO keyframes in its time range:
    Extract 1 frame at the chunk's midpoint:
    ffmpeg -ss {midpoint} -i video.mp4 -frames:v 1 frame_{midpoint}.jpg
```

### Data flow:
```
Input:  lecture.mp4, video_id="a3f8b2c1d4e5f678", duration=3600, height=1080
        │
        ├── Create dir: data/frames/a3f8b2c1d4e5f678/
        │
        ├── Run scene detection ffmpeg command
        │      │
        │      ├── Parse timestamps from stderr
        │      └── Rename files with timestamps
        │
        ├── Cap check: 45 frames < 600 max → no subsampling needed
        │
        └── Output: [
                {"path": "data/frames/.../frame_00042.300.jpg", "timestamp": 42.3},
                {"path": "data/frames/.../frame_00098.100.jpg", "timestamp": 98.1},
                {"path": "data/frames/.../frame_00156.700.jpg", "timestamp": 156.7},
                ... (45 keyframes total for a 1-hour lecture)
            ]
```

### Edge cases handled:
| Scenario | What happens |
|----------|-------------|
| Scene detection times out | Falls back to interval extraction (1 frame/10s) |
| Scene detection returns 0 frames (static screencast) | Fallback: 1 frame every 10 seconds |
| Too many frames (fast cuts, music video) | Cap at MAX_KEYFRAMES_PER_MINUTE=10, subsample uniformly |
| 4K+ resolution | Scale filter: `-vf scale=-1:720` added |
| ffmpeg crash | Falls back to interval extraction |
| Very short video (<5s) | Extracts whatever scene detection finds, minimum 1 frame |

### Output:
```python
keyframes = [
    {"path": "data/frames/.../frame_00042.300.jpg", "timestamp": 42.3},
    {"path": "data/frames/.../frame_00098.100.jpg", "timestamp": 98.1},
    ...
]
# Typically 30-80 keyframes for a 1-hour video
```

---

## PHASE 3: WHISPER TRANSCRIPTION (audio.py → transcribe)

### What happens:
Whisper base model transcribes the audio into timestamped text segments.

### Data flow:
```
Input:  "data/audio/a3f8b2c1d4e5f678.wav" (mono 16kHz, 1 hour)
        │
        ├── Load Whisper model (lazy, first call only)
        │      └── whisper.load_model("base", device="cuda")
        │          (if CUDA OOM → retry on CPU)
        │
        ├── Duration check: 3600s > 1800s (WHISPER_SEGMENT_MAX_SECONDS)
        │      └── YES → use windowed processing
        │
        ├── Windowed processing:
        │      window = 30 minutes (480,000 samples at 16kHz)
        │      overlap = 10 seconds (160,000 samples)
        │      step = window - overlap
        │
        │      Window 1: audio[0 : 28,800,000]          → segments with time offset 0
        │      Window 2: audio[28,640,000 : 57,440,000]  → segments with time offset 1790s
        │      Window 3: audio[57,280,000 : 86,080,000]  → segments with time offset 3580s
        │      ...
        │
        │      Each window → model.transcribe(chunk) → segments:
        │      [
        │        {"start": 0.0,  "end": 4.2,  "text": "Today we will learn about", "avg_logprob": -0.21},
        │        {"start": 4.2,  "end": 8.7,  "text": "JWT authentication and how", "avg_logprob": -0.18},
        │        {"start": 8.7,  "end": 12.1, "text": "to implement it in Node.js", "avg_logprob": -0.15},
        │        ...
        │      ]
        │      + time_offset added to each segment's start/end
        │
        ├── Overlap deduplication:
        │      Segments from overlap regions (where windows overlap by 10s):
        │      If two segments have start times within 10s of each other → keep higher confidence
        │
        ├── Hallucination detection:
        │      If same text appears in 3+ consecutive segments:
        │      "Thank you for watching" × 5 → keep first 2, remove rest
        │
        ├── Confidence check:
        │      avg_logprob across all segments
        │      If avg < -0.3 → low_confidence=True, has_transcript=False
        │      (indicates music/noise, not speech)
        │
        └── Output: {
                "segments": [
                    {"start": 0.0,  "end": 4.2,  "text": "Today we will learn about",    "confidence": -0.21},
                    {"start": 4.2,  "end": 8.7,  "text": "JWT authentication and how",   "confidence": -0.18},
                    {"start": 8.7,  "end": 12.1, "text": "to implement it in Node.js",   "confidence": -0.15},
                    ... (142 segments for a 1-hour lecture)
                ],
                "language": "en",
                "has_transcript": True,
                "low_confidence": False
            }
```

### What a "segment" is:
Whisper naturally breaks audio into breath-group segments — typically 3-10 seconds each,
corresponding to one sentence or clause. These are NOT chunks yet. A 1-hour video typically
produces 100-300 segments.

```
SEGMENT ≠ CHUNK

Segment: "JWT stands for JSON Web Token" [8.7s - 12.1s] (3.4 seconds, one sentence)
Chunk:   Multiple segments merged [0.0s - 65.0s] (65 seconds, one topic)
```

### Edge cases handled:
| Scenario | What happens |
|----------|-------------|
| audio_path is None | Returns empty result immediately |
| Silent audio | Whisper returns 0 segments → has_transcript=False |
| Music/noise only | avg_logprob < -0.3 → treated as "no speech" |
| Audio > 30 min | Windowed processing with 10s overlap |
| Whisper OOM | Catches RuntimeError, falls back to windowed mode |
| Model download fails | Retry once, then raise |
| CUDA load fails | Falls back to CPU |
| Hallucinated repetitions | "Thank you" × 5 → deduplicated to 2 |
| Overlapping windows | Dedup by start time, keep higher confidence |

### Output:
```python
transcript_result = {
    "segments": [{"start": 0.0, "end": 4.2, "text": "...", "confidence": -0.21}, ...],
    "language": "en",
    "has_transcript": True,
    "low_confidence": False,
}
# 142 segments for a typical 1-hour lecture
```

---

## PHASE 4: SEMANTIC CHUNKING (chunking.py → create_chunks)

This is the most complex phase. It transforms 142 small segments into ~20-30 meaningful chunks.

### Three-stage process:

### Stage A: Detect hard boundaries

```
Input: 142 segments
        │
        ├── Embed ALL segment texts using SentenceTransformer (all-MiniLM-L6-v2)
        │   142 texts → model.encode() → 142 vectors of 384-dim each
        │   L2-normalize all vectors
        │
        ├── For each consecutive pair (i, i+1), check 3 signals:
        │
        │   Signal 1: TRANSITIONAL PHRASE in segment[i+1]
        │   "now let's look at React" → contains "now let's" → HARD BOUNDARY at i
        │
        │   Signal 2: LONG PAUSE between segment[i].end and segment[i+1].start
        │   segment[41].end = 120.5, segment[42].start = 123.8 → gap = 3.3s > 2.0s → HARD BOUNDARY
        │
        │   Signal 3: DRAMATIC TOPIC SHIFT
        │   cosine_sim(embedding[i], embedding[i+1]) = 0.18 < 0.3 → HARD BOUNDARY
        │
        └── Result: hard_boundaries = {12, 41, 67, 89, 105, 128}  (6 boundaries)
```

**Why hard boundaries matter (the Information Splintering Problem):**
```
WITHOUT hard boundaries:
    Segment 40: "React uses a virtual DOM for efficient rendering"
    Segment 41: "Angular on the other hand uses real DOM manipulation"
    ─── These get merged into ONE chunk ───
    Segment 42: "Let's compare their performance characteristics"
    ─── This gets split into a DIFFERENT chunk ───

    Result: The comparison context is split! Neither chunk answers "React vs Angular?"

WITH hard boundaries:
    Segment 40: "React uses a virtual DOM..."
    Segment 41: "Angular on the other hand..." ← "on the other hand" = BOUNDARY before this
    ─── HARD SPLIT ───
    Segment 41-48: All comparison content stays together in one chunk
```

### Stage B: Similarity-based merging (respecting hard boundaries)

```
Input: 142 segments + 6 hard boundary positions
        │
        ├── Compute consecutive cosine similarities:
        │   sim[0]  = cos(emb[0], emb[1])  = 0.87
        │   sim[1]  = cos(emb[1], emb[2])  = 0.91
        │   sim[2]  = cos(emb[2], emb[3])  = 0.45  ← topic starting to shift
        │   ...
        │   sim[40] = cos(emb[40], emb[41]) = 0.12  ← dramatic shift (also hard boundary)
        │   ...
        │
        ├── Dynamic threshold:
        │   mean(all sims) = 0.72
        │   std(all sims) = 0.18
        │   threshold = 0.72 - 0.5 * 0.18 = 0.63
        │
        ├── Merge loop:
        │   Start with chunk = {segment[0]}
        │
        │   For each segment[i]:
        │     current_duration = segment[i].end - chunk.start
        │
        │     IF (i-1) is in hard_boundaries → SPLIT (always)
        │     ELSE IF duration >= 30s AND (sim[i-1] < 0.63 OR duration >= 120s) → SPLIT
        │     ELSE → merge into current chunk
        │
        └── Result: 24 chunks
```

### How segments become chunks — concrete example:

```
SEGMENTS (142 total):                                  CHUNKS (24 total):
┌─────────────────────────────────────┐
│ seg[0]  0.0-4.2s   "Today we..."   │
│ seg[1]  4.2-8.7s   "JWT stands..." │
│ seg[2]  8.7-12.1s  "It consists.." │──── Chunk 0 [0.0s - 62.3s]
│ seg[3]  12.1-16.5s "The header..." │     "Today we... JWT stands... It consists..."
│ ...                                 │     (15 segments merged, similar topic)
│ seg[14] 58.0-62.3s "...signature"  │
├─────────────────────────────────────┤
│ seg[15] 62.3-65.1s "Now let's ..." │◄── HARD BOUNDARY (transitional phrase)
│ seg[16] 65.1-69.8s "Open your..."  │
│ ...                                 │──── Chunk 1 [62.3s - 124.0s]
│ seg[28] 120.0-124.0s "...running"  │     "Now let's... Open your..."
├─────────────────────────────────────┤
│ seg[29] 127.5-131.0s "Moving on.." │◄── HARD BOUNDARY (3.5s pause + "moving on")
│ ...                                 │──── Chunk 2 [127.5s - 185.0s]
└─────────────────────────────────────┘
```

### Stage C: Post-processing

After chunks are formed, three enrichments happen:

**1. Overlap context:**
```
Chunk 2: {
    text: "Moving on to middleware, Express uses...",
    prev_context: "and that completes the JWT implementation",  ← last sentence of Chunk 1
    next_context: "Now let's deploy this to production",        ← first sentence of Chunk 3
}
```

**2. Neighbor density (topic density scoring):**
```
For Chunk 5 (about "database indexing"):
    Look at window of 2 chunks on each side:
    Chunk 3: "setting up PostgreSQL"    → sim with Chunk 5 = 0.71  (related)
    Chunk 4: "writing SQL queries"      → sim with Chunk 5 = 0.82  (related)
    Chunk 6: "query optimization"       → sim with Chunk 5 = 0.78  (related)
    Chunk 7: "deploying to AWS"         → sim with Chunk 5 = 0.23  (unrelated)

    neighbor_density = mean([0.71, 0.82, 0.78, 0.23]) = 0.635

For Chunk 12 (briefly mentions "caching"):
    Chunk 10: "user authentication"     → sim = 0.15
    Chunk 11: "session management"      → sim = 0.22
    Chunk 13: "error handling"          → sim = 0.18
    Chunk 14: "logging setup"           → sim = 0.12

    neighbor_density = mean([0.15, 0.22, 0.18, 0.12]) = 0.1675

    Chunk 5 (density 0.635) will rank HIGHER than Chunk 12 (density 0.168)
    for a query about their shared topic — because Chunk 5 is part of a
    sustained discussion while Chunk 12 is just a passing mention.
```

**3. Content richness boost (for short but complete topics):**
```
What if an instructor explains "caching" thoroughly in just 45 seconds (1 chunk)?
    Chunk 12: text = "Redis caching works by storing key-value pairs in memory.
               You set a TTL for each key. When a request comes in, check Redis
               first. If it's a cache hit, return immediately. If miss, query the
               database and store the result..." (350 characters)

    Raw density = 0.17 (neighbors are about different topics)
    BUT text length = 350 > 200 → richness_boost = min(0.3, 350/2000) = 0.175
    Adjusted density = min(1.0, 0.17 + 0.175) = 0.345

    This prevents punishing a concise-but-complete explanation just because
    the instructor only spent one chunk on it.
```

### Edge cases handled:
| Scenario | What happens |
|----------|-------------|
| No transcript segments at all | Falls back to fixed 60-second chunks |
| Single segment | Single chunk, density=1.0 |
| Very short video (<30s) | Single chunk, skip similarity computation |
| All segments identical (loop video) | Single chunk |
| >1000 tiny segments | Batch embedding computation (64 at a time) |
| Overlapping timestamps (Whisper bug) | Sort by start time, resolve overlaps |
| Gap in timestamps (60s → 180s) | Treated as hard boundary |
| Short but complete topic (45s) | Content richness boost prevents unfair penalty |

### Output:
```python
chunks = [
    {
        "start": 0.0,
        "end": 62.3,
        "text": "Today we will learn about JWT authentication...",
        "segment_indices": [0, 1, 2, ..., 14],
        "prev_context": "",
        "next_context": "Now let's set up our Node.js project",
        "neighbor_density": 0.73,
    },
    {
        "start": 62.3,
        "end": 124.0,
        "text": "Now let's set up our Node.js project...",
        "segment_indices": [15, 16, ..., 28],
        "prev_context": "and that covers the JWT token structure",
        "next_context": "Moving on to middleware configuration",
        "neighbor_density": 0.81,
    },
    ... (24 chunks total)
]
```

---

## PHASE 5: MAP KEYFRAMES TO CHUNKS (main.py)

### What happens:
Each keyframe gets assigned to the chunk whose time range contains it.

```
Keyframes:                          Chunks:
frame_00042.300.jpg (42.3s)    →    Chunk 0 [0.0s - 62.3s]      ✓ (42.3 is in range)
frame_00098.100.jpg (98.1s)    →    Chunk 1 [62.3s - 124.0s]    ✓ (98.1 is in range)
frame_00156.700.jpg (156.7s)   →    Chunk 2 [127.5s - 185.0s]   ✓ (156.7 is in range)
```

### Fallback for empty chunks:
```
Chunk 8 [480.0s - 540.0s] — NO keyframes in this range (talking head, no visual change)
    │
    ├── midpoint = (480 + 540) / 2 = 510.0
    ├── ffmpeg -ss 510 -i video.mp4 -frames:v 1 frame_00510.000.jpg
    │
    └── Chunk 8 now has 1 fallback keyframe
```

### Output:
```python
# Each chunk now has a "keyframes" field:
chunk = {
    "start": 0.0, "end": 62.3,
    "text": "Today we will learn about JWT...",
    "keyframes": [
        {"path": "data/frames/.../frame_00042.300.jpg", "timestamp": 42.3},
        {"path": "data/frames/.../frame_00055.800.jpg", "timestamp": 55.8},
    ],
    ...
}
```

---

## PHASE 6: OCR PER CHUNK (ocr.py → extract_ocr_for_chunk)

### What happens:
For each chunk, run EasyOCR on its keyframes to extract on-screen text.

### Data flow for one chunk:
```
Chunk 1 has 3 keyframes:
    frame_00070.200.jpg ── slide with title "JWT Authentication"
    frame_00085.500.jpg ── code: "const jwt = require('jsonwebtoken')"
    frame_00098.100.jpg ── same code, slightly scrolled

For each keyframe:
    ├── Open image with PIL → convert to RGB numpy array
    │
    ├── Blank check: mean pixel value
    │   mean(frame) = 142.3 → NOT blank (10 < 142.3 < 245) → proceed
    │
    ├── EasyOCR: reader.readtext(image)
    │   Returns: [
    │       ([bbox], "JWT Authentication", 0.92),    ← confidence 0.92 > 0.5 ✓
    │       ([bbox], "const jwt", 0.87),              ← confidence 0.87 > 0.5 ✓
    │       ([bbox], "xk2!", 0.31),                   ← confidence 0.31 < 0.5 ✗ filtered
    │   ]
    │
    ├── Noise check: "JWT Authentication const jwt" → words are real → NOT noisy
    │
    ├── Dedup check: overlap("JWT Authentication const jwt", previous_frame_text)
    │   Frame 1: "JWT Authentication" → NEW, keep
    │   Frame 2: "const jwt = require('jsonwebtoken')" → NEW, keep
    │   Frame 3: "const jwt = require('jsonwebtoken') const secret" → 80%+ overlap → SKIP
    │
    └── Combined text: "JWT Authentication const jwt = require('jsonwebtoken')"

Translation check (for non-English):
    ASCII ratio = 52/55 = 0.94 > 0.85 → likely English → skip translation

    If it were Korean: "JWT 인증 구현하기"
    ASCII ratio = 4/12 = 0.33 < 0.85 → translate
    GoogleTranslator("auto", "en").translate() → "Implementing JWT Authentication"
```

### Edge cases handled:
| Scenario | What happens |
|----------|-------------|
| No keyframes in chunk | Returns {"text": "", "has_ocr": False} |
| Blank/black frame (mean < 10) | Skipped |
| White/washed frame (mean > 245) | Skipped |
| OCR crash on a frame | Skip that frame, log warning, continue to next |
| Garbled output ("xk2! @# b") | >50% words < 2 chars → discarded |
| Duplicate text (same slide) | Word-overlap > 80% → skipped |
| Non-English text | Translated to English via Google Translate API |
| Translation API fails | Uses original text as fallback |
| Watermarks/logos | Deduplicated (same text every frame) |
| Code on screen | Preserved as-is (valuable for search) |

### Output (per chunk):
```python
chunk["ocr_text"] = "JWT Authentication const jwt = require('jsonwebtoken')"
chunk["has_ocr"] = True
```

---

## PHASE 7: VISUAL FEATURES PER CHUNK (visual.py → extract_visual_embedding)

### What happens:
CLIP ViT-B/32 encodes each chunk's keyframes into a single 512-dim visual embedding.

### Data flow for one chunk:
```
Chunk 1 has 3 keyframes:
    │
    ├── For each keyframe:
    │   ├── Open with PIL → RGB
    │   ├── Convert to numpy → blank check (mean pixel)
    │   │   frame_00070.200.jpg: mean=142.3 → valid
    │   │   frame_00085.500.jpg: mean=138.7 → valid
    │   │   frame_00098.100.jpg: mean=2.1   → BLANK, skip
    │   └── 2 valid images remain
    │
    ├── CLIP preprocessing: resize to 224×224, normalize
    │
    ├── Batch through CLIP image encoder:
    │   torch.stack([preprocess(img1), preprocess(img2)]) → tensor [2, 3, 224, 224]
    │   model.encode_image(tensor) → embeddings [2, 512]
    │   L2 normalize each row
    │
    ├── Aggregation strategy (N=2, which is ≤ 5):
    │   final_embedding = mean(embeddings, axis=0)  → [512]
    │   L2 normalize
    │
    └── Output: {
            "embedding": np.array([0.12, -0.34, 0.08, ...]),  # 512-dim
            "has_visual": True,
            "keyframe_count": 2,
        }
```

### Aggregation strategy by keyframe count:
```
N = 0  →  has_visual = False (no valid frames)
N = 1  →  Use that single CLIP embedding directly
N = 2-5 → Average all embeddings (each is a distinct visual state)
N > 5  →  KMeans(n_clusters=3) → average 3 cluster centroids
           (prevents one dominant visual theme from diluting others)
```

### Edge cases handled:
| Scenario | What happens |
|----------|-------------|
| All frames blank | has_visual=False, embedding=None |
| Corrupted JPEG | Skip with warning |
| CLIP CUDA OOM | Moves batch to CPU, processes, moves model back to GPU |
| CLIP model download fails | Retry once, then raise error |
| 4K frames | Already resized in Phase 2, plus CLIP preprocesses to 224x224 |
| >5 keyframes in chunk | KMeans clustering to 3 centroids prevents dilution |

### Output (per chunk):
```python
visual_result = {
    "embedding": np.array([...]),  # 512 floats, L2-normalized
    "has_visual": True,
    "keyframe_count": 2,
}
```

---

## PHASE 8: TEXT EMBEDDINGS (embeddings.py → generate_text_embeddings)

### What happens:
Combine transcript + OCR text, embed using SentenceTransformer.

### Data flow:
```
For each chunk:
    transcript = "Now let's set up our Node.js project and install..."
    ocr_text   = "npm init -y npm install express jsonwebtoken"
    combined   = "Now let's set up our Node.js project and install... npm init -y npm install express jsonwebtoken"
    │
    ├── Truncate if > 10,000 chars (SentenceTransformer max is 512 tokens anyway)
    │
    └── Collect all non-empty texts

Batch encode:
    model.encode(["combined text chunk 0", "combined text chunk 1", ...],
                 batch_size=64, normalize_embeddings=True)
    │
    └── Returns: [np.array(384), np.array(384), ..., None, np.array(384), ...]
                  ↑ chunk 0       ↑ chunk 1            ↑ chunk with no text
```

### Why transcript + OCR are combined:
```
Transcript alone: "Now let's install the dependencies"
OCR alone:        "npm install express jsonwebtoken bcrypt"

Combined embedding captures BOTH what the instructor said AND what's on screen.
A search for "install jsonwebtoken" matches this chunk because the OCR text contributes.
```

### Edge cases handled:
| Scenario | What happens |
|----------|-------------|
| No transcript AND no OCR | Returns None for that chunk (has_text=False) |
| Only OCR, no transcript | Still embeds (source="ocr_only") |
| Only transcript, no OCR | Still embeds (source="transcript_only") |
| Text > 10,000 chars | Truncated |

### Output:
```python
text_embeddings = [
    np.array([0.05, 0.78, -0.12, ...]),  # 384-dim, chunk 0
    np.array([0.23, 0.45, 0.67, ...]),   # 384-dim, chunk 1
    None,                                  # chunk 5 had no text
    np.array([...]),                       # 384-dim, chunk 6
    ...
]
```

---

## PHASE 9: STORE IN QDRANT (storage.py → store_chunks)

### What happens:
Each chunk becomes a Qdrant point with named vectors and rich payload.

### Data flow:
```
For each chunk i:
    text_emb = text_embeddings[i]     → 384-dim or None
    vis_emb  = visual_embeddings[i]   → 512-dim or None

    IF both are None → SKIP this chunk (no searchable content)

    vectors = {}
    IF text_emb is not None:  vectors["text"]   = [0.05, 0.78, ...] (384 floats)
    IF vis_emb is not None:   vectors["visual"] = [0.12, -0.34, ...] (512 floats)

    point = PointStruct(
        id = "uuid4",
        vector = vectors,
        payload = {
            "video_id": "a3f8b2c1d4e5f678",
            "video_filename": "lecture.mp4",
            "video_title": "lecture",
            "video_duration": 3600.0,
            "video_upload_date": "2026-03-20T10:30:00+00:00",
            "chunk_index": 0,
            "total_chunks": 24,
            "timestamp_start": 0.0,
            "timestamp_end": 62.3,
            "transcript_text": "Today we will learn about JWT...",
            "ocr_text": "JWT Authentication const jwt...",
            "detected_language": "en",
            "has_audio": True,
            "has_transcript": True,
            "has_visual": True,
            "has_ocr": True,
            "source": "both",
            "neighbor_density": 0.73,
            "keyframe_count": 2,
            "keyframe_timestamps": [42.3, 55.8],
            "prev_context": "",
            "next_context": "Now let's set up our Node.js project",
            "low_confidence": False,
        }
    )
```

### Qdrant collection structure:
```
Collection: "video_chunks"
    Named vectors:
        "text":   384-dim, Cosine distance
        "visual": 512-dim, Cosine distance

    Points can have:
        ├── Both vectors    (source = "both")
        ├── Text only       (source = "transcript_only" or "ocr_only")
        └── Visual only     (source = "visual_only")

    Missing vectors are OMITTED, not set to zero.
```

### Batch upsert:
```
24 chunks → filter out 0 with no vectors → 24 points
    Batch 1: points[0:100]  → client.upsert()
    (If batch fails → retry once → if still fails, log error, continue)
```

### Duplicate video handling:
```
video_id = SHA256(first_10MB + filesize)[:16]

Before ingestion:
    client.scroll(filter: video_id == "a3f8b2c1d4e5f678", limit=1)
    If found:
        --reindex flag → delete_video() then re-ingest
        No flag → "Video already indexed" → skip
```

### Edge cases handled:
| Scenario | What happens |
|----------|-------------|
| Chunk with no text AND no visual embedding | Skipped with warning |
| Null/None vectors | Omitted from vector dict (not inserted as zeros) |
| Qdrant connection drops mid-upsert | Retry batch once, then log and continue |
| Very long transcript text | Truncated to 10,000 chars |
| Null bytes in text | Stripped with .replace("\x00", "") |
| Collection exists with wrong dimensions | RuntimeError raised before ingestion |
| Same video re-ingested | Detected by video_id hash, skip or reindex |

### Output:
```
24 points stored in Qdrant collection "video_chunks"
```

---

## PHASE 10: CLEANUP (cleanup.py)

```
Delete: data/frames/a3f8b2c1d4e5f678/   (all keyframe JPEGs)
Delete: data/audio/a3f8b2c1d4e5f678.wav  (extracted audio)

If --keep-temp flag: skip cleanup
If permission error: log warning, don't crash
Runs in try/finally: cleanup happens even if pipeline crashes
```

---

---

# QUERY FLOW (query.py → search)

Now let's trace what happens when a user searches.

## Step-by-step query flow:

```
User query: "how to implement JWT authentication"
    │
    ├──[1]── Clean query
    │         Strip whitespace, truncate if >50 words
    │         → "how to implement JWT authentication"
    │
    ├──[2]── Generate TWO query embeddings
    │         │
    │         ├── SentenceTransformer: "how to implement JWT authentication"
    │         │   → text_emb = [0.12, 0.45, -0.23, ...] (384-dim)
    │         │   (Same model used during indexing → same vector space)
    │         │
    │         └── CLIP text encoder: "how to implement JWT authentication"
    │             → clip_emb = [0.08, -0.15, 0.67, ...] (512-dim)
    │             (CLIP can encode text into the SAME space as images)
    │
    ├──[3]── Two parallel searches in Qdrant
    │         │
    │         ├── Text search: query "text" vector with text_emb
    │         │   → top 20 hits by cosine similarity
    │         │   Hit 1: chunk 0, score 0.89 (instructor explains JWT)
    │         │   Hit 2: chunk 3, score 0.72 (mentions JWT in passing)
    │         │   ...
    │         │
    │         └── Visual search: query "visual" vector with clip_emb
    │             → top 20 hits by cosine similarity
    │             Hit 1: chunk 0, score 0.45 (screen shows JWT code)
    │             Hit 2: chunk 7, score 0.38 (screen shows auth diagram)
    │             ...
    │
    ├──[4]── Merge candidates (deduplicate by point ID)
    │         │
    │         candidate_pool = {
    │             "uuid-chunk-0": {text_score: 0.89, visual_score: 0.45},
    │             "uuid-chunk-3": {text_score: 0.72, visual_score: 0.0},  ← no visual match
    │             "uuid-chunk-7": {text_score: 0.0,  visual_score: 0.38}, ← no text match
    │             ...
    │         }
    │
    ├──[5]── Re-rank each candidate with multi-signal scoring
    │         │
    │         For chunk 0:
    │         │
    │         ├── Base score:
    │         │   has_text=True, has_visual=True → use both weights
    │         │   base = (0.6 × 0.89) + (0.3 × 0.45) + (0.1 × 0.73)
    │         │        = 0.534     + 0.135     + 0.073
    │         │        = 0.742
    │         │
    │         ├── Time decay:
    │         │   video_upload_date = "2026-03-20" → 3 days ago
    │         │   age_days = 3 < 180 → time_factor = 1.1 (recent boost)
    │         │
    │         ├── Title boost:
    │         │   query words: {"implement", "jwt", "authentication"}
    │         │   title words: {"jwt", "authentication", "tutorial"}
    │         │   overlap: {"jwt", "authentication"} → title_boost = 1.5
    │         │
    │         └── Final score:
    │             final = 0.742 × 1.1 × 1.5 = 1.224
    │
    │         For chunk 3 (passing mention, text-only):
    │         │   base = (0.9 × 0.72) + (0.1 × 0.17) = 0.665
    │         │   ← weight redistribution: text gets 0.6+0.3=0.9 because no visual
    │         │   time_factor = 1.1, title_boost = 1.5
    │         │   final = 0.665 × 1.1 × 1.5 = 1.097
    │         │
    │         │   Note: density 0.17 (passing mention) hurts this chunk
    │         │   vs chunk 0's density 0.73 (sustained discussion)
    │
    ├──[6]── Sort by final_score, take top K
    │         │
    │         1. Chunk 0:  score=1.224 (deep explanation + visual + title match)
    │         2. Chunk 1:  score=1.105 (continues JWT topic)
    │         3. Chunk 3:  score=1.097 (passing mention, penalized by density)
    │         4. Chunk 7:  score=0.632 (auth diagram, visual-only match)
    │         5. Chunk 12: score=0.445 (mentions "token" in different context)
    │
    └──[7]── Return formatted results
              │
              └── [
                    {
                      "video_title": "JWT Authentication Tutorial",
                      "timestamp_start": 0.0,
                      "timestamp_end": 62.3,
                      "transcript_snippet": "Today we will learn about JWT authentication...",
                      "score": 1.224,
                      "score_breakdown": {
                          "text": 0.89,
                          "visual": 0.45,
                          "density": 0.73,
                          "time_factor": 1.1,
                          "title_boost": 1.5
                      },
                      "prev_context": "",
                      "next_context": "Now let's set up our Node.js project"
                    },
                    ...
                  ]
```

## Why the two embedding spaces work together:

```
The query "how to implement JWT authentication" gets encoded TWO ways:

1. SentenceTransformer embedding (384-dim):
   Lives in the SAME space as the indexed transcript+OCR text
   Matches: chunks where the instructor SAYS "JWT" or where OCR detected "JWT" on screen

2. CLIP text embedding (512-dim):
   Lives in the SAME space as the indexed keyframe images
   Matches: chunks where the SCREEN SHOWS JWT-related content
   (code editors with auth files, architecture diagrams, login screens)

These are INDEPENDENT searches. A chunk that scores high on BOTH
means the instructor was talking about JWT while showing JWT code.
That's the most relevant result → ranked highest.
```

## Score weighting breakdown:
```
final_score = base_score × time_factor × title_boost

base_score = (0.6 × text) + (0.3 × visual) + (0.1 × density)
              │                │                 │
              │                │                 └── Was this topic discussed at length?
              │                │                     High = sustained discussion (0.7-1.0)
              │                │                     Low = passing mention (0.1-0.3)
              │                │
              │                └── Does the SCREEN show what the user is looking for?
              │                    Encoded via CLIP text→image similarity
              │
              └── Does the SPEECH/OCR TEXT match what the user is looking for?
                  Encoded via SentenceTransformer similarity

time_factor:    0.8 if video > 2 years old  (penalize stale content)
                1.0 if 6 months – 2 years
                1.1 if < 6 months old        (boost recent content)

title_boost:    1.5 if query words appear in video title
                1.0 otherwise
```

## Query edge cases handled:
| Scenario | What happens |
|----------|-------------|
| Empty query | Returns empty list immediately |
| Query > 200 words | Truncated to first 50 words |
| No results at all | Returns "No matching content found" message |
| All scores < 0.3 | Returns results with low-confidence warning |
| Visual-only chunks (no text) | Scored on visual + density only, text weight redistributed |
| Text-only chunks (no visual) | Scored on text + density only, visual weight redistributed |
| CLIP text encoder fails | Falls back to text-only search |
| Qdrant unreachable | Raises connection error |
| Cross-language query | SentenceTransformer is multilingual, works but warns |
| Filter by specific video | Qdrant filter: video_id == "xyz" |

---

# COMPLETE DATA TRANSFORMATION SUMMARY

```
INPUT: lecture.mp4 (1 hour, 1080p, has audio)
│
├── Phase 0 → video_meta: {duration: 3600, has_audio: true, ...}
│
├── Phase 1 → audio.wav (115 MB, mono 16kHz)
│
├── Phase 2 → 45 keyframes (each a distinct visual state)
│
├── Phase 3 → 142 segments [{start, end, text, confidence}, ...]
│             (one sentence each, 3-10 seconds)
│
├── Phase 4 → 24 chunks [{start, end, text, neighbor_density, prev/next_context}, ...]
│             (one topic each, 30-120 seconds)
│             Segments → Chunks via:
│               - Hard boundaries (phrases, pauses, topic shifts)
│               - Soft merging (similarity threshold + duration limits)
│               - Enrichment (overlap context, density scores)
│
├── Phase 5 → Each chunk gets its keyframes mapped by timestamp
│             (1-3 keyframes per chunk typically)
│
├── Phase 6 → Each chunk gets OCR text from its keyframes
│             (deduplicated, translated if non-English)
│
├── Phase 7 → Each chunk gets a 512-dim CLIP visual embedding
│             (aggregated from its keyframes)
│
├── Phase 8 → Each chunk gets a 384-dim text embedding
│             (from transcript + OCR combined)
│
├── Phase 9 → 24 Qdrant points, each with:
│             - "text" vector (384-dim) — what was SAID + written on screen
│             - "visual" vector (512-dim) — what was SHOWN on screen
│             - rich payload (timestamps, text, scores, metadata)
│
└── Phase 10 → Temp files deleted

QUERY: "how to implement JWT"
│
├── Generate text embedding (384-dim) + CLIP text embedding (512-dim)
├── Search both vector spaces in Qdrant (top 20 each)
├── Merge + deduplicate candidates
├── Re-rank: base_score × time_factor × title_boost
└── Return top 5 with score breakdown + timestamps + text snippets
```
