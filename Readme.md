# Video Semantic Search Engine

A multimodal video search engine that indexes videos across spoken audio, on-screen text, and visual scene keyframes. Users can query the index with natural language questions and receive ranked results with exact timestamps, matched transcript/OCR snippets, and detailed relevance breakdowns.

The system combines OpenAI Whisper (speech-to-text), EasyOCR (slide/editor text recognition), SentenceTransformers (text embeddings), and CLIP (visual embeddings), with storage and dual-vector similarity search handled by Qdrant.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [How It Works: End-to-End Flow](#how-it-works-end-to-end-flow)
  - [Ingestion Pipeline](#1-ingestion-pipeline)
  - [Query and Re-ranking Flow](#2-query-and-re-ranking-flow)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Start Qdrant](#2-start-qdrant)
  - [3. Start the Backend API](#3-start-the-backend-api)
  - [4. Start the Frontend](#4-start-the-frontend)
- [CLI Usage](#cli-usage)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

Keyword-based search fails when trying to navigate video libraries:
- **Vocabulary mismatch**: A query for *"how to authenticate users"* will miss lectures where the speaker only mentions *"signing in with JWT tokens"*.
- **Unspoken information**: In software demos and technical presentations, commands, syntax, and architectural diagrams often appear on slides or code editors without being read aloud.
- **Granularity**: Standard search returns an entire video file, leaving the user to manually scrub through 60–120 minutes of footage.

This project addresses these issues by:
1. Extracting both acoustic and visual signals from videos.
2. Segmenting transcripts using dynamic semantic boundaries rather than fixed time intervals.
3. Storing dual embeddings (`text` and `visual`) within Qdrant vector points.
4. Scoring search candidates through a composite ranking function combining text similarity, visual relevance, topic density, recency, and title match.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     React Web UI (Vite)                      │
│   • Search: natural language input, timestamped cards        │
│   • Upload: drag-and-drop with real-time SSE progress bar    │
│   • Admin: video library view, point counts, deletion        │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTP / Server-Sent Events (SSE)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (:8000)                   │
│   • Endpoints: /api/upload, /api/jobs/{id}/events,           │
│                /api/query, /api/videos, /api/videos/{id}     │
│   • Worker Thread: in-process background job runner          │
│   • Model Cache: pre-warmed embeddings in RAM/VRAM           │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌─────────────────────────────┐
│      Processing Engines      │ │     Qdrant Vector DB        │
│   • FFmpeg / FFprobe         │ │     Port 6333 (HTTP / gRPC) │
│   • Whisper (base)           │ │                             │
│   • EasyOCR                  │ │ Collection: video_chunks    │
│   • SentenceTransformers     │ │  • Vector: "text" (384d)    │
│   • OpenCLIP (ViT-B/32)      │ │  • Vector: "visual" (512d)  │
└──────────────────────────────┘ └─────────────────────────────┘
```

---

## How It Works: End-to-End Flow

### 1. Ingestion Pipeline

When a video is uploaded or passed via the CLI, it moves through a 10-stage processing pipeline:

```
[0. Validation] ──> [1. Audio Extraction] ──> [2. Keyframe Extraction]
                                                     │
[5. Keyframe Mapping] <── [4. Semantic Chunking] <── [3. Whisper Transcription]
       │
       ├──> [6. OCR Processing]
       │
       ├──> [7. CLIP Visual Embeddings]
       │
       └──> [8. SentenceTransformer Text Embeddings]
                                │
                                ▼
                     [9. Qdrant Batch Upsert] ──> [10. Temp File Cleanup]
```

1. **Validation (`utils/validation.py`)**
   - Validates system dependencies (`ffmpeg`, `ffprobe`, Qdrant reachability, CUDA availability).
   - Probes the input video for valid streams, duration, dimensions, and audio track presence.
   - Computes a deterministic `video_id` based on a SHA-256 hash of the video's first 10 MB and file size to prevent duplicate ingestions.

2. **Audio Extraction (`utils/video.py`)**
   - Extracts the primary audio stream into a 16 kHz mono 16-bit PCM WAV file optimized for speech models.

3. **Keyframe Extraction (`utils/video.py`)**
   - Executes scene change detection via FFmpeg filters (`select='gt(scene,0.3)',showinfo`).
   - Parses frame timestamps directly from FFmpeg metadata output.
   - Applies an adaptive cap (`MAX_KEYFRAMES_PER_MINUTE`) to avoid frame explosion on fast-cut videos, with an interval fallback for static recordings.

4. **Speech Transcription (`utils/audio.py`)**
   - Transcribes the extracted WAV file using OpenAI Whisper.
   - Generates granular segment objects containing start time, end time, and text.
   - Splits long audio files (>30 minutes) into sliding windows with overlap to preserve stability.

5. **Semantic Chunking (`utils/chunking.py`)**
   - Rather than splitting strictly every 60 seconds, consecutive transcript segments are embedded and compared using cosine similarity.
   - Topic shifts are detected using a dynamic threshold (`mean - std_factor * std`).
   - Transitional boundary phrases (e.g., *"moving on"*, *"next topic"*) and speech pauses (>2.0s) trigger hard boundaries.
   - Chunks are constrained within configurable minimum and maximum durations (60s to 180s) and enriched with surrounding context sentences.
   - Calculates a `neighbor_density` metric to represent how sustained a topic is across adjacent chunks.

6. **Keyframe Mapping (`main.py` / `server/jobs.py`)**
   - Associates extracted keyframes with chunks based on timestamp intervals.
   - If a chunk contains no keyframes, an automatic fallback extraction captures a single frame at the chunk's midpoint.

7. **On-Screen Text OCR (`utils/ocr.py`)**
   - Runs EasyOCR over chunk keyframes.
   - Discards blank or overexposed frames using pixel luminance heuristics.
   - Deduplicates repeated on-screen text across consecutive frames and applies translation if non-Latin scripts are detected.

8. **Visual Embeddings (`utils/visual.py`)**
   - Feeds keyframes through OpenCLIP (`ViT-B-32`).
   - If a chunk contains multiple keyframes (up to 5), embeddings are averaged; for dense keyframe sequences, KMeans clustering (k=3) computes representative visual centroids.
   - Produces a 512-dimensional normalized vector.

9. **Text Embeddings (`utils/embeddings.py`)**
   - Combines the spoken transcript and on-screen OCR text.
   - Encodes text into a 384-dimensional normalized vector using `all-MiniLM-L6-v2`.

10. **Storage & Cleanup (`utils/storage.py`, `utils/cleanup.py`)**
    - Creates points in Qdrant with named vectors: `"text"` (384d) and `"visual"` (512d).
    - Writes payload metadata including start/end timestamps, transcript, OCR text, adjacent context, and density metrics.
    - Removes temporary WAV and keyframe files unless `--keep-temp` is specified.

---

### 2. Query and Re-ranking Flow

```
User Query: "how to configure database connection"
       │
       ├──> SentenceTransformer ──> Text Query Vector (384d)
       └──> OpenCLIP Text Encoder ──> Visual Query Vector (512d)
                   │
                   ▼
       Parallel Qdrant Search (Cosine Distance)
       • Top-10 text candidates
       • Top-10 visual candidates
                   │
                   ▼
       Candidate Merging & Multi-Signal Re-Ranking
                   │
                   ▼
       Video-Level Deduplication & Response
```

When a user submits a search query:
1. **Dual Vector Generation**:
   - The query is embedded via `all-MiniLM-L6-v2` for transcript matching.
   - The same query is embedded using the CLIP text encoder into the visual embedding space, allowing natural language to match visual frames (e.g., code snippets or slides).
2. **Retrieval**:
   - Two concurrent searches run against the Qdrant `video_chunks` collection.
   - The candidate pools are merged and deduplicated by chunk ID.
3. **Composite Scoring Function**:
   Candidates are scored using the following formula:
   $$\text{Final Score} = \text{Base Score} \times \text{Time Factor} \times \text{Title Boost}$$
   Where:
   - $\text{Base Score} = (0.6 \times \text{Text Similarity}) + (0.3 \times \text{Visual Similarity}) + (0.1 \times \text{Topic Density})$
   - $\text{Time Factor}$: $1.1$ for recent videos (<6 months), $0.8$ for older uploads (>2 years), $1.0$ otherwise.
   - $\text{Title Boost}$: $1.5$ multiplier applied if query terms appear in the video title.
   - $\text{Topic Density}$: Rewards chunks situated within continuous discussions of the subject rather than passing mentions.
4. **Deduplication**:
   - Results are deduplicated so the single most relevant chunk per video is returned, avoiding repetitive hits from the same file.

---

## Repository Structure

```
.
├── config.py                 # Central configuration (thresholds, model names, weights)
├── main.py                   # CLI entrypoint for ingestion and querying
├── requirements.txt          # Core ML dependencies (torch, whisper, clip, qdrant)
│
├── server/
│   ├── api.py                # FastAPI endpoints (upload, SSE progress, search, admin)
│   ├── jobs.py               # Background job runner and SSE event dispatcher
│   └── requirements.txt      # Web backend dependencies (fastapi, uvicorn, multipart)
│
├── utils/
│   ├── validation.py         # System environment, file format, and GPU checks
│   ├── video.py              # FFmpeg audio and keyframe extraction utilities
│   ├── audio.py              # Whisper speech-to-text integration
│   ├── chunking.py           # Semantic segmentation and density calculation
│   ├── ocr.py                # EasyOCR frame processing and text deduplication
│   ├── visual.py             # CLIP embedding generation and keyframe clustering
│   ├── embeddings.py         # SentenceTransformer text embedding generation
│   ├── storage.py            # Qdrant client, schema setup, point upsert, and deletion
│   ├── query.py              # Multi-vector query orchestration and candidate re-ranking
│   └── cleanup.py            # Temp file management
│
└── web/
    ├── src/
    │   ├── pages/
    │   │   ├── Search.jsx    # Search bar, query execution, result cards
    │   │   ├── Upload.jsx    # Video dropzone and SSE stage progress display
    │   │   └── Admin.jsx     # Indexed video catalog and point management
    │   ├── api.js            # Frontend HTTP and SSE communication layer
    │   ├── App.jsx           # App shell and routing
    │   └── styles.css        # Clean UI styling
    ├── package.json
    └── vite.config.js        # Local proxy configuration (/api -> localhost:8000)
```

---

## Prerequisites

Before running the project locally, verify that you have:

1. **Python 3.11 or 3.12**
2. **Node.js 18+ & npm**
3. **FFmpeg & FFprobe**: Installed and available in your system `PATH`
   ```bash
   ffmpeg -version
   ffprobe -version
   ```
4. **Docker Desktop**: Required to run the local Qdrant container

---

## Local Setup

### 1. Environment Setup

Clone the repository and set up your Python virtual environment:

```powershell
# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install backend and pipeline dependencies
pip install -r requirements.txt
pip install -r server/requirements.txt

# Install frontend dependencies
cd web
npm install
cd ..
```

*Note: On the first run, model weights for Whisper `base` (~145 MB), SentenceTransformer `all-MiniLM-L6-v2` (~90 MB), and CLIP `ViT-B/32` (~350 MB) will download and cache locally.*

---

### 2. Start Qdrant

Launch Qdrant using Docker with a persistent storage volume:

```powershell
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

The Qdrant REST dashboard is accessible at `http://localhost:6333/dashboard`.

---

### 3. Start the Backend API

In a second terminal window (with the virtual environment activated):

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn server.api:app --reload --port 8000
```

The API will warm up the embedding models during startup and bind to `http://localhost:8000`.

---

### 4. Start the Frontend

In a third terminal window:

```powershell
cd web
npm run dev
```

Open your browser to `http://localhost:5173`.

---

## CLI Usage

The ingestion pipeline and query interface can be run directly from the command line without starting the web servers:

### Ingest a Video
```powershell
.\.venv\Scripts\Activate.ps1

# Standard ingestion
python main.py path/to/video.mp4

# Force re-indexing of an already ingested video
python main.py path/to/video.mp4 --reindex

# Keep extracted frames and WAV audio for manual inspection
python main.py path/to/video.mp4 --keep-temp

# Fast ingestion without visual embedding or OCR
python main.py path/to/video.mp4 --skip-ocr --skip-visual
```

### Search from the Terminal
```powershell
python main.py --query "how does token signing work"
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Accepts a multipart video file, writes to disk, and queues a background ingest job. Returns `{ job_id }`. |
| `GET` | `/api/jobs/{id}/events` | SSE endpoint streaming real-time pipeline status, progress percentage, and current processing stage. |
| `POST` | `/api/query` | Submits a query payload `{"query": "...", "top_k": 5}`. Returns ranked timestamps and snippets. |
| `GET` | `/api/videos` | Returns a list of all indexed videos with chunk counts and metadata. |
| `DELETE` | `/api/videos/{id}` | Deletes all chunks associated with `video_id` from Qdrant. |

---

## Configuration Reference

Key pipeline variables can be adjusted in `config.py`:

| Parameter | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `"base"` | Speech recognition model size (`tiny`, `base`, `small`, `medium`, `large`). |
| `CLIP_MODEL_NAME` | `"ViT-B-32"` | Visual embedding backbone. |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Text embedding model (384 dimensions). |
| `MIN_CHUNK_DURATION` | `60` | Minimum target length in seconds for a semantic chunk. |
| `MAX_CHUNK_DURATION` | `180` | Maximum duration before forcing a chunk boundary. |
| `TEXT_WEIGHT` | `0.6` | Weight assigned to text similarity in composite scoring. |
| `VISUAL_WEIGHT` | `0.3` | Weight assigned to visual similarity in composite scoring. |
| `DENSITY_WEIGHT` | `0.1` | Weight assigned to neighbor topic density in composite scoring. |
| `TITLE_BOOST_FACTOR` | `1.5` | Relevance score multiplier when query matches words in the video title. |

---

## Troubleshooting

- **`ffmpeg` / `ffprobe` not recognized:**
  Ensure FFmpeg is installed and added to your system `PATH`. Restart your terminal or system after installation.
- **Qdrant Connection Error:**
  Verify that the Docker container is running (`docker ps`) and that port `6333` is mapped properly.
- **CUDA Out of Memory:**
  If running on a GPU with limited VRAM (<4 GB), Whisper or CLIP may hit memory limits. The pipeline contains automatic fallbacks to CPU when OOM conditions occur, or you can switch `WHISPER_MODEL = "tiny"` in `config.py`.
- **First Search Query Latency:**
  The server pre-loads SentenceTransformer and OpenCLIP models during startup. If you restart without `--reload`, allow a few seconds for initialization.
