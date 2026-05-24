# Video Semantic Search

Upload your videos. Search them like you'd search Google — by meaning, not just keywords.

Instead of hunting through transcripts for exact words, just ask a question. The system finds the right timestamp, even if nobody said those exact words.

## What This Does

- **Transcribes** your videos with speech-to-text (Whisper)
- **Extracts** on-screen text (OCR)
- **Analyzes** visuals frame-by-frame (CLIP embeddings)
- **Chunks** videos into coherent segments
- **Stores** everything in a vector database
- **Searches** all of it semantically — meaning-based, not keyword-matching

## Quick Start

### Prerequisites

Make sure you have:
- **Python 3.11+** — https://www.python.org/downloads/
- **Node.js 18+** — https://nodejs.org
- **ffmpeg** — `winget install Gyan.FFmpeg` (Windows) or `brew install ffmpeg` (Mac)
- **Docker** — https://www.docker.com/products/docker-desktop

### Setup (5 minutes)

```powershell
# Create a virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt
pip install -r server/requirements.txt

# Install frontend dependencies
cd web
npm install
cd ..
```

### Run It (3 terminals)

**Terminal 1 — Vector database**
```powershell
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Terminal 2 — Backend API**
```powershell
.venv\Scripts\Activate.ps1
uvicorn server.api:app --reload --port 8000
```

**Terminal 3 — Frontend**
```powershell
cd web
npm run dev
```

Open **http://localhost:5173** in your browser.

## How to Use

### Upload Videos
1. Click **Upload**
2. Drag a video file (MP4, MKV, WebM, AVI, etc.)
3. Watch the progress bar
4. Pipeline runs: validation → audio extraction → transcription → keyframe extraction → OCR → visual embeddings → storage

A 5-minute video takes ~2-4 minutes on CPU (slower at "Transcribing" step — that's normal).

### Search
1. Click **Search**
2. Type a natural question, not keywords:
   - "how does authentication work"
   - "what is neural networks"
   - "show me the config file"
3. Get back ranked results with timestamps, snippets, score breakdown

### Admin
See all indexed videos, chunk counts, detected language, and when they were added. Delete videos here.

## Why This Exists

Keyword search is dumb. If you say "authentication" in a video but someone searches "how do I log in", they won't find it. Semantic search understands *meaning*. It works across multiple modalities (what's said, what's on screen, what's shown visually).

## Architecture

```
Browser (React)  ←→  FastAPI Backend  ←→  Qdrant Vector DB
                           ↓
                    Whisper (transcribe)
                    EasyOCR (text on screen)
                    CLIP (visual embeddings)
```

- **Frontend**: React SPA (Vite) on port 5173
- **Backend**: FastAPI on port 8000, runs pipeline in background threads
- **Database**: Qdrant (vector DB) on port 6333

Full architecture details in [ARCHITECTURE.md](ARCHITECTURE.md).

## Configuration

Most settings are in [config.py](config.py). Key ones:

- **Chunk duration**: `MIN_CHUNK_DURATION = 60`, `MAX_CHUNK_DURATION = 180` (seconds)
- **Whisper model**: `WHISPER_MODEL = "base"` (use `"tiny"` for speed, `"medium"`/`"large"` for accuracy)
- **Search weights**: `TEXT_WEIGHT = 0.6`, `VISUAL_WEIGHT = 0.3`, `DENSITY_WEIGHT = 0.1`

Tweak these if results don't feel right or if it's too slow.

## Command Line (No UI)

```bash
# Index a video
python main.py ingest path/to/video.mp4

# Search without UI
python main.py query "how does JWT work" --top-k 5

# Batch index
for f in videos/*.mp4; do python main.py ingest "$f"; done
```

## Performance Notes

- **First query is slow** (~2-3 sec on CPU) — models load. Subsequent queries are fast (~0.3-0.5 sec).
- **Transcription is the slowest step** — Whisper on CPU takes 20-60% of the video's duration. A 5-minute video takes 1-3 minutes.
- **GPU helps a lot** — if you have NVIDIA GPU, models run 10× faster. Set `device = "cuda"` in code.

## Troubleshooting

**"ffmpeg not found"**
Install ffmpeg and restart your terminal.

**"Cannot reach Qdrant"**
Make sure Terminal 1 is running the docker command.

**Progress bar frozen at 55% (Transcribing)**
That's Whisper working. It's normal. Don't reload.

**Search is slow**
First query loads models (~2 sec). Subsequent queries are fast. Or switch to GPU.

**Search returns wrong videos**
Check the `score_breakdown` — it shows text/visual/density scores. If visual score is high but text is low, the video shows something related but doesn't say it.

**"Video already indexed" message**
You uploaded the same video before (detected by content hash). Check the **Re-index** box to replace it.

## Deployment

For a live demo or personal use:

- **Local + ngrok**: Expose your machine via ngrok tunnel — free, full speed, shared URL
- **Hugging Face Spaces**: Free 16GB RAM, public URL, but CPU-only (slow transcription)
- **Google Colab + ngrok**: Free T4 GPU, fast transcription, good for quick demos

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## What's Inside

```
.
├── README.md                 ← you are here
├── ARCHITECTURE.md           ← system design & data flow
├── GUIDE.md                  ← full setup & feature walkthrough
├── DEPLOYMENT.md             ← how to deploy
├── FRONTEND.md               ← frontend & API details
│
├── config.py                 ← all tunable settings
├── main.py                   ← CLI (ingest / query without UI)
├── requirements.txt          ← Python dependencies
│
├── server/
│   ├── api.py                ← FastAPI server
│   ├── jobs.py               ← job registry & pipeline runner
│   └── requirements.txt       ← server-specific deps
│
├── utils/
│   ├── video.py              ← ffmpeg (audio, keyframes)
│   ├── audio.py              ← Whisper transcription
│   ├── visual.py             ← CLIP embeddings
│   ├── ocr.py                ← EasyOCR
│   ├── chunking.py           ← semantic chunking
│   ├── embeddings.py         ← text embeddings
│   ├── storage.py            ← Qdrant upsert/query
│   └── query.py              ← search & re-ranking
│
├── web/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Search.jsx    ← search UI
│   │   │   ├── Upload.jsx    ← upload UI
│   │   │   └── Admin.jsx     ← admin dashboard
│   │   ├── App.jsx           ← nav layout
│   │   ├── api.js            ← fetch helpers
│   │   └── styles.css        ← styling
│   └── vite.config.js        ← dev proxy config
│
└── data/
    ├── uploads/              ← uploaded video files
    ├── frames/               ← extracted keyframes (temp)
    └── audio/                ← extracted audio (temp)
```

## Why Not Just Use [Insert Tool Here]

This is built from scratch because:

- **Existing video search tools** don't do semantic search (they're keyword-based)
- **Existing semantic search tools** don't handle video well
- **Building it yourself** is actually simpler than integrating five different services
- **You own your data** — everything runs locally

## Performance Baseline

On a MacBook M1 (laptop CPU):

- 5-minute video → ~2-3 minutes to index (Whisper dominates)
- 1000 chunks in Qdrant → ~0.3 seconds per search query
- First query → ~2 seconds (models load)
- Subsequent queries → ~0.3 seconds

With GPU (NVIDIA T4):
- 5-minute video → ~30-60 seconds to index
- Same search speed, 10× faster model loading

## Contributing / Extending

Want to:
- Use a different embedding model? Edit [utils/embeddings.py](utils/embeddings.py)
- Add keyword search for comparison? Add an FTS index in [utils/storage.py](utils/storage.py)
- Deploy on Kubernetes? Use the Dockerfile in [DEPLOYMENT.md](DEPLOYMENT.md)
- Index PDFs instead of videos? Replace ffmpeg with a PDF parser, reuse the rest

The pipeline is modular — swap pieces out as needed.

## Known Limitations

- **In-memory jobs** — if the server restarts during upload, the job is lost (Qdrant data survives)
- **Transcription quality** — depends on audio clarity; background noise = worse transcripts
- **OCR accuracy** — small text on screen is hard to read; depends on frame quality
- **No full-text search** — if exact phrase matching matters, use keyword search alongside this
- **Single-machine only** — if you index 10,000 videos, consider splitting Qdrant to a managed service

## License

Build on top of this, modify it, use it however you want.

## Questions?

Check the [full GUIDE](GUIDE.md) for detailed walkthrough. Check [ARCHITECTURE.md](ARCHITECTURE.md) for how it all works.

Still stuck? The logs are in Terminal 2 — they usually tell you what went wrong.
