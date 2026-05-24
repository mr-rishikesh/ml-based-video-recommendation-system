# System Architecture — End-to-End

A complete walkthrough of how data flows from a user in the browser, through
the FastAPI backend, into the ingestion pipeline, into Qdrant, and back out
to the frontend as ranked search results.

---

## 1. Components

```
┌───────────────────────┐     HTTP + SSE      ┌─────────────────────────┐
│ Browser               │ ──────────────────► │ FastAPI server          │
│  - React SPA (Vite)   │ ◄────────────────── │  :8000   server/api.py  │
│  - /search /upload    │                     │                         │
│  - /admin             │                     │  ┌───────────────────┐  │
└───────────────────────┘                     │  │ JobRegistry       │  │
                                              │  │ (in-memory)       │  │
                                              │  └──────┬────────────┘  │
                                              │         ▼               │
                                              │  ┌───────────────────┐  │
                                              │  │ Worker thread     │  │
                                              │  │ run_ingest_job()  │  │
                                              │  │   └─ utils/*      │  │
                                              │  └──────┬────────────┘  │
                                              └─────────┼───────────────┘
                                                        │ upserts
                                                        ▼
                                              ┌─────────────────────────┐
                                              │ Qdrant  :6333           │
                                              │ collection: video_chunks│
                                              │ named vectors:          │
                                              │   text   (384, cosine)  │
                                              │   visual (512, cosine)  │
                                              └─────────────────────────┘
```

**Processes you run locally**
- `qdrant` — the vector database (Docker container)
- `uvicorn server.api:app` — FastAPI app on port 8000
- `vite` — dev server for the SPA on port 5173 (proxies `/api` → 8000)

There is no separate worker process. Ingestion runs in a **background thread
inside the FastAPI process**. That's intentional: one ingest job at a time is
the realistic limit anyway (Whisper + CLIP saturate a machine), and keeping
it in-process avoids a task queue, a broker, and inter-process state.

---

## 2. The data model

Everything in the system centers on a **chunk**: a coherent time window of a
video, typically 60–180 seconds. Each chunk becomes one point in Qdrant, with
up to two named vectors:

| Vector | Dim | Source | Model |
|---|---|---|---|
| `text`   | 384 | transcript (Whisper) ∪ OCR text (EasyOCR) | `all-MiniLM-L6-v2` |
| `visual` | 512 | keyframes in the chunk's time range      | CLIP `ViT-B-32`    |

Same chunk → same time window → both vectors describe the same moment from
different modalities. Re-ranking combines them with:

```
base_score   = 0.6·text_score + 0.3·visual_score + 0.1·density_score
final_score  = base_score · time_factor · title_boost
```

Alongside vectors, each point carries a **payload** used for filtering,
display, and re-ranking. See [utils/storage.py](utils/storage.py:201) for the
full shape — notable fields: `video_id`, `video_title`, `timestamp_start/end`,
`transcript_text`, `ocr_text`, `detected_language`, `neighbor_density`,
`prev_context`, `next_context`.

`video_id` is a deterministic SHA-256 over `file_size || first_10_MB`
([utils/storage.py:75](utils/storage.py#L75)). Same file content → same id,
regardless of filename. This is how "already indexed?" and delete-by-video
work without any separate videos table — Qdrant payload is the source of
truth.

---

## 3. Upload flow (frontend → disk → pipeline → Qdrant)

### 3.1 Browser initiates upload
[web/src/pages/Upload.jsx](web/src/pages/Upload.jsx) calls
`uploadVideo(file, { reindex })` from [web/src/api.js](web/src/api.js). It
uses `XMLHttpRequest` rather than `fetch` because `fetch` doesn't expose
upload progress — the file byte counter you see on the first progress bar
comes from `xhr.upload.onprogress`.

### 3.2 FastAPI receives the multipart body
[server/api.py](server/api.py) — `POST /api/upload`:

```python
async def upload_video(file: UploadFile = File(...), reindex: bool = False):
    # validate extension
    # stream to disk 1 MB at a time:
    while chunk := await file.read(1024 * 1024):
        f.write(chunk)
    # reject empty uploads
    job = registry.create(filename=file.filename)
    threading.Thread(target=run_ingest_job, args=(job, path, reindex),
                     daemon=True).start()
    return {"job_id": job.id, "filename": file.filename}
```

Key points:
- The upload is **streamed to `data/uploads/`** — memory use stays flat even
  for multi-GB files.
- A `Job` is created in memory with a short UUID; control returns to the
  browser immediately with the `job_id`.
- A daemon thread is spawned to run the pipeline. The HTTP request is done.

### 3.3 Browser subscribes to progress
Back in [Upload.jsx](web/src/pages/Upload.jsx), as soon as the upload call
resolves, the frontend opens an `EventSource` to
`/api/jobs/{id}/events`. That's [`streamJobEvents`](web/src/api.js):

```js
const es = new EventSource(`/api/jobs/${jobId}/events`);
es.onmessage = (e) => { const data = JSON.parse(e.data); onEvent(data); ... };
```

The SPA's state updates on every incoming event. When `status === "done"` or
`"error"` the connection closes.

### 3.4 Server emits SSE events
[server/api.py](server/api.py) — `GET /api/jobs/{id}/events`:

```python
def generator():
    yield f"data: {json.dumps(initial_state)}\n\n"
    while True:
        evt = job.events.get(timeout=30)   # blocking queue read
        yield f"data: {json.dumps(evt)}\n\n"
        if evt["status"] in ("done", "error"):
            break
return StreamingResponse(generator(), media_type="text/event-stream")
```

Each `Job` owns a `queue.Queue` of events
([server/jobs.py](server/jobs.py)). The pipeline pushes; the SSE generator
pops. Multiple clients can watch the same job; each opens its own
`EventSource` and the server instantiates a fresh generator — but note the
queue has a single consumer semantically, so for now assume one watcher per
job (typical: the upload page itself).

### 3.5 Pipeline emits progress as it runs
[server/jobs.py](server/jobs.py) wraps each pipeline step with `mark(step)`
which maps the step name to a completion percentage:

```python
STEPS = [
    ("validation",          0.05, "Validating video"),
    ("audio_extraction",    0.15, "Extracting audio"),
    ("keyframe_extraction", 0.30, "Extracting keyframes"),
    ("transcription",       0.55, "Transcribing audio"),
    ("chunking",            0.60, "Creating semantic chunks"),
    ("ocr",                 0.75, "Running OCR on frames"),
    ("visual",              0.88, "Computing visual features"),
    ("text_embeddings",     0.93, "Generating text embeddings"),
    ("storage",             1.00, "Storing in vector database"),
]
```

`mark("transcription")` calls `job.emit(status="running", progress=0.55,
step="Transcribing audio", ...)` — that pushes an event onto the queue,
which the SSE generator flushes to the browser, which updates the progress
bar.

A concrete event looks like:
```json
{ "status": "running", "progress": 0.55,
  "step": "Transcribing audio",
  "message": "Transcribing ~315s of audio (this may take a while)",
  "video_id": null, "error": null }
```

### 3.6 The pipeline steps (what's actually happening)
Running in the worker thread, roughly in order:

| # | Step | File | What it does |
|---|---|---|---|
| 1 | validate | [utils/validation.py](utils/validation.py) | ffprobe for duration/codec/streams, ffmpeg/Qdrant/GPU checks |
| 2 | audio | [utils/video.py:33](utils/video.py#L33) | ffmpeg → mono 16 kHz WAV in `data/audio/` |
| 3 | keyframes | [utils/video.py:66](utils/video.py#L66) | ffmpeg scene-change detection → JPEGs in `data/frames/{video_id}/` |
| 4 | transcribe | [utils/audio.py](utils/audio.py) | Whisper segments with timestamps + confidence |
| 5 | chunk | [utils/chunking.py](utils/chunking.py) | merge segments into 60–180 s chunks using similarity + boundary signals |
| 6 | keyframe→chunk map | [server/jobs.py](server/jobs.py) | assign each keyframe to its time-window chunk |
| 7 | OCR | [utils/ocr.py](utils/ocr.py) | EasyOCR on keyframes, dedup, optional translation |
| 8 | CLIP | [utils/visual.py](utils/visual.py) | per-chunk 512-dim visual embedding |
| 9 | text embed | [utils/embeddings.py](utils/embeddings.py) | per-chunk 384-dim `all-MiniLM-L6-v2` of `transcript + " " + ocr` |
| 10 | store | [utils/storage.py](utils/storage.py) | batch upsert into Qdrant |

### 3.7 Finalization
When the last step succeeds, `job.emit(status="done", progress=1.0,
message="Indexed N chunks")` fires. The SSE stream delivers it, closes,
and the UI shows "✓ Complete". In the `finally` block,
`cleanup_video_files(video_id)` removes extracted frames and the audio WAV.
The original upload in `data/uploads/` stays (that's the user's file).

If any step raises, the exception is caught at the top of
`run_ingest_job`, `job.emit(status="error", ...)` delivers the message,
and cleanup still runs.

---

## 4. Query flow (frontend → scoring → frontend)

### 4.1 Browser submits query
[web/src/pages/Search.jsx](web/src/pages/Search.jsx) calls
`queryVideos(text, { topK })`:

```js
POST /api/query
Content-Type: application/json
{ "text": "how does JWT authentication work", "top_k": 5 }
```

The UI sets `loading=true`, clears prior results, shows a spinner.

### 4.2 Server encodes the query twice
[server/api.py](server/api.py) → [utils/query.py](utils/query.py):

```python
text_emb = sentence_transformer.encode([query])[0]   # 384-dim
clip_emb = clip_encode_text(query, device=device)    # 512-dim
```

Two different embedding spaces, because the two Qdrant named vectors live
in two different spaces. We search each separately.

### 4.3 Two Qdrant searches, each top-20
```python
text_results   = client.query_points(collection="video_chunks",
                                     query=text_emb, using="text", limit=20)
visual_results = client.query_points(collection="video_chunks",
                                     query=clip_emb, using="visual", limit=20)
```

Each response is a list of hits with `{id, score, payload}` — the payload is
what we stored at ingest time.

### 4.4 Candidate merge
[utils/query.py](utils/query.py) builds a dict keyed by point id. If a point
appears in both result sets, it accumulates both scores; if only one, the
other score stays 0.

### 4.5 Multi-signal re-ranking (per chunk)
For each candidate:

```
has_text = payload.has_transcript OR payload.has_ocr
has_vis  = payload.has_visual
density  = payload.neighbor_density      # 0–1, computed at ingest

if has_text and has_vis:
    base = 0.6·text + 0.3·visual + 0.1·density
elif has_text:                           # re-allocate the missing modality's weight
    base = 0.9·text + 0.1·density
else:
    base = 0.9·visual + 0.1·density

time_factor = 1.1 if age < 180d else 0.8 if age > 730d else 1.0
title_boost = 1.5 if query_words ∩ title_words else 1.0
final = base · time_factor · title_boost
```

### 4.6 Video-level deduplication
`results.sort(...)` ranks every chunk. Then we collapse **one chunk per
video_id**, picking the best-scoring chunk per video. When a video has
multiple chunks in the candidate set, we nudge the winner's score up using a
weighted blend with the second-best chunk from that same video — videos with
sustained relevance outrank a lucky single-chunk match from a less relevant
video.

This lives in [utils/query.py](utils/query.py):

```python
seen = {}
for r in results:
    if r.video_id not in seen:
        seen[r.video_id] = r
    else:
        seen[r.video_id].score = 0.7·existing.score + 0.3·r.score
results = sorted(seen.values(), key=score, desc)[:top_k]
```

### 4.7 Response shape
```json
{
  "query": "how does JWT authentication work",
  "count": 3,
  "results": [
    {
      "video_id": "a1b2c3d4e5f60718",
      "video_filename": "1700000000_jwt_explained.mp4",
      "video_title": "jwt explained",
      "timestamp_start": 62.4, "timestamp_end": 180.0,
      "transcript_snippet": "so a JSON Web Token is basically three parts ...",
      "ocr_text": "header.payload.signature",
      "score": 0.874,
      "score_breakdown": {
        "text": 0.812, "visual": 0.641, "density": 0.93,
        "time_factor": 1.1, "title_boost": 1.5
      },
      "prev_context": "let me show you how auth works",
      "next_context": "now let's compare it to sessions",
      "source": "both"
    }
  ]
}
```

### 4.8 Browser renders
[Search.jsx](web/src/pages/Search.jsx) clears the spinner, renders one
`ResultCard` per item — title, filename, `mm:ss – mm:ss` timestamp, snippet,
on-screen text, expandable score breakdown. If `count === 0` we show the
fallback "No matches — try broader terms."

---

## 5. Admin flow (list + delete)

### 5.1 List videos
[web/src/pages/Admin.jsx](web/src/pages/Admin.jsx) polls every 3 s:

```
GET /api/videos   → Array<{
  video_id, video_filename, video_title, video_duration,
  video_upload_date, detected_language, chunk_count
}>
```

[server/api.py](server/api.py) implements this by **scrolling every payload
in the collection** and grouping by `video_id`. This is O(n) in total
chunks. Fine up to a few thousand videos; if you expect more, maintain a
separate aggregate (SQLite, or a second Qdrant collection keyed on
`video_id`).

Active jobs come from `/api/jobs` — the frontend filters to the ones with
`status in {queued, running}` and shows them in a small panel above the
table.

### 5.2 Delete a video
```
DELETE /api/videos/{video_id}
```
[utils/storage.py](utils/storage.py) scrolls all points with that
`video_id`, collects ids, issues a single bulk delete. Returns
`{ deleted_points: N }`. The admin page auto-refreshes on the next poll and
the row vanishes.

---

## 6. Protocol summary — who sends what, when

### Upload
```
Client                         Server
──────                         ──────
POST /api/upload  (multipart) ─►
                              ◄─ 200 { job_id: "abc123" }
GET  /api/jobs/abc123/events ─►
                              ◄─ data: {status:"running", progress:0.05, ...}
                              ◄─ data: {status:"running", progress:0.15, ...}
                              ◄─ data: {status:"running", progress:0.55, ...}
                              ◄─ data: {status:"running", progress:0.75, ...}
                              ◄─ data: {status:"done", progress:1.0, video_id:"..."}
                                 (server closes stream)
```

### Query
```
Client                                  Server                       Qdrant
──────                                  ──────                       ──────
POST /api/query {text,top_k} ─────────►
                                        encode text (384)
                                        encode clip text (512)
                                        POST /collections/.../query_points  ─►
                                        POST /collections/.../query_points  ─►
                                                                              ◄─ top-20
                                                                              ◄─ top-20
                                        re-rank + per-video dedup
                                      ◄─ 200 {query, count, results:[...]}
```

### Admin list (every 3 s)
```
Client                      Server                       Qdrant
──────                      ──────                       ──────
GET /api/videos ─────────►
                            scroll all points  ──────────►
                                                           ◄─ batches of 256 payloads
                            group by video_id
                          ◄─ [{video_id, title, ...}]
GET /api/jobs ───────────►
                          ◄─ [{id, filename, status, progress, step, ...}]
```

---

## 7. Error handling contract

| Layer | Failure mode | What happens | What the user sees |
|---|---|---|---|
| Upload | Bad extension | 400 from `/api/upload` | Red error under form |
| Upload | Empty file | 400 from `/api/upload` | Red error under form |
| Pipeline | ffprobe / Whisper / Qdrant exception | `job.emit(status="error", error=str(exc))` | "✗ Failed — {message}" on upload page |
| Pipeline | Video already indexed (no `reindex`) | `status="done", message="already indexed"` | Complete, with the message |
| Query | Empty text | 400 from `/api/query` | Red error under search bar |
| Query | Qdrant unreachable | 500 | Red error under search bar |
| Query | All scores < 0.3 | Returned but logged | `low_score` warning in logs; results still shown |
| Admin | Qdrant unreachable | 200 with `[]` (logged warning) | Empty list |

The pipeline never leaves a video half-indexed in a way that corrupts state:
`ensure_collection` validates schema, `store_chunks` is idempotent per
point, and `delete_video` is all-or-nothing. Worst case of a mid-pipeline
crash is a video with fewer chunks than it should have — re-running with
`reindex=True` fixes it.

---

## 8. Why in-memory jobs are OK here

- Jobs are short-lived (seconds to minutes). If the server dies mid-job,
  the user retries. No "half-processed forever" state because Qdrant only
  sees the upsert at the end.
- Browser reload loses the job reference. Fix: keep `job_id` in
  `localStorage` and resume on reload. Not implemented — keeping the
  surface small.
- Multiple uvicorn workers would each have their own registry and miss each
  other's jobs. Run with `--workers 1` (the default with `--reload`) or
  put a real task queue behind it.

When you outgrow in-memory jobs:
1. Swap `JobRegistry` for Redis-backed storage (same interface).
2. Run workers as separate processes (Celery / RQ / Dramatiq).
3. SSE still works — the event source becomes a Redis pub/sub subscription.

The seam is `JobRegistry` + the `job.emit(...)` contract. Nothing else
needs to change.

---

## 9. Local dev runbook

```bash
# 1. Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Python deps
pip install -r requirements.txt
pip install -r server/requirements.txt

# 3. Backend
uvicorn server.api:app --reload --port 8000

# 4. Frontend (another terminal)
cd web
npm install
npm run dev
```

Open http://localhost:5173.

Hit `/upload`, drop a video, watch both progress bars. Hit `/admin`, see it
appear with chunk count. Hit `/search`, ask something. That's the full
loop.
