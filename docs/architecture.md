# Kiến trúc

## Offline

```text
dataset/
  → DatasetManifest
  → canonical keyframe timelines
  ├─ supplied CLIP vectors
  ├─ Faster-Whisper segments + word timestamps
  ├─ OCR text + boxes
  ├─ normalized OpenImages objects
  └─ selective captions
       ↓
  overlapping temporal windows
       ↓
  independent indexes by modality
```

Mỗi modality có vector matrix và row-aligned JSONL riêng. `window_id` là khóa fusion chung. Metadata chính xác như `video_id`, `frame_idx`, `pts_time`, bbox và text gốc không nằm trong embedding.

## Online

```text
query
  → heuristic parser / replaceable LLM parser
  → safe variants
  → search each modality
  → weighted RRF
  → temporal diversification
  ├─ KIS: frame ranking + optional dense refinement
  ├─ Q&A: local frames → Gemini verifier
  └─ TRAKE: per-event retrieval → ordered Viterbi → dense refinement
```

## Backend boundaries

- `TextEncoder`: OpenCLIP, SentenceTransformer hoặc fake encoder.
- `NumpyIndex`: exact, memory-mapped, dễ kiểm tra.
- `FaissIndex`: optional production backend.
- `DenseFrameSource`: OpenCV decoder.
- `FrameScorer`: OpenCLIP hoặc model khác.
- `GeminiTransport`: mockable, không làm tests phụ thuộc network.

## Artifact invalidation

Temporal windows có canonical record hash. Window indexes ghi lại hash nguồn, model và builder version. Search loader fail-fast nếu index cũ hơn windows hoặc encoder config không khớp.

