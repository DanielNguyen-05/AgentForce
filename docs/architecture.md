# Kiến trúc

## Scope là một phần của kiến trúc

Pipeline không coi dataset root là experiment scope. Dataset có thể chứa nhiều
video, nhưng `[scope].video_ids` trong config xác định chính xác video nào được
phép đi vào canonical artifacts. Cấu hình mặc định hiện là:

```text
L21_V001, L21_V002, L21_V003
```

Manifest, timeline, temporal windows và mọi index đều ghi `video_ids` cùng
`scope_hash`. Builder từ chối ghi subset vào canonical path; runtime từ chối
index có scope khác config.

## Offline pipeline

```text
dataset/ (read-only)
  │
  ├─ videos/L21/*.mp4
  ├─ keyframes/
  ├─ map-keyframes/
  ├─ clip-features-32/
  └─ objects/ do BTC cung cấp
  │
  ▼
scoped DatasetManifest + canonical keyframe timeline
  │
  ├─ supplied CLIP vectors
  ├─ PhoWhisper CT2 transcript ở artifacts/transcripts/
  │    └─ segment + word timestamps
  ├─ EasyOCR vi,en trên keyframes
  └─ normalized object detections
  │
  ▼
overlapping temporal windows (10 s, stride 5 s)
  │
  ├─ visual keyframe/window index
  ├─ ASR text index
  ├─ OCR text index nếu có OCR text
  └─ object-label text index
```

Metadata video vẫn nằm trong manifest/JSON. Builder hỗ trợ `metadata_text` khi
truyền `--fields` rõ ràng, nhưng baseline không bật: cùng một title/description
lặp ở hàng trăm overlapping windows tạo vector trùng, tie rank tùy ý và có thể
boost sai frame. Một video-level metadata prior đúng nghĩa là hướng mở rộng sau.

Không có scene/keyframe extractor hoặc object detector inference trong baseline
này: project tận dụng keyframe, mapping, CLIP feature và object sidecar do dataset
cung cấp.

Không có caption pipeline. Gemini không chạy trong preprocessing và
`validate_artifacts.py` xem caption artifact còn sót lại là lỗi.

## PhoWhisper boundary

`scripts/prepare_phowhisper.py` là bước chuẩn bị model một lần:

```text
vinai/PhoWhisper-large
  → CTranslate2 converter trong staging directory
  → validate model.bin/config/tokenizer/preprocessor
  → models/phowhisper-large-ct2/
```

`scripts/transcribe_videos.py` load model CT2 một lần cho batch video rồi ghi một
JSON atomic cho mỗi video trong `artifacts/transcripts/`. File legacy được
parser riêng validate và có thể migrate sang canonical schema mà không
chạy ASR lại.

PhoWhisper thường tạo segment dài hơn retrieval window. Vì vậy alignment ưu
tiên word timestamp: word được gán theo midpoint vào window; segment-overlap chỉ
là fallback khi transcript không có usable word timestamp.

Trên Apple Silicon, CTranslate2 dùng CPU INT8 và không dùng MPS. Điều này được
phản ánh trực tiếp trong `[asr]` của config.

## Persistence boundary

Mỗi vector modality gồm ba thành phần:

```text
<name>.npy
<name>.jsonl
<name>.manifest.json
```

Hàng `i` trong `.npy` luôn tương ứng dòng `i` trong `.jsonl`. Manifest lưu row
count, dimension, dtype, encoder, scope và source/window hash. Metadata chính
xác như `video_id`, `frame_idx`, `pts_time`, bbox và text gốc không được nhét
vào embedding.

`artifacts/` chứa toàn bộ dữ liệu có thể rebuild, gồm transcript tại
`artifacts/transcripts/`. Window builder là điểm join các modality đó.

## Online pipeline

```text
query
  → heuristic parser
  → modality-specific query encoding
  → exact NumPy search từng index
  → weighted Reciprocal Rank Fusion
  → temporal/video diversification
  │
  ├─ KIS
  │    └─ top frames → optional OpenCV/OpenCLIP dense refinement
  │
  ├─ Q&A
  │    └─ local top frames → trusted frame registry → Gemini ranked batch answers
  │
  └─ TRAKE
       └─ per-event retrieval → group by video → ordered dynamic programming
            → optional dense refinement → enforce exact order again
```

`retrieval.visual_level` chọn `visual_keyframes` hoặc `visual_windows`; config
mặc định dùng keyframe để KIS/Q&A có frame chi tiết ngay từ coarse retrieval.

Để RRF thật sự kết hợp được hai mức biểu diễn, runtime chuẩn hóa candidate ID
theo thứ tự `representative_keyframe_uid` → `keyframe_uid` → `vector_id`.
Nhờ vậy visual keyframe và ASR/OCR/object window đại diện cho cùng khoảnh khắc
được cộng evidence vào một hit, trong khi `visual_vector_id`, `<modality>_vector_id`
và `<modality>_window_id` vẫn được giữ để debug nguồn gốc.

`scripts/visualize_results.py` là nhánh quan sát read-only: nó nhận raw search
hoặc task JSON, resolve ảnh lại qua manifest/timeline canonical và tạo standalone
HTML dưới `artifacts/visualizations/`. Query mode gọi local retrieval trực tiếp;
không đi vào Gemini. Dense frame không trùng keyframe được decode từ video gốc
và được ghi rõ là `exact_video_frame`.

## Gemini trust boundary

Gemini chỉ tồn tại ở nhánh Q&A cuối:

1. Local indexes retrieve và rank candidate.
2. Local manifest resolve candidate sang ảnh/keyframe tin cậy.
3. Mỗi Gemini call nhận một batch nhỏ frame cùng câu hỏi; các batch xếp hạng là
   competition mode mặc định, còn `--single-answer` dành cho smoke test một call.
4. Structured response phải trả `supporting_candidate_id` thuộc request.
5. `video_id` và `frame_idx` cuối được resolve lại từ registry local.
6. Request/response được cache và audit trong `artifacts/gemini/`.

API key được nạp từ `.env`; model lấy từ `[gemini].model` (hiện là
`gemini-3.6-flash`) và có thể đổi bằng config. Gemini không được dùng để
caption, dịch toàn dataset hay tạo index.

## Artifact invalidation và smoke safety

Thứ tự dependency bắt buộc:

```text
manifest/timeline
  → ASR + OCR + object
  → temporal windows
  → visual/text window indexes
  → artifact validation
```

Temporal windows có canonical record hash. Mọi window index lưu lại hash này;
runtime fail-fast nếu windows đã thay đổi mà index chưa rebuild. Encoder và
scope cũng được kiểm tra tương tự.

Các tùy chọn smoke như `--limit`, `--frame-limit`, subset video và hashing
encoder phải đi với một output không canonical, thường dưới `outputs/smoke/`.
Đây là invariant của builder, không chỉ là quy ước tài liệu.

## Backend boundaries

- `TextEncoder`: OpenCLIP cho visual query; SentenceTransformer cho text index;
  hashing chỉ dùng smoke test.
- `NumpyIndex`: exact normalized inner-product, memory-mapped và dễ debug.
- `FaissIndex`: implementation tùy chọn, chưa phải runtime mặc định.
- `DenseFrameSource`: OpenCV decode quanh timestamp.
- `FrameScorer`: OpenCLIP chấm frame đã decode.
- `GeminiTransport`: mockable để tests không phụ thuộc network.
