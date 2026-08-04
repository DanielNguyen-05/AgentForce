# Data contracts

## Experiment scope

Canonical artifacts thuộc đúng tập video trong `[scope].video_ids`. Với config
hiện tại:

```json
["L21_V001", "L21_V002", "L21_V003"]
```

Artifact manifest phải mang cùng `video_ids` và `scope_hash`. Dữ liệu từ video
ngoài scope hoặc thiếu một video trong scope làm validation/runtime fail. Vì
vậy `--all` khi scope đã cấu hình chỉ có nghĩa là cả ba video này.

## Canonical identifiers

```text
video_id:      L21_V001
keyframe_uid:  L21_V001_K000115
window_id:     L21_V001_W000022
```

`keyframe_number` là số thứ tự keyframe do BTC cung cấp và thường bắt đầu từ 1.
`visual_embedding_row` bắt đầu từ 0. `frame_idx` mới là frame ID dùng cho output
và submission.

## Transcript PhoWhisper

Mỗi video có một file:

```text
outputs/transcripts/<video_id>.json
```

Canonical root chứa:

```text
video_id
language
language_probability
duration_seconds
model_name
source_path
created_at
segments[]
```

Mỗi segment chứa `segment_id`, `start`, `end`, `text`, confidence diagnostics
và `words[]`. Mỗi word chứa `text`, `start`, `end`, `confidence`.

Reader kiểm tra kiểu dữ liệu, probability range, ID trùng, timestamp hữu hạn,
thứ tự segment/word và word có nằm trong segment hay không. Faster-Whisper đôi
khi trả một empty word token có timestamp; token này được chấp nhận để đọc lại
output thật nhưng bị bỏ qua khi ghép text vào window.

Ba transcript thử ban đầu có schema legacy:

```text
source_video
pipeline_config
audio_info
segments[].time_range
segments[].audio_metadata
```

`read_transcript()` hiểu rõ cả hai schema. Lệnh:

```bash
python scripts/transcribe_videos.py --rewrite-legacy
```

validate rồi atomically rewrite legacy JSON sang canonical JSON; file canonical
hợp lệ được reuse. Không dùng `audio_to_json.ipynb` làm data contract mới.

## Word-to-window alignment

Window dài 10 giây, stride 5 giây theo config. Khi segment có word timestamps,
word được đưa vào window dựa trên midpoint timestamp. Điều này tránh copy toàn
bộ segment PhoWhisper dài khoảng 30 giây vào nhiều window.

Nếu segment không có usable word timestamp, hệ thống fallback sang tỷ lệ overlap
giữa segment và window. `asr_segment_ids` vẫn giữ provenance của segment đã đóng
góp text.

## Temporal window

Một record trong `artifacts/windows/temporal_windows.jsonl` giữ:

- `window_id`, `video_id`, `window_number`;
- `start_time`, `end_time`, `start_frame`, `end_frame`;
- representative keyframe/frame/timestamp;
- danh sách `keyframe_uids` và `asr_segment_ids`;
- `asr_text`, `ocr_text`, `object_labels`;
- `metadata_text` là field dự phòng/opt-in, không được materialize trong baseline.

Không có `caption_text` trong pipeline hiện tại. Caption artifacts dưới
`artifacts/` bị validator báo lỗi.

## Vector persistence

Ví dụ một index:

```text
artifacts/indexes/asr_windows.npy
artifacts/indexes/asr_windows.jsonl
artifacts/indexes/asr_windows.manifest.json
```

Hàng `i` trong NPY phải tương ứng chính xác dòng `i` trong JSONL. Manifest khai
báo tối thiểu:

- tên index;
- vector/metadata path;
- row count, dimension, dtype và normalized state;
- encoder class/model;
- `video_ids` và `scope_hash`;
- `window_records_hash` đối với window index;
- builder version.

Runtime chỉ nhận text index được build bằng `SentenceTransformerEncoder` đúng
model trong config. Hashing encoder là smoke-only và không được đặt vào
canonical `artifacts/indexes/`.

## Structured persistence

| Nội dung | Định dạng |
|---|---|
| Dataset, window và index manifests | JSON |
| PhoWhisper transcript một video | JSON |
| Timeline, OCR, object, windows, row metadata | JSONL |
| Embedding matrices | NPY float16/float32 |
| Gemini cache/audit và task results | JSON/JSONL |
| Submission | CSV theo task |

Embedding không được serialize thành list float trong JSON. Text gốc, ID,
timestamp, bbox và provenance phải còn ở JSON/JSONL để debug và rebuild.

## Canonical versus smoke outputs

Các path mặc định dưới `artifacts/` là canonical. Builder chỉ ghi vào đó khi
selection khớp toàn bộ configured scope và không có `--limit`/`--frame-limit`.

Smoke output phải nằm ở path riêng, ví dụ:

```text
outputs/smoke/transcripts/
outputs/smoke/ocr/
outputs/smoke/objects/
outputs/smoke/temporal_windows.jsonl
outputs/smoke/indexes/
```

Không copy file smoke vào canonical path. Manifest/hash là hàng rào phát hiện,
không phải cách hợp thức hóa việc trộn dữ liệu.

## Gemini trust boundary

Gemini chỉ được gọi ở Q&A final stage sau local retrieval. Mỗi request chỉ chứa
một tập con các candidate trong `[gemini].max_candidates` đã được local retrieval
chọn và local manifest resolve; không request nào được nhìn frame ngoài tập đó.

Gemini trả `supporting_candidate_id`; hệ thống không tin `video_id` hoặc
`frame_idx` tự sinh. Verifier map candidate ID về registry local, validate JSON
schema, cache response và lưu audit diagnostics.

API key chỉ được đọc từ biến `GEMINI_API_KEY` do project `.env` cung cấp trong
workflow chuẩn. Model Gemini là config data ở `[gemini].model`, không nằm trong
artifact và không hardcode trong task solver.

## Artifact validation contract

```bash
python scripts/validate_artifacts.py --strict --require-ocr
```

Strict mode yêu cầu:

- manifest/timeline/transcript/object/window hợp lệ cho đủ ba video;
- `visual_keyframes`, `visual_windows`, `asr_windows`, `objects_windows`;
- vectors và metadata có cùng row count/dimension/dtype;
- scope/hash/encoder không stale;
- không có caption artifact.

`--require-ocr` nâng OCR thiếu/partial từ warning thành error. OCR index chỉ được
sinh nếu windows có OCR text thực tế.
