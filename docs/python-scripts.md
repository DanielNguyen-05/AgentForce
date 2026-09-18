# Bản đồ các file Python

Chạy lệnh từ project root. Mỗi hàng là một entry file độc lập; không có CLI
trung gian.

## Scope mặc định

`configs/default.toml` đang giới hạn canonical pipeline vào:

```text
L21_V001 L21_V002 L21_V003
```

Với scope này, các preprocessing script không có cờ chọn video sẽ tự chạy đủ
ba video. `--all` cũng chỉ chọn đủ ba video configured. `--video-ids` dùng để
chọn subset bên trong scope, chủ yếu cho output smoke không canonical.

Batch 1 KIS là luồng riêng: `scripts/run_kis_batch.py` mặc định dùng
`configs/batch1_full.toml` và `artifacts/batch1_full/` trên toàn dataset, không
dùng hay thay đổi scope ba video của `configs/default.toml`.

## Bảng entry files

| Việc cần làm | File/lệnh trực tiếp | Output mặc định |
|---|---|---|
| Chuẩn bị PhoWhisper một lần | `python scripts/prepare_phowhisper.py` | `models/phowhisper-large-ct2/` |
| Kiểm tra môi trường | `python scripts/check_environment.py` | stdout JSON |
| Validate dataset | `python scripts/validate_dataset.py --strict` | `artifacts/manifests/dataset.json` |
| Validate artifact graph | `python scripts/validate_artifacts.py --strict` | stdout JSON |
| Export timeline | `python scripts/export_timelines.py` | `artifacts/timelines/keyframes.jsonl` |
| PhoWhisper ASR | `python scripts/transcribe_videos.py` | `artifacts/transcripts/<video>.json` |
| Smoke PhoWhisper 15 giây | `python scripts/smoke_phowhisper.py --video-id L21_V001` | `outputs/smoke/phowhisper/*.json` |
| EasyOCR | `python scripts/run_ocr.py` | `artifacts/ocr/<video>.jsonl` |
| Chuẩn hóa object | `python scripts/normalize_objects.py` | `artifacts/objects/<video>.jsonl` |
| Build temporal windows | `python scripts/build_windows.py` | `artifacts/windows/temporal_windows.jsonl` |
| Build visual indexes | `python scripts/build_visual_indexes.py --level both` | `artifacts/indexes/visual_*` |
| Build text indexes | `python scripts/build_text_indexes.py` | `artifacts/indexes/*_windows.*` |
| Xem retrieval thô | `python scripts/run_search.py --query "..."` | stdout hoặc `--output` |
| KIS | `python scripts/run_kis.py --query "..."` | stdout hoặc `--output` |
| Batch 1 KIS | `python scripts/run_kis_batch.py` | JSON: `outputs/query_batch1/kis/`; CSV: `artifacts/submissions/batch1/submission/` |
| Q&A + Gemini final | `python scripts/run_qa.py --query "..." --question "..."` | stdout hoặc `--output` |
| Q&A request đã chuẩn bị | `python scripts/verify_qa_request.py --request ...` | stdout hoặc `--output` |
| TRAKE | `python scripts/run_trake.py --query "event 1, sau đó event 2"` | stdout hoặc `--output` |
| Visualize keyframes | `python scripts/visualize_results.py --input <result.json>` | `artifacts/visualizations/*.html` |
| Đánh giá | `python scripts/evaluate_results.py --ground-truth ... --predictions ...` | stdout hoặc `--output` |
| Ghi submission | `python scripts/write_submission.py --task kis --predictions ... --output ...` | CSV |

Không có `caption_keyframes.py`. Gemini không phải preprocessing script.

## One-time PhoWhisper setup

Cài dependency:

```bash
python -m pip install -e '.[phowhisper]'
```

Convert checkpoint sang CT2 INT8:

```bash
python scripts/prepare_phowhisper.py \
  --source-model vinai/PhoWhisper-large \
  --output-dir models/phowhisper-large-ct2 \
  --quantization int8
```

Các option được hỗ trợ:

- `--revision`: pin Hugging Face revision.
- `--low-cpu-mem-usage`: giảm peak RAM, cần Accelerate.
- `--force`: thay model cũ sau khi model staging mới validate thành công.

Trên Apple Silicon dùng `cpu/int8`; CTranslate2 không dùng MPS.

## Transcript commands

Validate/reuse/migrate ba transcript hiện có:

```bash
python scripts/transcribe_videos.py --rewrite-legacy
```

Sinh file còn thiếu, giữ file hợp lệ:

```bash
python scripts/transcribe_videos.py
```

Regenerate cả scope:

```bash
python scripts/transcribe_videos.py --overwrite --fail-fast -v
```

Chọn rõ nhiều video nếu cần:

```bash
python scripts/transcribe_videos.py \
  --video-ids L21_V001 L21_V002 L21_V003
```

Hoặc lặp `--video-id`:

```bash
python scripts/transcribe_videos.py \
  --video-id L21_V001 \
  --video-id L21_V002
```

### Smoke PhoWhisper không đụng transcript production

```bash
python scripts/smoke_phowhisper.py \
  --video-id L21_V001 \
  --start-seconds 4 \
  --duration-seconds 12 \
  -v
```

Duration mặc định là 15 giây và giới hạn cứng là 60 giây. Script luôn ghi dưới
`outputs/smoke/`; mọi đường dẫn canonical như `artifacts/transcripts/...` sẽ bị từ chối. Muốn
thay một smoke file đã tồn tại phải thêm `--overwrite`. JSON kết quả có text,
segment/word timestamp tương đối và tuyệt đối, language probability và timing.

## Exact canonical rebuild order

```bash
python scripts/check_environment.py
python scripts/validate_dataset.py --strict
python scripts/export_timelines.py
python scripts/normalize_objects.py
python scripts/transcribe_videos.py --rewrite-legacy
python scripts/run_ocr.py
python scripts/build_windows.py
python scripts/build_visual_indexes.py --level both
python scripts/build_text_indexes.py --encoder sentence-transformer
python scripts/validate_artifacts.py --strict --require-ocr
```

Máy mới chưa cache SentenceTransformer cần thêm `--allow-model-download` ở lần
build text đầu; rebuild bình thường dùng local cache theo config.

Lý do windows đứng sau ba modality nguồn: nó join word-level ASR, keyframe OCR
và object labels. Mọi window index phía sau lưu hash của chính window records;
thay một modality nguồn đồng nghĩa phải rebuild windows và các index phụ thuộc.

EasyOCR là backend mặc định, languages `vi,en`, CPU. Thêm `--overwrite` nếu cần
thay các OCR file hợp lệ đã có. `build_text_indexes.py` chỉ tạo modality có text;
hashing encoder không phải production.

Text builder mặc định tạo `asr_text,ocr_text,object_labels`. `metadata_text` chỉ
được tạo khi truyền `--fields` rõ ràng vì metadata video lặp trên từng window sẽ
tạo tie rank và bias frame; metadata gốc vẫn được giữ trong manifest JSON.

`run_ocr.py --batch-size 2` bật detector batching theo đúng thứ tự keyframe;
mặc định `1` an toàn hơn và không đổi hành vi cũ. Chỉ tăng lên `4` sau khi đã
theo dõi RAM. `--show-backend-warnings` bật lại cảnh báo EasyOCR/PyTorch bị ẩn.

## Artifact validation

Read-only validation:

```bash
python scripts/validate_artifacts.py
```

Production validation:

```bash
python scripts/validate_artifacts.py \
  --strict \
  --require-ocr \
  --output artifacts/reports/artifact_validation.json
```

`--strict` yêu cầu visual keyframe/window, ASR và object text indexes. OCR thiếu
là warning trừ khi có `--require-ocr`. Caption artifact luôn là lỗi.

## Smoke commands

Canonical builders kiểm tra exact scope. Với subset, `--limit`, `--frame-limit`
hoặc hashing, dùng output riêng:

```bash
python scripts/smoke_phowhisper.py \
  --video-id L21_V001 \
  --start-seconds 4 \
  --duration-seconds 12

python scripts/run_ocr.py \
  --video-ids L21_V001 \
  --frame-limit 20 \
  --output-dir outputs/smoke/ocr

python scripts/normalize_objects.py \
  --video-ids L21_V001 \
  --frame-limit 20 \
  --output-dir outputs/smoke/objects

python scripts/build_windows.py \
  --video-ids L21_V001 \
  --output outputs/smoke/temporal_windows.jsonl

python scripts/build_visual_indexes.py \
  --video-ids L21_V001 \
  --level both \
  --windows outputs/smoke/temporal_windows.jsonl \
  --output-dir outputs/smoke/indexes

python scripts/build_text_indexes.py \
  --video-ids L21_V001 \
  --windows outputs/smoke/temporal_windows.jsonl \
  --output-dir outputs/smoke/indexes \
  --encoder hashing \
  --limit 50
```

Nếu dùng `--limit` cùng canonical output, script cố ý fail. Không copy smoke
files vào `artifacts/`.

## Query commands

Raw retrieval:

```bash
python scripts/run_search.py \
  --query "một người đang mở laptop" \
  --task kis \
  --limit 20 \
  --output outputs/search_debug.json
```

KIS:

```bash
python scripts/run_kis.py \
  --query "một người đang mở laptop" \
  --query-id KIS001 \
  --limit 100 \
  --dense-refine \
  --dense-top-n 20 \
  --output outputs/KIS001.json
```

Batch 1 KIS:

```bash
python scripts/run_kis_batch.py
```

Script mặc định dùng `configs/batch1_full.toml`, đọc
`query_batch1/*-kis.txt`, đồng thời bỏ qua mọi file query Q&A và TRAKE. Rich
JSON phục vụ kiểm tra được ghi vào `outputs/query_batch1/kis/`. CSV chính thức
được ghi vào `artifacts/submissions/batch1/submission/`; mỗi file không có
header và mỗi dòng đúng hai trường `<video_name>,<frame_id>`.

Chưa ZIP thư mục `submission/` cho đến khi đã hoàn tất đủ các task KIS, Q&A và
TRAKE.

Q&A, sau khi `.env` có `GEMINI_API_KEY`:

```bash
python scripts/run_qa.py \
  --query "người phụ nữ đang cầm một chiếc ly" \
  --question "Chiếc ly có màu gì?" \
  --query-id QA001 \
  --max-candidates 12 \
  --batch-size 4 \
  --answer-limit 100 \
  --output outputs/QA001.json
```

Đây là competition mode mặc định: Gemini xác minh từng batch frame đã được local
retrieval chọn, rồi script xếp hạng các dự đoán cho R@1/5/20/50/100. Mọi Gemini
call chỉ nhìn các frame trong batch local tương ứng; API không caption dataset.
Để smoke test rẻ bằng đúng một Gemini call, thêm `--single-answer`. Gemini model
lấy từ `[gemini].model` (hiện là `gemini-3.6-flash`). Structured VQA mặc định
dùng `thinking_level="minimal"`, output budget `2048` và trần retry `8192`.
Chỉ `finish_reason=MAX_TOKENS` làm attempt sau tăng gấp đôi budget; audit ghi
finish reason/token usage cho cả success và failure. Một `--single-answer` vẫn
có thể có nhiều API attempt nội bộ theo `max_attempts`.

TRAKE:

```bash
python scripts/run_trake.py \
  --query "một người bước vào, sau đó ngồi xuống" \
  --query-id TRAKE001 \
  --limit 100 \
  --dense-refine \
  --output outputs/TRAKE001.json
```

### Visualize Search/KIS/QA/TRAKE

Tạo gallery trực tiếp từ query, không gọi Gemini:

```bash
python scripts/visualize_results.py \
  --query "một xe đầu kéo trên đường" \
  --task kis \
  --max-results 20 \
  --output artifacts/visualizations/xe_dau_keo.html
```

Hoặc đọc JSON đã có:

```bash
python scripts/visualize_results.py \
  --input outputs/KIS001.json \
  --output artifacts/visualizations/KIS001.html \
  --open
```

HTML chứa thumbnail base64, filter theo video/text, sort theo rank/score và
evidence collapsible. Dense-refined frame không trùng keyframe sẽ được decode
từ video gốc và gắn badge `exact video frame`, không giả làm keyframe gần nhất.

## Call graph

```text
scripts/run_<task>.py
  → scripts/_bootstrap.py
       ├─ add src/ vào sys.path
       └─ load project-root .env
  → src/agentforce/config.py
  → src/agentforce/runtime.py
  → src/agentforce/engine.py + src/agentforce/tasks/<task>.py
  → src/agentforce/retrieval/ + artifacts/indexes/
```

Riêng Q&A, sau call graph local ở trên mới đi tiếp vào
`src/agentforce/gemini/`. `scripts/_bootstrap.py` không phân luồng command và
không che file đang chạy.
