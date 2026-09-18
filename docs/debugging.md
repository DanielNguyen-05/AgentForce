# Debug bằng các file Python trực tiếp

## Biết file nào đang chạy

Mỗi thao tác có đúng một entry file. File đầu tiên đang chạy chính là file sau
`python`:

```bash
python scripts/run_kis.py --query "một người mở laptop"
```

Mỗi script còn in:

```text
[RUN] /absolute/path/to/AgentForce/scripts/run_kis.py
```

Không có dispatcher, command router hoặc CLI ẩn ở giữa.

## Kiểm tra tổng quát trước

```bash
python scripts/check_environment.py \
  --output outputs/environment_report.json
```

Report cho biết:

- config và scope đang dùng;
- dataset component paths;
- `ffmpeg`, `ffprobe`, `tesseract`;
- dependency Python;
- PhoWhisper model path có tồn tại không;
- trạng thái/schema của ba transcript;
- `.env` đã cung cấp Gemini key chưa.

Scope mặc định phải là `L21_V001`, `L21_V002`, `L21_V003`. Nếu report hiện
scope khác, dừng lại và kiểm tra `configs/default.toml` trước khi rebuild.

## PhoWhisper model chưa tồn tại

Triệu chứng:

```text
Local PhoWhisper model not found ... Run python scripts/prepare_phowhisper.py first
```

Chạy:

```bash
python scripts/prepare_phowhisper.py \
  --source-model vinai/PhoWhisper-large \
  --output-dir models/phowhisper-large-ct2 \
  --quantization int8
```

Trên Apple Silicon phải dùng `device="cpu"`, `compute_type="int8"` trong
config; CTranslate2 không dùng MPS. Nếu conversion thiếu RAM, thêm
`--low-cpu-mem-usage`. Nếu một model cũ hợp lệ đã tồn tại, converter skip; chỉ
dùng `--force` khi muốn thay thế có chủ đích.

Converter làm việc trong hidden staging directory và chỉ publish sau validation.
Không trỏ `--output-dir` vào project root, home hoặc một thư mục dữ liệu.

Để kiểm tra model thật mà không đụng transcript production, chạy một clip 12
giây. Kết quả và timing chỉ được ghi dưới `outputs/smoke/`:

```bash
python scripts/smoke_phowhisper.py \
  --video-id L21_V001 \
  --start-seconds 4 \
  --duration-seconds 12 \
  -v
```

Nếu output smoke đã tồn tại, chọn khoảng thời gian khác hoặc thêm `--overwrite`.
Script từ chối mọi `--output` nằm ngoài `outputs/smoke/`, bao gồm
`artifacts/transcripts/`.

## Transcript cũ hoặc không hợp lệ

Ba file production nằm ở `artifacts/transcripts/`. Kiểm tra read-only:

```bash
python scripts/check_environment.py
python scripts/validate_artifacts.py
```

Để migrate schema PhoWhisper legacy mà không transcribe lại:

```bash
python scripts/transcribe_videos.py --rewrite-legacy
```

Script validate từng file và atomically rewrite legacy JSON. File canonical hợp
lệ được skip. Nếu báo `Existing transcript is invalid`, kiểm tra đúng file/video
ID; chỉ dùng lệnh sau khi thực sự muốn trả chi phí thời gian ASR:

```bash
python scripts/transcribe_videos.py --overwrite --fail-fast -v
```

## ASR text trong window quá dài hoặc lặp

PhoWhisper có thể trả segment khoảng 30 giây trong khi window chỉ dài 10 giây.
Code mới dùng word timestamps trước, segment overlap chỉ là fallback. Kiểm tra
transcript có `segments[].words` và config có:

```toml
word_timestamps = true
```

Sau khi thay transcript, luôn rebuild theo thứ tự:

```bash
python scripts/build_windows.py
python scripts/build_visual_indexes.py --level window
python scripts/build_text_indexes.py --encoder sentence-transformer
python scripts/validate_artifacts.py --strict
```

## EasyOCR

Config mặc định dùng EasyOCR `vi,en` trên CPU. Cài dependency:

```bash
python -m pip install -e '.[easyocr]'
```

Lần đầu chạy có thể cần tải model OCR:

```bash
python scripts/run_ocr.py -v
```

Nếu OCR file tồn tại nhưng validator báo thiếu row/partial, rebuild:

```bash
python scripts/run_ocr.py --overwrite -v
python scripts/build_windows.py
python scripts/build_visual_indexes.py --level window
python scripts/build_text_indexes.py --encoder sentence-transformer
```

Nếu semantic model chưa có trong Hugging Face cache, thêm
`--allow-model-download` cho lần build đầu tiên. Không thêm cờ ở các lần sau;
builder sẽ dùng cache local và tránh lỗi mạng/HTTP HEAD không cần thiết.

`--gpu` chỉ dùng khi môi trường EasyOCR thực sự có GPU phù hợp; không thêm cờ đó
cho cấu hình Apple CPU mặc định.

Trên ba video hiện tại, mọi keyframe đều là `1280×720`, nên có thể benchmark
`--batch-size 2` rồi `4`. Mặc định vẫn là `1`; batching chỉ tăng tốc một phần
pipeline EasyOCR và tốn thêm RAM. Script ẩn riêng cảnh báo `pin_memory` vô hại,
lặp lại của PyTorch. Dùng `--show-backend-warnings` nếu cần xem cảnh báo đó.

## Dataset hoặc artifact scope sai

Validate dataset:

```bash
python scripts/validate_dataset.py --strict
```

Mở `artifacts/manifests/dataset.json` và xác nhận chỉ có ba video configured.
Không dùng một manifest 873 video cùng index ba video.

Validate toàn bộ graph:

```bash
python scripts/validate_artifacts.py \
  --strict \
  --require-ocr \
  --output artifacts/reports/artifact_validation.json
```

Strict mode trả exit code `2` nếu có error. Bỏ `--require-ocr` khi đang thử
baseline không OCR; OCR thiếu lúc đó là warning.

Các lỗi thường gặp:

- `*_scope_mismatch`: artifact được build cho tập video khác config.
- `*_row_count_mismatch`: JSONL bị partial hoặc không khớp keyframe/vector.
- `index_window_hash_mismatch`: windows đổi nhưng index chưa rebuild.
- `encoder_model` hoặc `expected_encoder` mismatch: config/model khác lúc build.
- `caption_artifacts_forbidden`: còn artifact caption của pipeline cũ.

Không sửa manifest/hash bằng tay. Rebuild theo dependency order.

## Search báo stale hoặc thiếu index

Chuỗi rebuild đầy đủ:

```bash
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

Nếu chỉ ASR/OCR/object thay đổi, có thể bắt đầu lại từ `build_windows.py`, nhưng
cả visual window index và text indexes phía sau đều phải rebuild.

Runtime production từ chối hashing text index. Nếu thấy lỗi `not built with a
production semantic encoder`, chạy lại `build_text_indexes.py` với
`--encoder sentence-transformer` vào canonical index directory.

## Smoke safeguard báo lỗi

Các lỗi như:

```text
--frame-limit requires an explicit non-canonical --output-dir
--limit requires an explicit non-canonical --output
Canonical ... does not match configured scope
```

là hành vi bảo vệ dữ liệu. Không bỏ check và không dùng path canonical. Ví dụ:

```bash
python scripts/run_ocr.py \
  --video-ids L21_V001 \
  --frame-limit 20 \
  --output-dir outputs/smoke/ocr

python scripts/build_text_indexes.py \
  --video-ids L21_V001 \
  --windows outputs/smoke/temporal_windows.jsonl \
  --output-dir outputs/smoke/indexes \
  --encoder hashing \
  --limit 50
```

Không di chuyển các file này sang `artifacts/indexes/`.

## Gemini Q&A lỗi

Gemini chỉ chạy sau local retrieval. Nếu KIS/search local đã lỗi thì sửa index
trước, không debug API trước.

Checklist:

1. `GEMINI_API_KEY` chỉ được điền trong `.env` ở project root.
2. `python scripts/check_environment.py` phải báo `gemini_key_configured: true`.
3. `[gemini].model` trong `configs/default.toml` hiện là `gemini-3.6-flash`;
   đổi nếu tài khoản không hỗ trợ model đó.
4. Candidate image path phải đọc được.
5. Kiểm tra quota, rate limit và timeout.
6. Xem cache/audit dưới `artifacts/gemini/`.

Config mặc định dành cho câu trả lời VQA JSON ngắn:

```toml
thinking_level = "minimal"
max_output_tokens = 2048
max_retry_output_tokens = 8192
```

Nếu SDK báo `finish_reason = MAX_TOKENS`, client xem response JSON là bị cắt và
tăng gấp đôi budget cho attempt kế tiếp, tối đa `8192`. Client không tăng budget
cho lỗi quota, timeout, schema hoặc JSON hỏng mà không có `MAX_TOKENS`; các lỗi
đó vẫn dùng retry/backoff bình thường khi phù hợp.

Audit JSONL dưới `artifacts/gemini/` ghi cả request thành công lẫn thất bại. Khi
debug, đối chiếu `finish_reason`, budget của attempt, token output/total/thoughts
(nếu SDK trả về), `status` và `error`. Failed response không được đưa vào cache.

Chạy lại đúng request đã lỗi trong log:

```bash
python scripts/run_qa.py \
  --query "một xe đầu kéo lưu thông trên đường tránh phía tây thành phố Buôn Ma Thuột" \
  --question "Loại xe nào xuất hiện trong cảnh?" \
  --query-id QA001 \
  --max-candidates 12 \
  --single-answer \
  --output outputs/QA001.json
```

`--single-answer` là một lượt xác minh QA, nhưng client vẫn có thể retry cùng
request theo `max_attempts`. Gemini ở đây chỉ trả lời sau khi local retrieval đã
chọn frame; không dùng Gemini để captioning hay build index.

Test một request đã chuẩn bị:

```bash
python scripts/verify_qa_request.py \
  --request examples/qa_request.json \
  --output outputs/qa_request_result.json
```

Client retry và validate structured JSON. Verifier từ chối
`supporting_candidate_id` không nằm trong batch. Không có caption API để debug.

## TRAKE không có path hợp lệ

- Query cần ít nhất hai event, phân cách bằng `sau đó`, `rồi`, `->` hoặc numbered
  lines.
- Kiểm tra từng event bằng
  `python scripts/run_search.py --query "mô tả event" --task trake`.
- Điều chỉnh `retrieval.top_keyframes`, `retrieval.top_videos` hoặc gap trong
  config.
- Thêm `--dense-refine` khi cần exact frame; solver enforce temporal order lại
  sau refinement.
- Kiểm tra FPS và map-keyframes nếu frame sai.

## Theo dõi call stack

```bash
python -m pdb scripts/run_kis.py --query "một người mở laptop"
```

Đặt breakpoint đầu trong `scripts/run_kis.py`, rồi step vào:

```text
src/agentforce/runtime.py
src/agentforce/tasks/kis.py
src/agentforce/retrieval/
src/agentforce/indexing/numpy_index.py
```

Xem import timing:

```bash
python -X importtime scripts/run_kis.py --query "một người mở laptop"
```

Khi so sánh hai run, giữ lại config, index manifest, source hashes, model version
và result JSON. Không so score giữa hai artifact graph có scope/hash khác nhau.
