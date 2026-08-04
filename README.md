# AgentForce AIC 2026

Pipeline truy xuất video đa phương thức cho ba task:

- KIS: tìm đúng video và frame từ mô tả văn bản.
- Q&A: retrieval local chọn các frame tốt nhất, sau đó Gemini trả lời câu hỏi.
- TRAKE: tìm nhiều event và căn chỉnh chúng theo đúng thứ tự thời gian.

Project chỉ dùng các file Python trực tiếp, không có lệnh `agentforce`, command
router hay `cli.py`:

```bash
python scripts/run_kis.py --query "một người đang mở laptop"
```

Mỗi entry file in `[RUN] /đường/dẫn/tuyệt/đối/tới/file.py` khi bắt đầu, nên luôn
biết chính xác file nào đang chạy.

## Phạm vi thử nghiệm hiện tại

File `configs/default.toml` đang khóa toàn bộ canonical pipeline vào đúng ba
video:

```toml
[scope]
video_ids = ["L21_V001", "L21_V002", "L21_V003"]
```

Dataset gốc vẫn có thể chứa hàng trăm video, nhưng manifest, timeline, object,
window và index mặc định chỉ được phép chứa ba ID trên. Khi scope đã được cấu
hình, chạy script không truyền lựa chọn video sẽ tự dùng đủ ba video; `--all`
cũng chỉ có nghĩa là toàn bộ configured scope, không phải toàn bộ dataset 873
video.

Muốn đổi experiment, sửa `[scope].video_ids` rồi rebuild toàn bộ canonical
artifacts. Runtime và `validate_artifacts.py` kiểm tra scope/hash để không vô
tình trộn index của hai experiment.

## Cấu trúc chính

```text
AgentForce/
├── configs/default.toml            # Scope và toàn bộ tham số runtime
├── scripts/                        # Các entry file chạy bằng python
│   ├── prepare_phowhisper.py       # Download/convert PhoWhisper một lần
│   ├── check_environment.py
│   ├── validate_dataset.py
│   ├── validate_artifacts.py
│   ├── export_timelines.py
│   ├── transcribe_videos.py
│   ├── smoke_phowhisper.py         # ASR thật trên clip ngắn, chỉ ghi outputs/smoke
│   ├── run_ocr.py
│   ├── normalize_objects.py
│   ├── build_windows.py
│   ├── build_visual_indexes.py
│   ├── build_text_indexes.py
│   ├── run_search.py
│   ├── run_kis.py
│   ├── run_qa.py
│   ├── run_trake.py
│   ├── evaluate_results.py
│   └── write_submission.py
├── src/agentforce/                 # Logic Python nội bộ
├── dataset/                        # Dữ liệu BTC, chỉ đọc
├── outputs/                        # Kết quả task và smoke test
├── artifacts/                      # Transcript, manifest, JSONL, index và cache
├── models/                         # PhoWhisper CT2 local, không commit Git
├── tests/
└── docs/
```

## Cài đặt

Khuyến nghị Python 3.11 hoặc 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Cài đúng các thành phần đang dùng cho experiment ba video:

```bash
python -m pip install -e '.[phowhisper,easyocr,vision,semantic,video,gemini]'
```

Hoặc cài toàn bộ:

```bash
python -m pip install -e '.[all,dev]'
```

Kiểm tra dependency, `ffmpeg`, model local, transcript và API key:

```bash
python scripts/check_environment.py
```

### Apple Silicon

CTranslate2 trên máy Apple ARM64 chạy PhoWhisper bằng CPU, không dùng MPS.
Cấu hình mặc định vì vậy là:

```toml
[asr]
device = "cpu"
compute_type = "int8"
audio_sample_rate = 16000
audio_channels = 1
normalize_lufs = true
integrated_loudness = -16.0
hotwords = []
```

Không đổi sang `float16` trên CPU Apple. Nếu chuyển sang máy NVIDIA CUDA, hãy
đổi đồng bộ model conversion, `device` và `compute_type` sau khi kiểm tra phần
cứng.

Có thể thêm tên riêng/địa danh thường bị nhận sai vào `hotwords`, ví dụ
`["Buôn Ma Thuột", "Cửu Long"]`. Pipeline truyền danh sách này vào decoder
PhoWhisper; để rỗng nếu chưa có danh sách miền đáng tin cậy.

## Gemini API qua `.env`

Tạo file secret một lần:

```bash
cp .env.example .env
```

Sau đó chỉ điền key vào `.env`:

```dotenv
GEMINI_API_KEY=your_real_key_here
```

Mọi file trong `scripts/` tự nạp `.env` ở project root, kể cả khi chạy từ
working directory khác. Không ghi API key vào TOML, source code hoặc command.
Biến shell đã export, nếu có, được ưu tiên và không bị `.env` ghi đè.

Gemini chỉ được gọi ở bước trả lời cuối của Q&A, sau khi retrieval local đã chọn
frame. Gemini không caption keyframe, không tạo text cho dataset và không xây
index. Config hiện đặt `[gemini].model = "gemini-3.6-flash"`; có thể đổi giá trị
này theo model mà tài khoản API hỗ trợ. `run_qa.py` không hardcode model.

## Chuẩn bị PhoWhisper

### 1. Convert model một lần

Lệnh mặc định tải checkpoint chính thức rồi convert sang CTranslate2 INT8:

```bash
python scripts/prepare_phowhisper.py \
  --source-model vinai/PhoWhisper-large \
  --output-dir models/phowhisper-large-ct2 \
  --quantization int8
```

Converter ghi vào thư mục staging, kiểm tra các file cần thiết rồi mới thay thế
output. Model hợp lệ đã tồn tại sẽ được reuse. Dùng `--force` chỉ khi muốn
convert lại; model cũ vẫn được giữ cho tới khi model mới vượt qua validation.
Nếu RAM hạn chế trong lúc convert, có thể thêm `--low-cpu-mem-usage`.

### 2. Kiểm tra và reuse ba transcript hiện có

Ba file mong đợi nằm tại:

```text
artifacts/transcripts/L21_V001.json
artifacts/transcripts/L21_V002.json
artifacts/transcripts/L21_V003.json
```

Kiểm tra read-only bằng:

```bash
python scripts/check_environment.py
python scripts/validate_artifacts.py
```

Nếu đây là JSON PhoWhisper theo schema notebook cũ, lệnh sau validate rồi rewrite
atomic sang canonical schema. Transcript canonical hợp lệ chỉ được kiểm tra và
skip, không chạy ASR lại:

```bash
python scripts/transcribe_videos.py --rewrite-legacy
```

Để sinh những transcript còn thiếu và reuse file hợp lệ đã có:

```bash
python scripts/transcribe_videos.py
```

Chỉ dùng `--overwrite` khi thực sự muốn chạy lại cả ba video:

```bash
python scripts/transcribe_videos.py --overwrite
```

Output giữ timestamp ở mức segment và word. Window builder ưu tiên word timestamp
để không sao chép nguyên một segment PhoWhisper dài khoảng 30 giây vào nhiều
window 10 giây.

### Smoke PhoWhisper thật trên 15 giây

Trước khi chạy ASR cho cả video, kiểm tra model local bằng một clip ngắn:

```bash
python scripts/smoke_phowhisper.py \
  --video-id L21_V001 \
  --start-seconds 4 \
  --duration-seconds 12 \
  -v
```

Script chỉ xử lý một video, giới hạn tối đa 60 giây và bắt buộc output nằm dưới
`outputs/smoke/`. Output mặc định của lệnh trên là
`outputs/smoke/phowhisper/L21_V001_start4_duration12.json`; các file
`artifacts/transcripts/*.json` không được đọc, rewrite hoặc overwrite. JSON smoke
ghi cả timestamp tương đối trong clip và timestamp tuyệt đối trong video, cùng
thời gian extract audio, load model + inference và tổng thời gian.

## Rebuild canonical artifacts theo đúng thứ tự

Chạy các lệnh sau từ project root. Vì `[scope]` đã có ba video, không cần truyền
`--video-ids` hoặc `--all`.

### 1. Manifest và canonical timeline

```bash
python scripts/validate_dataset.py --strict
python scripts/export_timelines.py
```

### 2. Các modality nguồn

Object detector không được chạy lại; script chỉ chuẩn hóa detection do dataset
cung cấp:

```bash
python scripts/normalize_objects.py
```

ASR reuse transcript hợp lệ và chỉ chạy model cho file còn thiếu:

```bash
python scripts/transcribe_videos.py --rewrite-legacy
```

OCR mặc định dùng EasyOCR với `vi,en` trên CPU. Lần đầu EasyOCR có thể tải model:

```bash
python scripts/run_ocr.py
```

Nếu cần tạo lại OCR đã tồn tại:

```bash
python scripts/run_ocr.py --overwrite
```

`--batch-size` mặc định là `1` để ổn định. Vì 855 keyframe hiện tại đều có kích
thước `1280×720`, có thể thử `--batch-size 2` rồi mới tăng lên `4` nếu máy đủ
RAM. EasyOCR chỉ batch chủ yếu phần phát hiện chữ nên tốc độ CPU có thể không
tăng tuyến tính. Cảnh báo `pin_memory` lặp lại của PyTorch được ẩn có chọn lọc;
thêm `--show-backend-warnings` khi cần debug backend.

### 3. Temporal windows

Phải build windows sau khi ASR, OCR và object đã hoàn tất:

```bash
python scripts/build_windows.py
```

Window mặc định dài 10 giây, stride 5 giây. `build_windows.py` đọc transcript,
OCR và object từ `artifacts/`.

### 4. Visual indexes

```bash
python scripts/build_visual_indexes.py --level both
```

Lệnh tạo cả `visual_keyframes` và `visual_windows` từ CLIP feature do dataset
cung cấp.

### 5. Text indexes production

```bash
python scripts/build_text_indexes.py --encoder sentence-transformer
```

Config mặc định ưu tiên model đã cache để các lần rebuild không gửi HTTP HEAD.
Trên máy mới chưa có model, chạy lần đầu với
`--allow-model-download`; các lần sau bỏ cờ này để chạy offline ổn định.

Mặc định tạo ba text index `asr`, `ocr`, `objects`. `metadata_text` vẫn là field
opt-in qua `--fields`, nhưng không bật mặc định: title/description giống hệt nhau
ở mọi window của một video sẽ tạo nhiều vector trùng và làm thứ hạng frame bị
lệch. `hashing` chỉ dành cho smoke test; runtime production chủ động từ chối
index hashing.

### 6. Validate toàn bộ artifact graph

```bash
python scripts/validate_artifacts.py \
  --strict \
  --require-ocr \
  --output artifacts/reports/artifact_validation.json
```

Validator kiểm tra scope ba video, row count, JSON/JSONL, vector shape/dtype,
window hash, encoder, transcript và artifact caption bị cấm. Exit code `2` trong
strict mode nghĩa là còn lỗi cần xử lý. Có thể bỏ `--require-ocr` nếu muốn OCR
chỉ là modality tùy chọn; khi đó OCR thiếu được báo warning.

Tóm tắt dependency của rebuild:

```text
validate_dataset
  → export_timelines
  → normalize_objects + transcribe_videos + run_ocr
  → build_windows
  → build_visual_indexes
  → build_text_indexes
  → validate_artifacts
```

Không có bước captioning trong chuỗi này.

Khi search, visual keyframe và các text window được nối bằng
`representative_keyframe_uid`; vì vậy RRF có thể cộng evidence CLIP + ASR + OCR +
object cho cùng một khoảnh khắc mà vẫn giữ ID window/vector gốc trong output debug.

## Smoke test không làm hỏng canonical artifacts

Các builder từ chối ghi một scope thiếu hoặc dữ liệu `--limit` vào đường dẫn
canonical. Khi dùng `--limit`, `--frame-limit`, subset một video hoặc hashing
encoder, luôn ghi sang `outputs/smoke/`:

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

Không copy smoke index vào `artifacts/indexes/`.

## Chạy retrieval và ba task

### Xem candidate thô

```bash
python scripts/run_search.py \
  --query "một người đang mở laptop" \
  --task kis \
  --limit 20 \
  --output outputs/search_debug.json
```

### KIS

```bash
python scripts/run_kis.py \
  --query "một người đang mở laptop" \
  --query-id KIS001 \
  --limit 100 \
  --output outputs/KIS001.json
```

Thêm `--dense-refine` để decode video quanh các coarse hit và chấm lại exact
frame bằng OpenCV/OpenCLIP.

### Q&A

Sau khi điền `GEMINI_API_KEY` trong `.env`:

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

Luồng thực tế:

```text
visual + ASR + OCR + object indexes local
  → retrieve/rank/diversify frame candidates
  → resolve ảnh và video/frame ID từ registry local
  → gửi tối đa N frame đã chọn tới Gemini theo batch
  → validate structured answer và candidate ID
  → cache/audit response trong artifacts/gemini/
```

Gemini không nhìn toàn dataset và không được dùng để captioning. Mỗi API call chỉ
nhận một batch frame do local retrieval đã chọn và registry local đã xác thực.
Mặc định script giữ các câu trả lời đã xác minh thành danh sách xếp hạng để phù
hợp cách chấm R@1/5/20/50/100. Khi chỉ smoke test API, thêm `--single-answer` để
gọi Gemini đúng một lần. Model API được lấy từ `[gemini].model`; đổi giá trị đó
nếu muốn dùng model khác.

### TRAKE

```bash
python scripts/run_trake.py \
  --query "vận động viên chạy đà, sau đó giậm nhảy, sau đó tiếp đất" \
  --query-id TRAKE001 \
  --limit 100 \
  --dense-refine \
  --output outputs/TRAKE001.json
```

TRAKE retrieve riêng từng event, gom candidate theo video rồi dùng dynamic
programming để giữ đúng temporal order.

## Dữ liệu nào lưu vector, dữ liệu nào lưu JSON

| Dữ liệu | Định dạng | Lý do |
|---|---|---|
| CLIP keyframe/window vectors | `.npy` float16/float32 | Mmap và exact search nhanh |
| ASR/OCR/object vectors | `.npy` float16/float32 | Semantic search từng modality |
| Video metadata gốc | `.json`/manifest | Mặc định không lặp thành vector cho mọi window |
| Vector row metadata | `.jsonl` | Row `i` khớp vector row `i` |
| Dataset/index/window manifest | `.json` | Scope, row count, hash, encoder, dtype |
| PhoWhisper transcript | `.json` | Segment, word timestamp và model metadata |
| Timeline, OCR, object, windows | `.jsonl` | Một record mỗi keyframe/window, dễ debug |
| Query/Gemini/evaluation result | `.json` | Evidence và score có cấu trúc |
| Submission | `.csv` | Định dạng nộp bài |

Không lưu list embedding float trong JSON. ID, timestamp, bbox và text gốc vẫn
ở JSON/JSONL để kiểm tra độc lập với vector.

## Evaluation, submission và test

```bash
python scripts/evaluate_results.py \
  --ground-truth examples/ground_truth.json \
  --predictions outputs/KIS001.json \
  --output artifacts/reports/KIS001.json
```

```bash
python scripts/write_submission.py \
  --task kis \
  --predictions outputs/KIS001.json \
  --output artifacts/submissions/KIS001.csv
```

Chạy test suite:

```bash
python -m pytest -q
python -m ruff check .
python -m pip check
```

Xem thêm [bản đồ file Python](docs/python-scripts.md),
[hướng dẫn debug](docs/debugging.md), [kiến trúc](docs/architecture.md) và
[data contracts](docs/data-contracts.md).
