# AgentForce AIC 2026

Pipeline truy xuất video đa phương thức cho ba dạng bài AIC 2026:

- Textual KIS: tìm đúng video và frame.
- Q&A: retrieval local, sau đó dùng Gemini API xác minh trên một tập frame nhỏ.
- TRAKE: tìm từng event, căn chỉnh thứ tự và có thể dò dense frame.

Project chỉ dùng các file Python chạy trực tiếp. Không có lệnh terminal riêng,
không có command router và không còn `cli.py`. Ví dụ:

```bash
python scripts/run_kis.py --query "một người đang mở laptop"
```

## Trạng thái dữ liệu hiện tại

- 873 video, 177.321 keyframe và 873 CLIP shard đã được phát hiện.
- Dataset validation: 0 error, 0 warning.
- 93.902 temporal window đã được build.
- Visual keyframe index và visual window index đã có trong `artifacts/indexes/`.
- NumPy exact search là backend mặc định dễ kiểm tra; FAISS là tùy chọn.
- Faster-Whisper, OCR, OpenCLIP, Sentence Transformers và Gemini được import theo nhu cầu.

## Cấu trúc project

```text
AgentForce/
├── scripts/                       # Các file được chạy trực tiếp bằng Python
│   ├── validate_dataset.py
│   ├── transcribe_videos.py
│   ├── run_ocr.py
│   ├── normalize_objects.py
│   ├── caption_keyframes.py
│   ├── build_windows.py
│   ├── build_visual_indexes.py
│   ├── build_text_indexes.py
│   ├── run_search.py
│   ├── run_kis.py
│   ├── run_qa.py
│   ├── run_trake.py
│   ├── evaluate_results.py
│   └── write_submission.py
├── src/agentforce/                # Thư viện Python nội bộ
│   ├── data/                      # Manifest, schema và canonical timeline
│   ├── preprocessing/             # ASR, OCR, object, caption, windows
│   ├── embeddings/                # OpenCLIP và text encoder adapters
│   ├── indexing/                  # NumPy/FAISS và index builders
│   ├── retrieval/                 # Query parser, fusion, ranking
│   ├── temporal/                  # Alignment và dense refinement
│   ├── tasks/                     # KIS, Q&A và TRAKE solvers
│   ├── gemini/                    # Client, schema, retry và cache
│   ├── evaluation/                # R@1/5/20/50/100 và Final Score
│   ├── submission/                # CSV theo từng task
│   ├── runtime.py                 # Khởi tạo index/model dùng chung
│   └── engine.py                  # Retrieval đa modality
├── configs/default.toml           # Tham số pipeline
├── tests/                         # Unit/integration tests nhẹ
├── docs/                          # Kiến trúc, data contracts và debug
├── examples/                      # JSON mẫu
├── dataset/                       # Dữ liệu gốc, chỉ đọc
└── artifacts/                     # Dữ liệu phát sinh và cache
```

Tên package `agentforce` trong `src/` chỉ là namespace để các file Python
import lẫn nhau; nó không phải một lệnh terminal.

## Cài đặt

Khuyến nghị Python 3.11 hoặc 3.12 để tương thích tốt với Torch, FAISS và
Faster-Whisper.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Cài theo chức năng cần dùng:

```bash
python -m pip install -e '.[vision,semantic,video]'
python -m pip install -e '.[asr]'
python -m pip install -e '.[ocr]'
python -m pip install -e '.[gemini]'
```

Hoặc cài toàn bộ:

```bash
python -m pip install -e '.[all,dev]'
```

OCR Tesseract còn cần binary và language pack `vie`. Kiểm tra máy bằng:

```bash
python scripts/check_environment.py
```

Không bắt buộc cài package editable để chạy script: `scripts/_bootstrap.py` tự
thêm `src/` vào import path. Tuy nhiên vẫn cần cài các dependency tương ứng.

## Chuẩn bị dữ liệu và indexes

### 1. Validate dataset

```bash
python scripts/validate_dataset.py --strict
```

Manifest được ghi vào `artifacts/manifests/dataset.json`.

### 2. Build temporal windows và visual indexes

```bash
python scripts/build_windows.py --all
python scripts/build_visual_indexes.py --level both
```

Vector nằm trong file `.npy`; metadata cùng hàng nằm trong `.jsonl`; thông tin
phiên bản và source hash nằm trong `.manifest.json`.

### 3. Bổ sung modality

Nên thử trên một video trước:

```bash
python scripts/normalize_objects.py --video-id L21_V001 --frame-limit 20
python scripts/run_ocr.py --video-id L21_V001 --frame-limit 20
python scripts/transcribe_videos.py --video-id L21_V001 --model base
```

Sau khi kiểm tra JSON/JSONL đầu ra, chạy toàn corpus:

```bash
python scripts/normalize_objects.py --all
python scripts/run_ocr.py --all
python scripts/transcribe_videos.py --all
python scripts/build_windows.py --all
python scripts/build_text_indexes.py
```

Các bước dài bỏ qua output đã tồn tại; dùng `--overwrite` khi thực sự muốn chạy
lại. Captioning là tùy chọn và gọi API:

```bash
export GEMINI_API_KEY="..."
python scripts/caption_keyframes.py --video-id L21_V001 --stride 3 --frame-limit 10
```

Chạy caption toàn bộ cần xác nhận chi phí rõ ràng:

```bash
python scripts/caption_keyframes.py --all --stride 3 --confirm-api-cost
```

## Chạy query

### Xem candidate retrieval thô

```bash
python scripts/run_search.py \
  --query "một người đang mở laptop" \
  --task kis \
  --limit 20
```

### Textual KIS

```bash
python scripts/run_kis.py \
  --query "một người đang mở laptop" \
  --query-id KIS001 \
  --output artifacts/runs/KIS001.json
```

Muốn tìm frame chính xác quanh các coarse hit tốt nhất, thêm `--dense-refine`.

### Q&A dùng Gemini API

```bash
export GEMINI_API_KEY="..."
python scripts/run_qa.py \
  --query "Người phụ nữ mặc váy đỏ đang cầm một chiếc ly" \
  --question "Chiếc ly có màu gì?" \
  --query-id QA001 \
  --output artifacts/runs/QA001.json
```

Luồng Q&A:

```text
local multimodal indexes
  → retrieve và diversify candidate frames
  → gửi tối đa N frame đại diện tới Gemini theo batch nhỏ
  → validate structured answer
  → chỉ chấp nhận video/frame ID có trong candidate local
  → cache response và xếp hạng answer alternatives
```

Gemini chỉ dùng cho bước hiểu ảnh/trả lời cuối; retrieval toàn bộ corpus vẫn chạy
local để giới hạn latency và chi phí API. Có thể test một request frame chuẩn bị
sẵn bằng:

```bash
python scripts/verify_qa_request.py \
  --request examples/qa_request.json \
  --output artifacts/runs/qa_request_result.json
```

### TRAKE

```bash
python scripts/run_trake.py \
  --query "vận động viên chạy đà, sau đó giậm nhảy, sau đó bay qua xà, sau đó tiếp đất" \
  --query-id TRAKE001 \
  --dense-refine \
  --output artifacts/runs/TRAKE001.json
```

`--dense-refine` decode video quanh coarse timestamp bằng OpenCV và chấm lại
từng frame bằng OpenCLIP. Bước này hữu ích khi khoảng ground truth rất ngắn.

## Dữ liệu nào là embedding, dữ liệu nào là JSON

| Dữ liệu | Định dạng | Lý do |
|---|---|---|
| CLIP image vectors | `.npy` float16/float32 | Ma trận số lớn, cần mmap/search nhanh |
| Window visual vectors | `.npy` float16/float32 | Mean-pooled vector để retrieval |
| ASR/OCR/caption/object text vectors | `.npy` float16/float32 | Semantic search theo từng modality |
| Dataset manifest và index manifest | `.json` | Cấu hình, cardinality, hash, model version |
| ASR transcript | `.json` | Segment, word timestamp và metadata lồng nhau |
| OCR, object, caption, timeline, windows | `.jsonl` | Mỗi frame/window một record, đọc tuần tự dễ debug |
| Query result, Gemini response, evaluation | `.json` | Cần giữ evidence và score có cấu trúc |
| Submission | `.csv` | Định dạng nộp bài |

Không ghi embedding float vào JSON. ID, timestamp, bbox, text gốc, model name và
source hash không nằm trong vector; chúng được lưu ở JSON/JSONL để có thể kiểm
tra và rebuild độc lập.

## Evaluation và submission

```bash
python scripts/evaluate_results.py \
  --ground-truth examples/ground_truth.json \
  --predictions examples/predictions.json \
  --output artifacts/reports/example.json
```

```bash
python scripts/write_submission.py \
  --task kis \
  --predictions examples/predictions.json \
  --query-id KIS001 \
  --output artifacts/submissions/KIS001.csv
```

## Tests và debug

```bash
pytest -q
```

Test suite không tải model và không gọi Gemini thật. Để debug một query:

```bash
python -m pdb scripts/run_kis.py --query "một người mở laptop"
```

Đặt breakpoint đầu tiên trong chính `scripts/run_kis.py`, sau đó step vào
`src/agentforce/runtime.py`, `src/agentforce/tasks/kis.py` và các module
`src/agentforce/retrieval/`.

Xem [bản đồ file Python](docs/python-scripts.md),
[hướng dẫn debug](docs/debugging.md), [kiến trúc](docs/architecture.md) và
[data contracts](docs/data-contracts.md).
