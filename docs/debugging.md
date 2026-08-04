# Debug bằng các file Python trực tiếp

Mỗi thao tác có đúng một entry file trong `scripts/`. Khi chạy query, file đầu
tiên đang chạy chính là file xuất hiện sau `python3`, ví dụ:

```bash
python scripts/run_kis.py --query "một người mở laptop"
```

Đặt breakpoint trong `scripts/run_kis.py`, sau đó đi tiếp vào các module được
import từ `src/agentforce/`. Không có dispatcher hoặc command router ở giữa.

## Dataset báo thiếu component

```bash
python scripts/validate_dataset.py --strict
```

Mở `artifacts/manifests/dataset.json`, rồi lọc `issues` theo `video_id`.

## Search báo stale index

Một preprocessing artifact đã đổi. Build lại theo thứ tự:

```bash
python scripts/build_windows.py --all
python scripts/build_visual_indexes.py --level window
python scripts/build_text_indexes.py
```

Không xoá dữ liệu gốc hoặc sửa manifest hash bằng tay.

## Không có OpenCLIP/SentenceTransformer

```bash
pip install -e '.[vision,semantic]'
python scripts/check_environment.py
```

Model có thể được tải ở lần chạy đầu tiên; nên chuẩn bị cache trước ngày thi.

## Gemini trả JSON lỗi

Client tự retry và validate schema. Kiểm tra:

- model trong `configs/default.toml`;
- cache entry trong `artifacts/gemini/`;
- candidate image có đọc được;
- API quota/rate limit;
- biến môi trường `GEMINI_API_KEY`.

## TRAKE không có path hợp lệ

- Xác nhận query có ít nhất hai event, mỗi event nằm trên một dòng.
- Tăng `retrieval.top_keyframes` hoặc `retrieval.top_videos` trong config.
- Kiểm tra từng event bằng `python scripts/run_search.py`.
- Truyền `--dense-refine` cho `scripts/run_trake.py` khi nhiều event gần nhau.
- Kiểm tra FPS và frame mapping của video.

## Theo dõi import/call stack

Vì entry file đã rõ, dùng công cụ chuẩn của Python:

```bash
python -m pdb scripts/run_kis.py --query "một người mở laptop"
```

Trong IDE, chọn chính file dưới `scripts/` làm Run/Debug target. Nếu cần xem
module nào được import, dùng:

```bash
python -X importtime scripts/run_kis.py --query "một người mở laptop"
```

## Kết quả thay đổi sau rebuild

Giữ lại config TOML, index manifests, query JSON, raw modality scores, model
versions và evaluation report. Không so sánh hai run chỉ bằng Final Score nếu
source hash hoặc model revision khác nhau.
