# Bản đồ các file Python

Chạy tất cả lệnh từ thư mục gốc `AgentForce/`. Mỗi hàng dưới đây là một entry
file độc lập; không có CLI trung gian.

| Việc cần làm | File chạy trực tiếp |
|---|---|
| Kiểm tra dependency/path | `python scripts/check_environment.py` |
| Validate dataset | `python scripts/validate_dataset.py --strict` |
| Export timeline | `python scripts/export_timelines.py --video-id L21_V001` |
| ASR | `python scripts/transcribe_videos.py --video-id L21_V001` |
| OCR | `python scripts/run_ocr.py --video-id L21_V001` |
| Chuẩn hoá object | `python scripts/normalize_objects.py --video-id L21_V001` |
| Caption keyframe | `python scripts/caption_keyframes.py --video-id L21_V001` |
| Build temporal windows | `python scripts/build_windows.py --all` |
| Build visual indexes | `python scripts/build_visual_indexes.py --level both` |
| Build text indexes | `python scripts/build_text_indexes.py` |
| Xem retrieval thô | `python scripts/run_search.py --query "..."` |
| Textual KIS | `python scripts/run_kis.py --query "..."` |
| Q&A end-to-end | `python scripts/run_qa.py --query "context" --question "..."` |
| Q&A từ request JSON | `python scripts/verify_qa_request.py --request ...` |
| TRAKE | `python scripts/run_trake.py --query "event 1, sau đó event 2"` |
| Đánh giá | `python scripts/evaluate_results.py --ground-truth ... --predictions ...` |
| Xuất submission | `python scripts/write_submission.py --task kis ...` |

Luồng code khi chạy một query:

```text
scripts/run_<task>.py
  → src/agentforce/runtime.py
  → src/agentforce/engine.py hoặc src/agentforce/tasks/<task>.py
  → src/agentforce/retrieval/*.py
  → artifacts/indexes/*
```

`scripts/_bootstrap.py` chỉ thêm thư mục `src/` vào Python import path. Nó
không phân luồng command, không gọi subprocess và không che entry file đang chạy.
