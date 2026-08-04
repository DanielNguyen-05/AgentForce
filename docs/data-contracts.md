# Data contracts

## Canonical identifiers

```text
video_id:      L25_V064
keyframe_uid:  L25_V064_K000115
window_id:     L25_V064_W000022
```

`keyframe_number` là số thứ tự keyframe do BTC cung cấp. `frame_idx` mới là frame ID dùng để nộp.

## Vector persistence

```text
visual_windows.npy
visual_windows.jsonl
visual_windows.manifest.json
```

Hàng `i` trong NPY luôn tương ứng dòng `i` trong JSONL. Manifest lưu dimension, dtype, normalization, encoder và source hash.

## Structured persistence

- Manifest/cấu hình/report nhỏ: JSON.
- Dữ liệu nhiều record: JSONL trong baseline; có thể đổi sang Parquet mà không đổi domain schema.
- Vector: NPY/FAISS, không lưu list float trong JSON.
- Submission: CSV task-specific.

## Temporal window

Mỗi window giữ thời gian/frame bounds, keyframe IDs và text đã căn chỉnh. ASR segment đi qua boundary được gán theo overlap, không bị loại chỉ vì không nằm trọn trong một scene.

## Gemini trust boundary

Gemini chỉ được trả `supporting_candidate_id`. `video_id` và `frame_idx` cuối được resolve từ registry local, không tin ID do model tự sinh.

