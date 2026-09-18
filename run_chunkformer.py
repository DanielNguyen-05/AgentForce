"""
Batch ASR bằng ChunkFormer (arXiv:2502.14673, ICASSP 2025) - phương án thử nghiệm thứ 2
song song với PhoWhisper trong audio_to_json.py, để so sánh chất lượng trên cùng video.
------------------------------------------------------------------------------------
Khác biệt kiến trúc so với PhoWhisper/Faster-Whisper:
  - ChunkFormer = Conformer-CTC, KHÔNG phải Whisper -> không convert sang CTranslate2/
    faster-whisper được. Dùng package "chunkformer" riêng (pip install chunkformer).
  - Xử lý audio theo chunk với "relative right context", tối ưu cho audio rất dài
    (test tới 16h) và cho GPU ít VRAM (bảng tham khảo: 24GB VRAM ~ xử lý được 240 phút
    audio dồn 1 batch, 80GB ~ 980 phút). Chạy CPU vẫn được nhưng chậm hơn nhiều.
  - endless_decode() trả về list các đoạn dạng {"decode": text, "start": "HH:MM:SS.mmm",
    "end": "HH:MM:SS.mmm"} - không có confidence per-segment như faster-whisper
    (avg_logprob), nên trường "confidence" trong JSON output sẽ để null.

Cài đặt:
    pip install chunkformer

Model card: https://huggingface.co/khanhld/chunkformer-ctc-large-vie
Lần đầu chạy sẽ tự tải checkpoint (~vài GB) từ huggingface.co, cần internet.

Cách dùng: sửa CONFIG bên dưới rồi chạy:
    python3 run_chunkformer.py
Không dùng argparse, giống quy ước của audio_to_json.py.
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================== CONFIG ======================================
INPUT_DIR = "uploads"
OUTPUT_DIR = "outputs/transcripts_chunkformer"
RECURSIVE = False
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
SKIP_EXISTING = True

TEMP_AUDIO_PATH = "audio_pipeline/_tmp_audio_cf.wav"
FFMPEG_BINARY: Optional[str] = None   # None -> tự dùng imageio-ffmpeg nếu có (xem audio_to_json.py)

# --- Cấu hình model ChunkFormer ---
MODEL_ID = "khanhld/chunkformer-ctc-large-vie"   # 1.55B tham số, có thể đổi sang bản
                                                  # -rnnt-large-vie nếu muốn thử biến thể RNN-T
DEVICE = "auto"          # "auto" tự chọn cuda:0 nếu có GPU, fallback "cpu"
CHUNK_SIZE = 64
LEFT_CONTEXT_SIZE = 128
RIGHT_CONTEXT_SIZE = 128
# Giảm xuống 300-600 nếu chạy GPU ít VRAM (<12GB) hoặc CPU để tránh out-of-memory;
# xem bảng tham khảo trong docstring ở trên.
TOTAL_BATCH_DURATION = 1800   # giây
MAX_SILENCE_DURATION = 0.5    # giây, dùng để tách câu (tương đương VAD ở mục 4.2)

# --- Chuẩn hoá audio (mục 4.1, giống audio_to_json.py để so sánh công bằng) ---
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_LUFS = -16.0
# ============================================================================


def resolve_ffmpeg_binary() -> str:
    if FFMPEG_BINARY:
        return FFMPEG_BINARY
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def extract_audio(video_path: str, audio_path: str) -> None:
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg_binary()
    cmd = [
        ffmpeg_bin, "-y", "-i", video_path, "-vn",
        "-ac", str(TARGET_CHANNELS), "-ar", str(TARGET_SAMPLE_RATE),
        "-af", f"loudnorm=I={TARGET_LUFS}:LRA=11:TP=-1.5",
        "-acodec", "pcm_s16le", audio_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Không tìm thấy '{ffmpeg_bin}'. Chạy: pip install imageio-ffmpeg"
        ) from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg thất bại:\n{result.stderr}")


def clean_text(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?…":
        t += "."
    return t


def hhmmss_to_seconds(ts: str) -> float:
    """Parse timestamp ChunkFormer trả về sang giây (float).

    Dùng regex thay vì split(":") cứng nhắc vì các version/biến thể khác nhau của
    package chunkformer có thể trả về định dạng lệch nhau:
      - "HH:MM:SS.mmm"  (dấu chấm trước phần mili-giây - bản đã verify từ PyPI 1.2.2)
      - "HH:MM:SS:mmm"  (dấu hai chấm trước phần mili-giây - gặp ở 1 số version khác)
    """
    m = re.match(r"^(\d+):(\d+):(\d+)[.:](\d+)$", ts.strip())
    if not m:
        raise ValueError(f"Không nhận diện được định dạng timestamp: '{ts}'")
    h, mi, s, frac = m.groups()
    # frac có thể là mili-giây (3 chữ số) hoặc phần thập phân khác độ dài -> quy đổi
    # tổng quát theo số chữ số thay vì giả định cố định 3 chữ số.
    seconds = int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / (10 ** len(frac))
    return round(seconds, 3)


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def configure_pydub_ffmpeg() -> None:
    """ChunkFormer dùng pydub nội bộ (AudioSegment.from_file) để đọc audio, và pydub mặc
    định tự dò 'ffmpeg'/'avconv' qua PATH hệ thống - độc lập với cách extract_audio() ở
    trên tìm ffmpeg qua imageio-ffmpeg. Nếu không đồng bộ, máy chưa cài ffmpeg hệ thống
    (dù imageio-ffmpeg đã có) sẽ khiến bước decode bên trong chunkformer lỗi âm thầm
    hoặc cảnh báo "Couldn't find ffmpeg or avconv". Trỏ pydub dùng chung 1 binary ffmpeg
    với phần còn lại của script để tránh phụ thuộc PATH hệ thống.
    """
    try:
        from pydub import AudioSegment
        ffmpeg_bin = resolve_ffmpeg_binary()
        AudioSegment.converter = ffmpeg_bin
        AudioSegment.ffmpeg = ffmpeg_bin
    except ImportError:
        pass  # pydub sẽ được cài kèm theo dependency của chunkformer, bỏ qua nếu chưa có


def find_videos(input_dir: Path) -> list:
    pattern_iter = input_dir.rglob("*") if RECURSIVE else input_dir.glob("*")
    return sorted(p for p in pattern_iter if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def call_endless_decode(model, audio_path: str):
    """Gọi model.endless_decode() nhưng chỉ truyền các tham số mà version chunkformer
    đang cài đặt THỰC SỰ hỗ trợ (dò bằng inspect.signature). Một số version không có
    tham số max_silence_duration, nên hard-code đủ tham số dễ gây lỗi
    "unexpected keyword argument" khi package được cập nhật/hạ version khác nhau.
    """
    import inspect

    candidate_kwargs = {
        "audio_path": audio_path,
        "chunk_size": CHUNK_SIZE,
        "left_context_size": LEFT_CONTEXT_SIZE,
        "right_context_size": RIGHT_CONTEXT_SIZE,
        "total_batch_duration": TOTAL_BATCH_DURATION,
        "return_timestamps": True,
        "max_silence_duration": MAX_SILENCE_DURATION,
    }
    try:
        accepted = set(inspect.signature(model.endless_decode).parameters.keys())
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
        dropped = set(candidate_kwargs) - set(kwargs)
        if dropped:
            print(f"  [Lưu ý] Version chunkformer hiện tại không hỗ trợ tham số: "
                  f"{', '.join(sorted(dropped))} -> bỏ qua, dùng giá trị mặc định của model.")
    except (TypeError, ValueError):
        # Không introspect được (vd hàm bọc qua decorator lạ) -> thử nguyên bộ tham số gốc,
        # nếu vẫn lỗi thì rơi xuống except TypeError bên dưới để tự rút gọn dần.
        kwargs = candidate_kwargs

    try:
        return model.endless_decode(**kwargs)
    except TypeError as e:
        # Phòng hờ trường hợp introspect "đoán" sai (vd hàm nhận **kwargs tổng quát nhưng
        # thực ra không xử lý hết) -> rút gọn dần về bộ tham số tối thiểu chắc chắn có.
        print(f"  [Lưu ý] Gọi endless_decode với đủ tham số bị lỗi ({e}), thử lại với bộ "
              f"tham số tối thiểu.")
        minimal = {
            "audio_path": audio_path,
            "chunk_size": CHUNK_SIZE,
            "left_context_size": LEFT_CONTEXT_SIZE,
            "right_context_size": RIGHT_CONTEXT_SIZE,
            "total_batch_duration": TOTAL_BATCH_DURATION,
        }
        return model.endless_decode(**minimal)


def parse_decode_result(raw_segments) -> list:
    """Chuẩn hoá kết quả trả về của endless_decode() về dạng list[{"decode","start","end"}],
    vì định dạng trả về có thể khác nhau giữa các version của package chunkformer:
      - list[dict] với key "decode"/"start"/"end" (bản đã kiểm chứng từ source PyPI 1.2.2)
      - chuỗi string nhiều dòng dạng "[00:00:01.200] - [00:00:02.400]: nội dung"
      - list[str] mỗi phần tử 1 dòng theo định dạng trên
    """
    if isinstance(raw_segments, str):
        raw_segments = raw_segments.strip().splitlines()

    line_pattern = re.compile(
        r"\[(?P<start>[\d:.]+)\]\s*-\s*\[(?P<end>[\d:.]+)\]\s*:\s*(?P<text>.*)"
    )
    parsed = []
    for item in raw_segments:
        if isinstance(item, dict):
            if "decode" in item and "start" in item and "end" in item:
                parsed.append({"decode": item["decode"], "start": item["start"], "end": item["end"]})
            else:
                raise ValueError(f"Không nhận diện được cấu trúc dict trả về: {item}")
        elif isinstance(item, str):
            m = line_pattern.match(item.strip())
            if not m:
                continue  # bỏ qua dòng trống/không khớp định dạng
            parsed.append({"decode": m.group("text"), "start": m.group("start"), "end": m.group("end")})
        else:
            raise ValueError(f"Kiểu dữ liệu không hỗ trợ trong kết quả endless_decode: {type(item)}")
    return parsed


def process_video(video_path: Path, model) -> dict:
    print("  [1/3] Trích xuất & chuẩn hoá audio ...")
    extract_audio(str(video_path), TEMP_AUDIO_PATH)

    print("  [2/3] Chạy ChunkFormer endless_decode ...")
    raw_segments = parse_decode_result(call_endless_decode(model, TEMP_AUDIO_PATH))

    print("  [3/3] Đóng gói JSON ...")
    segments = []
    for idx, seg in enumerate(raw_segments, start=1):
        raw = seg["decode"].strip()
        start = hhmmss_to_seconds(seg["start"])
        end = hhmmss_to_seconds(seg["end"])
        segments.append({
            "segment_id": idx,
            "time_range": {"start": start, "end": end},
            "audio_metadata": {
                "raw_text": raw,
                "clean_text": clean_text(raw),
                "confidence": None,   # CTC greedy decode không có avg_logprob per-segment
            },
        })
        print(f"[{start:>7.2f}s -> {end:>7.2f}s] {raw}")

    return {
        "source_video": video_path.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_config": {
            "asr_model": MODEL_ID,
            "architecture": "chunkformer-ctc",
            "device": resolve_device(),
            "chunk_size": CHUNK_SIZE,
            "left_context_size": LEFT_CONTEXT_SIZE,
            "right_context_size": RIGHT_CONTEXT_SIZE,
            "total_batch_duration": TOTAL_BATCH_DURATION,
            "max_silence_duration": MAX_SILENCE_DURATION,
        },
        "segments": segments,
    }


def main():
    try:
        from chunkformer import ChunkFormerModel
    except ImportError:
        print("Chưa cài package 'chunkformer'. Chạy: pip install chunkformer", file=sys.stderr)
        sys.exit(1)

    configure_pydub_ffmpeg()

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    if not input_dir.exists():
        print(f"Không tìm thấy thư mục input: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if not videos:
        print(f"Không tìm thấy video nào trong {input_dir}")
        return

    device = resolve_device()
    print(f"Tìm thấy {len(videos)} video trong {input_dir}")
    print(f"Nạp ChunkFormer ({MODEL_ID}, device={device}) — lần đầu sẽ tự tải checkpoint ...")
    model = ChunkFormerModel.from_pretrained(MODEL_ID).to(device)

    done, skipped, failed = 0, 0, 0
    for idx, video_path in enumerate(videos, start=1):
        out_path = output_dir / f"{video_path.stem}.json"
        print(f"\n[{idx}/{len(videos)}] {video_path.name}")

        if SKIP_EXISTING and out_path.exists():
            print(f"  -> Bỏ qua (đã có {out_path.name})")
            skipped += 1
            continue

        try:
            result = process_video(video_path, model)
        except Exception as e:
            print(f"  -> LỖI: {e}", file=sys.stderr)
            failed += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  -> Đã ghi {len(result['segments'])} segment vào {out_path.name}")
        done += 1

    print(
        f"\nHoàn tất batch: {done} thành công, {skipped} bỏ qua, {failed} lỗi, "
        f"trên tổng {len(videos)} video."
    )


if __name__ == "__main__":
    main()
