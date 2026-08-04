"""
Convert PhoWhisper (checkpoint HuggingFace, VinAI) sang định dạng CTranslate2 để
faster-whisper load được như một model local, thay vì các checkpoint "openai/whisper-*"
mặc định.

PhoWhisper là bản Whisper được VinAI fine-tune thêm trên ~844 giờ audio tiếng Việt đa
vùng miền -> nhận dạng tên riêng/địa danh/phát âm vùng miền tốt hơn Whisper large-v3 gốc.

Chỉ cần chạy 1 lần (mất thời gian tải + convert), sau đó trỏ MODEL_SIZE trong
audio_to_json.py tới OUTPUT_DIR bên dưới.

Cách dùng:
    pip install transformers ctranslate2
    python3 convert_phowhisper.py
"""

import subprocess
import sys

# ============================== CONFIG ======================================
# Các checkpoint PhoWhisper công khai trên HuggingFace (VinAI), chọn 1 theo nhu cầu:
#   vinai/PhoWhisper-tiny / -base / -small / -medium / -large
# "large" chính xác nhất nhưng nặng nhất, cần nhiều RAM/VRAM hơn khi convert lẫn khi chạy.
SOURCE_MODEL = "vinai/PhoWhisper-large"
OUTPUT_DIR = r"./models/phowhisper-large-ct2"

# Lượng tử hoá: "int8" chạy CPU nhanh & nhẹ RAM (khuyến nghị nếu không có GPU),
# "float16" nếu chạy GPU (cần đổi DEVICE="cuda" trong audio_to_json.py).
QUANTIZATION = "int8"
# ============================================================================


def main():
    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", SOURCE_MODEL,
        "--output_dir", OUTPUT_DIR,
        "--quantization", QUANTIZATION,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]
    print(f"--- Đang convert {SOURCE_MODEL} -> {OUTPUT_DIR} (quantization={QUANTIZATION}) ---")
    print("Lưu ý: lần đầu chạy sẽ tự tải checkpoint từ huggingface.co, cần internet.")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            "\nConvert thất bại. Nếu lỗi liên quan tới thiếu file tokenizer/preprocessor, "
            "thử bỏ bớt --copy_files hoặc cài thêm: pip install -U transformers sentencepiece",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\nHoàn tất. Trong audio_to_json.py, đặt:\n  MODEL_SIZE = r\"{OUTPUT_DIR}\"")


if __name__ == "__main__":
    main()
