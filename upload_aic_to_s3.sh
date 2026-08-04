#!/usr/bin/env bash
# Chạy trên EC2 (cùng region với bucket): tải các zip AIC, giải nén,
# upload lên S3 theo cấu trúc dataset/videos/L##/ và dataset/keyframes/L##/
#
# Cách dùng:   ./upload_aic_to_s3.sh s3://ten-bucket
# Chạy nền:    nohup ./upload_aic_to_s3.sh s3://ten-bucket > upload.log 2>&1 &

set -euo pipefail

BUCKET="${1:?Cách dùng: $0 s3://ten-bucket}"
BUCKET="${BUCKET%/}"
WORKDIR="${WORKDIR:-$HOME/aic_ingest}"

LINKS=(
  https://aic-data.ledo.io.vn/Keyframes_L21.zip
  https://aic-data.ledo.io.vn/Keyframes_L22.zip
  https://aic-data.ledo.io.vn/Keyframes_L23.zip
  https://aic-data.ledo.io.vn/Keyframes_L24.zip
  https://aic-data.ledo.io.vn/Keyframes_L25.zip
  https://aic-data.ledo.io.vn/Keyframes_L26_a.zip
  https://aic-data.ledo.io.vn/Keyframes_L26_b.zip
  https://aic-data.ledo.io.vn/Keyframes_L26_c.zip
  https://aic-data.ledo.io.vn/Keyframes_L26_d.zip
  https://aic-data.ledo.io.vn/Keyframes_L26_e.zip
  https://aic-data.ledo.io.vn/Keyframes_L27.zip
  https://aic-data.ledo.io.vn/Keyframes_L28.zip
  https://aic-data.ledo.io.vn/Keyframes_L29.zip
  https://aic-data.ledo.io.vn/Keyframes_L30.zip
  https://aic-data.ledo.io.vn/Videos_L21_a.zip
  https://aic-data.ledo.io.vn/Videos_L22_a.zip
  https://aic-data.ledo.io.vn/Videos_L23_a.zip
  https://aic-data.ledo.io.vn/Videos_L24_a.zip
  https://aic-data.ledo.io.vn/Videos_L25_a.zip
  https://aic-data.ledo.io.vn/Videos_L26_a.zip
  https://aic-data.ledo.io.vn/Videos_L26_b.zip
  https://aic-data.ledo.io.vn/Videos_L26_c.zip
  https://aic-data.ledo.io.vn/Videos_L26_d.zip
  https://aic-data.ledo.io.vn/Videos_L26_e.zip
  https://aic-data.ledo.io.vn/Videos_L27_a.zip
  https://aic-data.ledo.io.vn/Videos_L28_a.zip
  https://aic-data.ledo.io.vn/Videos_L29_a.zip
  https://aic-data.ledo.io.vn/Videos_L30_a.zip
)

command -v unzip >/dev/null || { echo "Thiếu unzip. Cài: sudo dnf install -y unzip (hoặc yum/apt)"; exit 1; }
command -v aws   >/dev/null || { echo "Thiếu aws cli"; exit 1; }

# Tăng số kết nối song song lên S3 (keyframes là hàng vạn file jpg nhỏ)
aws configure set default.s3.max_concurrent_requests 32
aws configure set default.s3.multipart_chunksize 64MB

mkdir -p "$WORKDIR"
cd "$WORKDIR"

total=${#LINKS[@]}
i=0
for url in "${LINKS[@]}"; do
  i=$((i + 1))
  f=$(basename "$url")                       # vd: Videos_L26_a.zip
  lname=$(grep -oE 'L[0-9]+' <<<"$f" | head -1)   # vd: L26

  case "$f" in
    Videos_*)    prefix="dataset/videos/$lname" ;;
    Keyframes_*) prefix="dataset/keyframes/$lname" ;;
    *)           prefix="dataset/other/${f%.zip}" ;;
  esac
  dest="$BUCKET/$prefix/"

  # Đã xử lý xong ở lần chạy trước thì bỏ qua (cho phép chạy lại an toàn)
  if aws s3 ls "$BUCKET/dataset/.done/$f" >/dev/null 2>&1; then
    echo "[$i/$total] $f — đã xong trước đó, bỏ qua"
    continue
  fi

  echo "[$i/$total] $f  ->  $dest"

  # aria2c: 8 connection/file để né server bóp băng thông từng connection,
  # tự resume file dở; fallback curl có chống treo (dưới 50KB/s trong 60s thì cắt và retry)
  if command -v aria2c >/dev/null; then
    aria2c -x16 -s16 -c --max-tries=0 --retry-wait=10 \
           --timeout=60 --lowest-speed-limit=0 -o "$f" "$url"
  else
    curl -fL --retry 10 --retry-delay 10 -C - \
         --speed-limit 51200 --speed-time 60 -o "$f" "$url"
  fi

  rm -rf extracted
  mkdir extracted
  unzip -q "$f" -d extracted

  # Nếu zip có 1 folder bọc ngoài (vd Keyframes_L21/) thì đi sâu vào trong
  # để key trên S3 không bị lặp tên folder
  src="extracted"
  while [ "$(find "$src" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ] \
        && [ -d "$src/$(ls "$src")" ]; do
    src="$src/$(ls "$src")"
  done

  aws s3 sync "$src" "$dest" --only-show-errors

  echo done | aws s3 cp - "$BUCKET/dataset/.done/$f"
  rm -rf "$f" extracted
done

echo "=== HOÀN TẤT. Kiểm tra: aws s3 ls $BUCKET/dataset/ --recursive --summarize | tail -3"
