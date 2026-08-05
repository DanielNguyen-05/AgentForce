```bash
#!/usr/bin/env bash
# Chạy trên EC2:
#   ./upload_aic_to_s3.sh s3://ten-bucket
#
# Chạy nền:
#   nohup ./upload_aic_to_s3.sh s3://ten-bucket > upload.log 2>&1 &

set -e
set -u
if [ -n "${BASH_VERSION:-}" ]; then
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/upload.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

echo "=== Starting upload_aic_to_s3.sh at $(date '+%Y-%m-%d %H:%M:%S') ==="

BUCKET="${1:?Cách dùng: $0 s3://ten-bucket}"
BUCKET="${BUCKET%/}"

WORKDIR="${WORKDIR:-$HOME/aic_ingest}"

# =========================
# Tuning
# =========================

# Mỗi ZIP chỉ dùng từng này connection
DOWNLOAD_CONNECTIONS="${DOWNLOAD_CONNECTIONS:-4}"

# Segment size
DOWNLOAD_SPLIT_SIZE="${DOWNLOAD_SPLIT_SIZE:-1M}"

# S3 concurrent upload
S3_CONCURRENT_REQUESTS="${S3_CONCURRENT_REQUESTS:-32}"

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

# =========================
# Check dependencies
# =========================

command -v unzip >/dev/null || {
  echo "ERROR: thiếu unzip"
  echo "Cài bằng: sudo apt install -y unzip"
  exit 1
}

command -v aws >/dev/null || {
  echo "ERROR: thiếu aws cli"
  exit 1
}

command -v aria2c >/dev/null || {
  echo "ERROR: thiếu aria2c"
  echo "Cài bằng: sudo apt install -y aria2"
  exit 1
}

# =========================
# AWS CLI tuning
# =========================

aws configure set default.s3.max_concurrent_requests "$S3_CONCURRENT_REQUESTS"
aws configure set default.s3.multipart_chunksize 64MB

# =========================
# Prepare
# =========================

mkdir -p "$WORKDIR/downloads"
cd "$WORKDIR"

total=${#LINKS[@]}

echo "=========================================="
echo "AIC Dataset -> S3"
echo "=========================================="
echo "Bucket:              $BUCKET"
echo "Workdir:             $WORKDIR"
echo "Connections / file:  $DOWNLOAD_CONNECTIONS"
echo "Total files:         $total"
echo "=========================================="

# ============================================================
# STEP 1 -> 3:
# Download -> Extract -> Upload từng file
#
# Chỉ xử lý 1 ZIP tại một thời điểm.
# ============================================================

i=0

for url in "${LINKS[@]}"; do
  i=$((i + 1))

  f=$(basename "$url")
  zip="$WORKDIR/downloads/$f"

  # =========================
  # Determine S3 destination
  # =========================

  lname=$(grep -oE 'L[0-9]+' <<< "$f" | head -1)

  case "$f" in
    Videos_*)
      prefix="dataset/videos/$lname"
      ;;

    Keyframes_*)
      prefix="dataset/keyframes/$lname"
      ;;

    *)
      prefix="dataset/other/${f%.zip}"
      ;;
  esac

  dest="$BUCKET/$prefix/"

  echo
  echo "=========================================="
  echo "[$i/$total] $f"
  echo "Destination: $dest"
  echo "=========================================="

  # =========================
  # Check already completed
  # =========================

  if aws s3 ls "$BUCKET/dataset/.done/$f" >/dev/null 2>&1; then
    echo "[SKIP] $f — đã upload trước đó"
    continue
  fi

  # ==========================================================
  # DOWNLOAD
  # ==========================================================

  echo
  echo "[DOWNLOAD] $f"

  aria2c \
    --continue=true \
    --max-connection-per-server="$DOWNLOAD_CONNECTIONS" \
    --split="$DOWNLOAD_CONNECTIONS" \
    --min-split-size="$DOWNLOAD_SPLIT_SIZE" \
    --file-allocation=none \
    --max-tries=10 \
    --retry-wait=10 \
    --timeout=60 \
    --connect-timeout=20 \
    --lowest-speed-limit=0 \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --summary-interval=5 \
    --console-log-level=notice \
    --dir="$WORKDIR/downloads" \
    --out="$f" \
    "$url"

  # =========================
  # Verify ZIP exists
  # =========================

  if [ ! -f "$zip" ]; then
    echo "ERROR: download xong nhưng không tìm thấy:"
    echo "$zip"
    exit 1
  fi

  echo "[DOWNLOAD OK] $f"

  # ==========================================================
  # EXTRACT
  # ==========================================================

  rm -rf "$WORKDIR/extracted"
  mkdir -p "$WORKDIR/extracted"

  echo
  echo "[EXTRACT] $f"

  unzip -q "$zip" -d "$WORKDIR/extracted"

  # Nếu ZIP có 1 folder bọc ngoài thì đi xuống folder đó.
  #
  # extracted/
  #   Keyframes_L21/
  #       xxx.jpg
  #
  # -> src = extracted/Keyframes_L21

  src="$WORKDIR/extracted"

  while true; do
    count=$(find "$src" -mindepth 1 -maxdepth 1 -print | wc -l)

    if [ "$count" -ne 1 ]; then
      break
    fi

    child=$(find "$src" -mindepth 1 -maxdepth 1 -print -quit)

    if [ -d "$child" ]; then
      src="$child"
    else
      break
    fi
  done

  echo "[EXTRACT OK] Source: $src"

  # ==========================================================
  # UPLOAD
  # ==========================================================

  echo
  echo "[UPLOAD] $src -> $dest"

  aws s3 sync \
    "$src" \
    "$dest" \
    --only-show-errors

  echo "[UPLOAD OK] $f"

  # ==========================================================
  # MARK DONE
  # ==========================================================

  echo "done" | aws s3 cp - "$BUCKET/dataset/.done/$f"

  echo "[DONE] $f"

  # ==========================================================
  # CLEANUP
  # ==========================================================

  echo "[CLEANUP] $f"

  rm -rf "$zip"
  rm -rf "$WORKDIR/extracted"

  echo
  echo "=========================================="
  echo "Completed [$i/$total]: $f"
  echo "=========================================="

done

echo
echo "=========================================="
echo "HOÀN TẤT"
echo "=========================================="

echo "Kiểm tra:"
echo "aws s3 ls $BUCKET/dataset/ --recursive --summarize | tail -3"
```
