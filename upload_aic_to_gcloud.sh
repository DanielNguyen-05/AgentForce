#!/usr/bin/env bash
# Cách dùng:
#   ./upload_aic_to_gcloud.sh gs://ten-bucket
#
# Chạy nền:
#   nohup ./upload_aic_to_gcloud.sh gs://ten-bucket > /dev/null 2>&1 &

set -e
set -u
if [ -n "${BASH_VERSION:-}" ]; then
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/upload_gcloud.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

echo "=== Starting upload_aic_to_gcloud.sh at $(date '+%Y-%m-%d %H:%M:%S') ==="

BUCKET="${1:?Cách dùng: $0 gs://ten-bucket}"
BUCKET="${BUCKET%/}"

WORKDIR="${WORKDIR:-$HOME/aic_ingest}"

# =========================
# Tuning
# =========================

DOWNLOAD_CONNECTIONS="${DOWNLOAD_CONNECTIONS:-4}"
DOWNLOAD_SPLIT_SIZE="${DOWNLOAD_SPLIT_SIZE:-1M}"

LINKS=(
  https://aic-data.ledo.io.vn/map-keyframes-aic25-b1.zip
  https://aic-data.ledo.io.vn/media-info-aic25-b1.zip
  https://aic-data.ledo.io.vn/objects-aic25-b1.zip
)

# =========================
# Check dependencies
# =========================

command -v unzip >/dev/null || {
  echo "ERROR: thiếu unzip"
  exit 1
}

command -v gcloud >/dev/null || {
  echo "ERROR: thiếu gcloud CLI"
  exit 1
}

command -v aria2c >/dev/null || {
  echo "ERROR: thiếu aria2c"
  echo "Cài bằng: brew install aria2 (macOS) hoặc sudo apt install -y aria2"
  exit 1
}

# =========================
# Prepare
# =========================

mkdir -p "$WORKDIR/downloads"
cd "$WORKDIR"

total=${#LINKS[@]}

echo "=========================================="
echo "AIC Dataset -> GCS"
echo "=========================================="
echo "Bucket:              $BUCKET"
echo "Workdir:             $WORKDIR"
echo "Connections / file:  $DOWNLOAD_CONNECTIONS"
echo "Total files:         $total"
echo "=========================================="

i=0

for url in "${LINKS[@]}"; do
  i=$((i + 1))

  f=$(basename "$url")
  zip="$WORKDIR/downloads/$f"

  # =========================
  # Determine GCS destination
  # =========================
  #
  # map-keyframes / media-info: file nằm thẳng trong folder cha
  #   dataset/map-keyframes/L21_V001.csv
  #   dataset/media-info/L21_V001.json
  #
  # objects: giữ cấu trúc con L21_V001/, L21_V002/, ...
  #   dataset/objects/L21_V001/0001.json

  case "$f" in
    map-keyframes-*)
      prefix="dataset/map-keyframes"
      ;;

    media-info-*)
      prefix="dataset/media-info"
      ;;

    objects-*)
      prefix="dataset/objects"
      ;;

    *)
      prefix="dataset/other/${f%.zip}"
      ;;
  esac

  dest="$BUCKET/$prefix"

  echo
  echo "=========================================="
  echo "[$i/$total] $f"
  echo "Destination: $dest/"
  echo "=========================================="

  # =========================
  # Check already completed
  # =========================

  if gcloud storage ls "$BUCKET/dataset/.done/$f" >/dev/null 2>&1; then
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

  # Nếu ZIP có folder bọc ngoài (vd extracted/map-keyframes/...)
  # thì đi xuống folder trong cùng còn chứa nhiều hơn 1 mục.

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
  echo "[UPLOAD] $src -> $dest/"

  gcloud storage rsync -r \
    "$src" \
    "$dest"

  echo "[UPLOAD OK] $f"

  # ==========================================================
  # MARK DONE
  # ==========================================================

  echo "done" | gcloud storage cp - "$BUCKET/dataset/.done/$f"

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
echo "gcloud storage ls -r $BUCKET/dataset/ | tail -5"
