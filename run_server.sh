#!/usr/bin/env bash
# CantoBench 在 server 上的一鍵執行腳本。
#
# 用法：
#   ./run_server.sh                    # 試產 500 張，檢查品質
#   ./run_server.sh full               # 產出完整 benchmark（config.py 的規模）
#   ./run_server.sh full /mnt/nvme/cb  # 指定輸出目錄
#
# 第一次執行會建 venv、裝套件、下載約 250MB 的 OFL 字型，需要十幾分鐘。
# 之後再跑會沿用快取，語料與字型都不會重抓。

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODE="${1:-test}"
OUT="${2:-$HERE/output}"
WORKERS="${WORKERS:-$(( $(nproc 2>/dev/null || echo 4) - 2 ))}"
[ "$WORKERS" -lt 1 ] && WORKERS=1

echo "=============================================================="
echo " CantoBench  模式=$MODE  輸出=$OUT  workers=$WORKERS"
echo "=============================================================="

# ---- 1. Python 環境 ----
if [ ! -d .venv ]; then
    echo "[1/4] 建立 virtualenv ..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[1/4] 安裝／確認套件 ..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# ---- 2. 字型 ----
# 沒有字型的話什麼都做不了，而乾淨的 Ubuntu server 通常一個中文字型都沒有。
echo
echo "[2/4] 下載 OFL 字型 ..."
python fetch_fonts.py

# ---- 3. 產出 ----
echo
if [ "$MODE" = "full" ]; then
    echo "[3/4] 產出完整 benchmark ..."
    python build_benchmark.py --out "$OUT" --workers "$WORKERS"
else
    echo "[3/4] 試產 500 張（要跑完整規模請用：./run_server.sh full）..."
    python build_benchmark.py --out "$OUT" --total 500 --workers "$WORKERS"
fi

# ---- 4. 檢查 ----
echo
echo "[4/4] 品質檢查 ..."
python verify.py --out "$OUT"

echo
echo "=============================================================="
echo " 完成。接下來："
echo
echo "   1. 把 $OUT/contact_sheet.jpg 拉回本機用眼睛看一遍"
echo "   2. 複核 $OUT/charset_review.tsv，"
echo "      要排除的字填進 data/exclude_chars.txt，漏掉的填 data/include_chars.txt"
echo "   3. 確認沒問題後跑：./run_server.sh full"
echo "=============================================================="
