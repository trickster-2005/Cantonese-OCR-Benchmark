# -*- coding: utf-8 -*-
"""把 evaluate.py 的輸出（output/eval_results.csv）轉成評測結果瀏覽器要的資料。

跟 export_samples.py（資料集預覽用的 53 張精選樣本）是不同的東西——這支是把
「完整」評測結果（9,860 筆、對應全部 4,930 張圖）匯出，給獨立的評測瀏覽器頁面
（docs/eval/、docs/eval/en/）用。

產出：
    docs/eval/assets/eval_data.json   精簡過欄位的評測記錄，給前端篩選/排序用
    docs/eval/assets/images/*.jpg     4,930 張圖（兩個模型共用同一批，不重複複製）

為什麼欄位要精簡：prediction_raw、error、latency_sec 對一般讀者瀏覽評測結果
沒有直接幫助，且會讓 JSON 肥不少；error 全部是空字串（本次評測零失敗），完全
沒有資訊量，直接省略。真的要看原始輸出的人，GitHub repo 裡的 eval_results.csv
本來就留著，不會因為這支腳本精簡就消失。
"""

import csv
import json
import os
import shutil
import sys

import config

KEEP_FIELDS = [
    "model", "file", "tier", "role", "source", "difficulty", "plane",
    "font_family", "n_chars", "minimal_pair_with",
    "gt", "prediction", "cer_clipped", "acc", "ned_sim", "f1", "edit_distance",
]

NUMERIC_INT = {"n_chars", "edit_distance"}
NUMERIC_FLOAT = {"cer_clipped", "acc", "ned_sim", "f1"}


def load_rows(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                # 這次評測零失敗，理論上不會進到這裡；留著是保險，
                # 免得哪次重跑評測有失敗筆數，卻被靜默地當成正常資料放上網頁。
                continue
            item = {}
            for k in KEEP_FIELDS:
                v = r.get(k, "")
                if k in NUMERIC_INT:
                    v = int(v) if v else 0
                elif k in NUMERIC_FLOAT:
                    v = round(float(v), 4) if v else 0.0
                item[k] = v
            rows.append(item)
    return rows


def copy_images(rows, images_src, images_dest):
    files = sorted({r["file"] for r in rows})
    os.makedirs(images_dest, exist_ok=True)
    # 先清掉舊圖，避免改了語料重新評測後，資料夾裡混著新舊不一致的殘留檔案
    for f in os.listdir(images_dest):
        os.remove(os.path.join(images_dest, f))
    missing = []
    for f in files:
        src = os.path.join(images_src, f)
        if not os.path.exists(src):
            missing.append(f)
            continue
        shutil.copy2(src, os.path.join(images_dest, f))
    return len(files), missing


def main():
    csv_path = os.path.join(config.OUT_DIR, "eval_results.csv")
    images_src = os.path.join(config.OUT_DIR, config.IMAGES_SUBDIR)
    if not os.path.exists(csv_path):
        sys.exit(f"找不到 {csv_path}，請先執行 evaluate.py")

    rows = load_rows(csv_path)
    print(f"[export_eval] 讀入 {len(rows):,} 筆評測記錄")

    eval_dir = os.path.join(config.ROOT, "docs", "eval")
    assets_dir = os.path.join(eval_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    json_path = os.path.join(assets_dir, "eval_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = os.path.getsize(json_path) / 1024 / 1024
    print(f"[export_eval] JSON → {json_path} ({size_mb:.2f} MB)")

    n_images, missing = copy_images(rows, images_src, os.path.join(assets_dir, "images"))
    if missing:
        print(f"[export_eval] 警告：{len(missing)} 張圖片在 {images_src} 找不到，"
              f"例如 {missing[:3]}", file=sys.stderr)
    total_img_mb = sum(
        os.path.getsize(os.path.join(assets_dir, "images", f))
        for f in os.listdir(os.path.join(assets_dir, "images"))
    ) / 1024 / 1024
    print(f"[export_eval] 圖片 {n_images:,} 張 → {assets_dir}/images/ "
          f"({total_img_mb:.1f} MB)")

    # 統計摘要，之後寫頁面說明文字時直接引用，不用重新算
    models = sorted({r["model"] for r in rows})
    summary = dict(
        n_records=len(rows),
        n_images=n_images,
        models=models,
        by_model={m: sum(1 for r in rows if r["model"] == m) for m in models},
    )
    print(f"[export_eval] 摘要: {summary}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
