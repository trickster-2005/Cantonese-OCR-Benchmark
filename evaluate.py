# -*- coding: utf-8 -*-
"""在 CantoBench 上評測 VLM。

指標用學術與產業慣例，四個一起看：

  CER        字元錯誤率 = 編輯距離 / ground truth 長度。OCR 文獻的主指標，越低越好。
             可以 >1（模型輸出一堆廢話時），所以另外記 clip 到 1 的版本。
  ACC        整句完全正確的比例（sequence accuracy）。場景文字辨識的產業慣例。
  1-NED      1 − 正規化編輯距離。ICDAR 競賽慣例，比 CER 對長度不敏感。
  F1         字元級 precision/recall 調和平均。中文沒有詞邊界，固定用字元級。

三個刻意的設計決定：

  prompt 不提「繁體中文」。提了會把模型推向書面中文，而「粵語字被糾正成書面語字」
  正是本 benchmark 要測的現象——寫進 prompt 等於自己製造混淆變因。

  後處理只做「格式雜訊清理」，不做任何針對個別模型的客製。原始輸出永久保留在
  prediction_raw 欄，任何人都能用別的清理規則重算。這是 OCR 評測結果無法互相
  比較的最大單一原因。

  逐筆寫檔、可續跑。4,930 張 × 2 個模型是數小時的工作，中途斷掉不能從頭來。
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import config

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# 中性 prompt：只要求輸出看到的文字，不暗示語言或字集
OCR_PROMPT = (
    "請辨識圖片中的文字，逐字輸出。"
    "只回傳文字本身，不要加任何解釋、標籤、引號或格式標記。"
    "圖片中有什麼字就輸出什麼字，不要改寫、不要翻譯、不要糾正用字。"
)

MODELS = [
    {"key": "qwen3vl4b", "tag": "qwen3-vl:4b",
     "label": "Qwen3-VL 4B", "params": "4B", "kind": "general VLM"},
    {"key": "internvl35_4b", "tag": "internvl3.5:4b-q4_k_m",
     "label": "InternVL3.5 4B", "params": "4B", "kind": "general VLM"},
]

GEN_OPTIONS = {"temperature": 0.0, "num_predict": 96, "repeat_penalty": 1.15}


# ---------------------------------------------------------------- 後處理

FENCE = re.compile(r"```[a-zA-Z]*\n?|```")
THINK = re.compile(r"<think>.*?</think>", re.S)
TAGS = re.compile(r"</?(?:div|p|span|br|b|i|em|strong|html|body)\s*/?>", re.I)
LEAD = re.compile(
    r"^(?:圖片中的文字[是為:：]?|文字內容[是為:：]?|辨識結果[是為:：]?|"
    r"答案[是為:：]?|輸出[:：]?|結果[:：]?|The text (?:is|reads)[:：]?|Text[:：]?)\s*",
    re.I)


def postprocess(raw):
    """清掉格式雜訊，留下答案本身。不做任何模型專屬的特例處理。"""
    if not raw:
        return ""
    s = THINK.sub("", raw)
    s = FENCE.sub("", s)
    s = TAGS.sub("", s)
    s = s.strip()

    # JSON 包裝：["字"] 或 [{"text": "字"}]
    if s.startswith("[") or s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                parts = []
                for o in obj:
                    if isinstance(o, str):
                        parts.append(o)
                    elif isinstance(o, dict):
                        parts.append(str(o.get("text", o.get("content", ""))))
                s = "".join(parts)
            elif isinstance(obj, dict):
                s = str(obj.get("text", obj.get("content", s)))
        except Exception:
            pass

    s = LEAD.sub("", s.strip())
    s = s.strip().strip("「」『』\"'`　 ")
    # 模型有時會輸出多行；OCR 單行任務取所有非空行接起來
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    return "".join(lines)


def normalize(s):
    """評分前的正規化：NFKC（統一全半形）＋ 去所有空白。

    刻意不做繁簡轉換——把「係」正規化成「系」會直接抹掉本 benchmark 要測的東西。
    """
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", "", s)


# ---------------------------------------------------------------- 指標

def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(pred, gt):
    """回傳四個指標。pred/gt 都已正規化。"""
    d = levenshtein(pred, gt)
    cer = d / max(1, len(gt))
    ned = d / max(1, len(pred), len(gt))

    pc, gc = Counter(pred), Counter(gt)
    inter = sum((pc & gc).values())
    prec = inter / max(1, len(pred))
    rec = inter / max(1, len(gt))
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    return dict(cer=cer, cer_clipped=min(cer, 1.0), acc=1.0 if pred == gt else 0.0,
                ned_sim=1.0 - ned, f1=f1, edit_distance=d)


# ---------------------------------------------------------------- 推論

def call_ollama(tag, image_path, prompt, timeout=180):
    """呼叫 Ollama /api/generate，回傳 (文字, 延遲秒, 錯誤)。"""
    import urllib.request
    import urllib.error

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    payload = json.dumps({
        "model": tag, "prompt": prompt, "images": [b64],
        "stream": False, "options": GEN_OPTIONS,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_HOST + "/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
        return body.get("response", ""), time.time() - t0, None
    except Exception as e:
        return "", time.time() - t0, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- 主流程

FIELDS = ["model", "file", "tier", "role", "source", "difficulty", "plane",
          "font_family", "n_chars", "minimal_pair_with",
          "gt", "prediction_raw", "prediction",
          "cer", "cer_clipped", "acc", "ned_sim", "f1", "edit_distance",
          "latency_sec", "error"]


def load_benchmark(out_dir):
    path = os.path.join(out_dir, config.METADATA_FILE)
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_done(csv_path):
    """讀已完成的 (model, file)，支援續跑。"""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error"):        # 失敗的重跑
                continue
            done.add((row["model"], row["file"]))
    return done


def run(models, recs, out_dir, results_csv, limit=0, resume=True):
    img_dir = os.path.join(out_dir, config.IMAGES_SUBDIR)
    done = load_done(results_csv) if resume else set()
    new_file = not os.path.exists(results_csv) or not resume

    todo = []
    for m in models:
        sel = recs if not limit else recs[:limit]
        for r in sel:
            if (m["key"], r["file"]) not in done:
                todo.append((m, r))

    print(f"[eval] 待跑 {len(todo):,} 筆"
          f"（已完成 {len(done):,}，模型 {len(models)} 個）")
    if not todo:
        return

    mode = "w" if new_file else "a"
    with open(results_csv, mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()

        t0 = time.time()
        cur_model = None
        for i, (m, r) in enumerate(todo, 1):
            if m["key"] != cur_model:
                cur_model = m["key"]
                print(f"\n[eval] === {m['label']} ({m['tag']}) ===")

            raw, lat, err = call_ollama(m["tag"], os.path.join(img_dir, r["file"]),
                                        OCR_PROMPT)
            pred = postprocess(raw)
            s = score(normalize(pred), normalize(r["text"]))

            w.writerow({
                "model": m["key"], "file": r["file"], "tier": r["tier"],
                "role": r.get("role", ""), "source": r.get("source", ""),
                "difficulty": r.get("difficulty", ""), "plane": r.get("plane", ""),
                "font_family": r.get("font_family", ""), "n_chars": r.get("n_chars", 0),
                "minimal_pair_with": r.get("minimal_pair_with", ""),
                "gt": r["text"], "prediction_raw": raw, "prediction": pred,
                "latency_sec": round(lat, 3), "error": err or "",
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in s.items()},
            })
            f.flush()

            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / max(el, 1e-9)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"  {i:>6,}/{len(todo):,}  {rate:.2f} 張/秒  "
                      f"ETA {eta/3600:.1f} 小時", flush=True)


def main():
    ap = argparse.ArgumentParser(description="在 CantoBench 上評測 VLM")
    ap.add_argument("--out", default=config.OUT_DIR, help="benchmark 目錄")
    ap.add_argument("--results", default=None, help="結果 CSV 路徑")
    ap.add_argument("--limit", type=int, default=0, help="每個模型只跑前 N 張（0=全部）")
    ap.add_argument("--models", default="", help="逗號分隔的 model key，預設全部")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    results_csv = args.results or os.path.join(out_dir, "eval_results.csv")

    models = MODELS
    if args.models:
        keys = {k.strip() for k in args.models.split(",")}
        models = [m for m in MODELS if m["key"] in keys]
    if not models:
        sys.exit("沒有符合的模型")

    recs = load_benchmark(out_dir)
    print(f"[eval] benchmark {len(recs):,} 張  模型 "
          f"{', '.join(m['label'] for m in models)}")
    run(models, recs, out_dir, results_csv, limit=args.limit,
        resume=not args.no_resume)
    print(f"\n[eval] 結果寫入 {results_csv}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
