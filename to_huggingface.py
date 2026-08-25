# -*- coding: utf-8 -*-
"""把產出的資料集打包成 HuggingFace 格式，並產生雙語 dataset card。

為什麼用 Parquet 而不是散檔：

  HF 官方建議單張影像 <1MB 時用 Parquet，本資料集平均 8KB，正中。
  好處是 Dataset Viewer 直接可用、load_dataset() 秒開、不會有 5,000 個散檔。
  HF 的 repo 限制是「單 repo <100k 檔案、單資料夾 <10k 檔案」，散檔在
  訓練集規模（數十萬張）會直接違反，Parquet 沒有這個問題。

  影像欄位必須是 struct（bytes + path），並在 README 的 YAML 宣告
  dtype: image，Viewer 才認得。

用法：
    python to_huggingface.py                      # 產生 hf_export/
    python to_huggingface.py --push USER/REPO     # 產生後直接上傳
"""

import argparse
import io
import json
import os
import sys

import config

REPO_SLUG = "cantonese-benchmark-synth"
GH_REPO = "https://github.com/trickster-2005/Cantonese-OCR-Benchmark"
PAGE_URL = "https://trickster-2005.github.io/Cantonese-OCR-Benchmark/"


def load_records(out_dir):
    path = os.path.join(out_dir, config.METADATA_FILE)
    if not os.path.exists(path):
        sys.exit(f"找不到 {path}，請先執行 build_benchmark.py")
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_report(out_dir):
    path = os.path.join(out_dir, "build_report.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def load_eval(out_dir, expected_per_model=None):
    """讀評測結果並彙總。沒跑分就回傳 None，dataset card 會標成待補。

    刻意做成「必須明確要求才帶跑分」（--with-eval）：評測是數小時的工作，
    中途讀到只跑一半的 CSV 會把不完整的平均值寫進 dataset card，
    而那個數字看起來跟跑完的一模一樣，沒有任何地方會提示它是錯的。
    """
    import csv
    from collections import defaultdict

    path = os.path.join(out_dir, "eval_results.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if not r.get("error")]
    if not rows:
        return None

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = r["model"]
        for k in ("cer_clipped", "acc", "ned_sim", "f1"):
            agg[m][k].append(float(r[k]))
        agg[m]["tier_" + r["tier"]].append(float(r["acc"]))
        if r.get("plane") in ("ExtA", "ExtB+"):
            agg[m]["ext"].append(float(r["acc"]))
    out = {}
    for m, d in agg.items():
        out[m] = {k: (sum(v) / len(v) if v else None) for k, v in d.items()}
        out[m]["n"] = len(d["acc"])
    return out


# ---------------------------------------------------------------- Parquet

def build_parquet(recs, out_dir, dest):
    """把影像 bytes 與 metadata 寫成單一 Parquet。"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("需要 pyarrow：pip install pyarrow")

    img_dir = os.path.join(out_dir, config.IMAGES_SUBDIR)
    cols = {k: [] for k in (
        "text", "tier", "role", "source", "difficulty", "font", "font_family",
        "font_license", "plane", "corpus_freq", "charset_source",
        "minimal_pair_with", "jyutping", "canto_chars", "n_chars",
        "ink_ratio", "contrast", "width", "height", "file")}
    images = []

    for r in recs:
        with open(os.path.join(img_dir, r["file"]), "rb") as f:
            images.append({"bytes": f.read(), "path": r["file"]})
        for k in cols:
            v = r.get(k)
            if k in ("corpus_freq", "n_chars", "width", "height"):
                v = int(v) if v is not None else None
            elif k in ("ink_ratio", "contrast"):
                v = float(v) if v is not None else None
            elif v is None:
                v = ""
            cols[k].append(v)

    table = pa.table({
        "image": pa.array(images, type=pa.struct(
            [("bytes", pa.binary()), ("path", pa.string())])),
        **{k: pa.array(v) for k, v in cols.items()},
    })
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    pq.write_table(table, dest, compression="zstd")
    return os.path.getsize(dest), table.num_rows


# ---------------------------------------------------------------- Dataset card

def _fmt(v, pct=False):
    if v is None:
        return "TBD"
    return f"{v*100:.1f}%" if pct else f"{v:.4f}"


def build_card(recs, report, evals, repo_id):
    s = report.get("stats", {})
    cs = report.get("charset", {})
    fonts = report.get("fonts", {})
    tiers = s.get("by_tier", {})
    planes = s.get("by_plane", {})
    n = s.get("n_images", len(recs))
    size_cat = "1K<n<10K" if n < 10000 else "10K<n<100K"

    # 跑分表格
    if evals:
        name = {"qwen3vl4b": "Qwen3-VL 4B", "internvl35_4b": "InternVL3.5 4B"}
        rows_en, rows_zh = [], []
        for key, d in sorted(evals.items()):
            label = name.get(key, key)
            cells = (f"| {label} | {_fmt(d.get('cer_clipped'))} | "
                     f"{_fmt(d.get('acc'), True)} | {_fmt(d.get('ned_sim'))} | "
                     f"{_fmt(d.get('tier_char'), True)} | "
                     f"{_fmt(d.get('tier_sentence'), True)} | "
                     f"{_fmt(d.get('ext'), True)} |")
            rows_en.append(cells)
            rows_zh.append(cells)
        eval_note_en = ""
        eval_note_zh = ""
    else:
        rows_en = ["| Qwen3-VL 4B | TBD | TBD | TBD | TBD | TBD | TBD |",
                   "| InternVL3.5 4B | TBD | TBD | TBD | TBD | TBD | TBD |"]
        rows_zh = list(rows_en)
        eval_note_en = "\n> Evaluation is still running; the table will be filled in when it completes.\n"
        eval_note_zh = "\n> 評測仍在執行中，完成後會填入表格。\n"

    y = [
        "---",
        "license: cc-by-sa-4.0",
        "task_categories:",
        "  - image-to-text",
        "language:",
        "  - yue",
        "tags:",
        "  - ocr",
        "  - cantonese",
        "  - synthetic",
        "  - hong-kong",
        "  - low-resource",
        "size_categories:",
        f"  - {size_cat}",
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - split: test",
        "        path: data/test-*.parquet",
        "dataset_info:",
        "  features:",
        "    - name: image",
        "      dtype: image",
        "    - name: text",
        "      dtype: string",
        "    - name: tier",
        "      dtype: string",
        "    - name: role",
        "      dtype: string",
        "    - name: source",
        "      dtype: string",
        "    - name: difficulty",
        "      dtype: string",
        "    - name: font",
        "      dtype: string",
        "    - name: font_family",
        "      dtype: string",
        "    - name: font_license",
        "      dtype: string",
        "    - name: plane",
        "      dtype: string",
        "    - name: corpus_freq",
        "      dtype: int64",
        "    - name: charset_source",
        "      dtype: string",
        "    - name: minimal_pair_with",
        "      dtype: string",
        "    - name: jyutping",
        "      dtype: string",
        "    - name: canto_chars",
        "      dtype: string",
        "    - name: n_chars",
        "      dtype: int64",
        "    - name: ink_ratio",
        "      dtype: float64",
        "    - name: contrast",
        "      dtype: float64",
        "    - name: width",
        "      dtype: int64",
        "    - name: height",
        "      dtype: int64",
        "    - name: file",
        "      dtype: string",
        "  splits:",
        "    - name: test",
        f"      num_examples: {n}",
        "---",
        "",
    ]

    body = f"""# CantoBench-Synth

A synthetic OCR benchmark for written Cantonese (廣東話).
粵語（廣東話）OCR 合成評測資料集。

**[English](#english) · [繁體中文](#繁體中文)**

- Project page: [{PAGE_URL}]({PAGE_URL})
- Code: [{GH_REPO}]({GH_REPO})
- OCF AI Research Internship 2026

---

## English

### What this is

{n:,} synthetic text-line images testing whether vision-language models can read
the characters used in written Cantonese. Everyday writing in Hong Kong uses many
characters absent from Standard Written Chinese (嘅, 咗, 冇, 喺, 𨋢), and these
barely appear in mainstream OCR training data.

Surveying existing resources turned up three gaps: there are twenty-odd Cantonese
datasets on HuggingFace and all are text or audio, with none providing OCR data as
images; recent Cantonese benchmarks measure language understanding rather than
visual recognition; and some characters are missing from fonts entirely, with
`𠮶` (U+20BB6) supported by none of the 33 open-licensed font instances used here.

### Structure

Three tiers, with the same character deliberately appearing across them.

| Tier | Content | Images | Distinct texts | Length |
|---|---|---|---|---|
| `char` | Cantonese-specific characters | {tiers.get('char', 0):,} | 166 | 1 |
| `word` | Words containing Cantonese characters | {tiers.get('word', 0):,} | 600 | 2–4 |
| `sentence` | Real Cantonese sentences | {tiers.get('sentence', 0):,} | 667 | 4–25 |

> **The core design.** If a model reads `冇` correctly in isolation but returns
> `有` inside a sentence, the failure is not visual: the language-model prior
> overrode the glyph. A sentence-only benchmark cannot separate those two cases.

Fifteen minimal pairs are tagged via `minimal_pair_with`. The matching Standard
Chinese characters (有, 既, 左, 但 …) are included as `role: counterpart` so there
is a baseline to compare against.

Character tier by Unicode block: {planes.get('BMP', 0):,} basic,
{planes.get('ExtA', 0):,} Extension A, {planes.get('ExtB+', 0):,} Extension B+.
Extension blocks are over-represented on purpose: sampling by corpus frequency
would fill the benchmark with common characters and never show the rare ones.

### Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="test")
print(ds[0]["text"], ds[0]["tier"], ds[0]["difficulty"])

# Extension-block characters only
ext = ds.filter(lambda r: r["plane"] in ("ExtA", "ExtB+"))

# Minimal pairs, for the character-vs-sentence comparison
pairs = ds.filter(lambda r: r["minimal_pair_with"] != "")
```

### Baseline results
{eval_note_en}
| Model | CER ↓ | ACC ↑ | 1−NED ↑ | ACC char | ACC sentence | ACC ext. blocks |
|---|---|---|---|---|---|---|
{chr(10).join(rows_en)}

Metrics follow OCR convention: CER (character error rate), ACC (exact sequence
match), 1−NED (normalised edit similarity). The prompt deliberately does not
mention Traditional Chinese, since "Cantonese characters get corrected into
Standard Chinese" is the phenomenon under test.

### How it was built

Corpora: [Cantonese Wikipedia](https://huggingface.co/datasets/jed351/cantonese-wikipedia)
(CC BY-SA 4.0), [HKCanCor](https://docs.pycantonese.org/stable/data.html#hkcancor) (CC BY 4.0),
[rime-cantonese](https://github.com/rime/rime-cantonese) (CC BY 4.0).

The {cs.get('total', 0)}-character Cantonese set is derived by subtracting the
mainland, Taiwan and Hong Kong standard character lists plus Japanese-only kanji
from corpus character frequencies, then reviewed by hand.

Fonts are SIL OFL 1.1 only ({fonts.get('usable_instances', 0)} instances):
Chiron Hei HK, Chiron Sung HK, Noto CJK, Hong Kong Character Set. System Chinese
fonts are not used, as their licences do not permit generating a publicly
distributed dataset and their extension-block coverage is zero.

Augmentation covers rotation, perspective, elastic distortion, stroke weight,
blur, noise and JPEG artefacts across four difficulty levels
({s.get('by_difficulty', {})}).

Every image passes three checks against tofu characters: the font's cmap is
verified before rendering, ink ratio and contrast are verified after, and an
independent pass re-checks glyph coverage across the whole dataset.

Reproducible: each image's RNG derives from `(seed={report.get('seed')}, index)`,
independent of worker count.

Full methodology is in the
[GitHub README]({GH_REPO}).

### Licence

**CC BY-SA 4.0.** The dataset is copyleft because Cantonese Wikipedia is CC BY-SA
and that propagates; derivative works must use the same licence.

Please cite this dataset and the three upstream corpora.

Generation code is MIT. Fonts are SIL OFL 1.1 and are not redistributed here.

### Limitations

- All images are synthetic; performance on real Cantonese images needs separate
  validation.
- Horizontal text lines only. Hong Kong signage is frequently vertical.
- Handwriting is approximated by elastic distortion and stroke-weight variation
  rather than being genuine.
- `𠮶` (U+20BB6) and `𤗲` (U+245F2) are in the character set but produce no images
  because no available font contains them.

### Acknowledgement

Produced with assistance from Claude (Anthropic). All figures, methods and claims
were reviewed and verified by the author.

---

## 繁體中文

### 這是什麼

{n:,} 張合成文字行影像，用來測視覺語言模型認不認得粵語書寫特有的字。
香港的日常書寫大量使用標準中文沒有的字（嘅、咗、冇、喺、𨋢），
而這些字在主流 OCR 的訓練資料中極少出現。

盤點現有資源後發現三件事：HuggingFace 上的粵語資料集有二十多個，全部是純文字或語音，
未找到以影像形式提供的；近年的粵語基準測試評估的是語言理解而非視覺辨識；
部分字連字型都缺，`𠮶`（U+20BB6）在本專案使用的 33 個開源字型實例中支援數為 0。

### 結構

三層結構，同一個字刻意跨層出現。

| 層級 | 內容 | 張數 | 相異文字 | 字數 |
|---|---|---|---|---|
| `char` | 粵語特有字 | {tiers.get('char', 0):,} | 166 | 1 |
| `word` | 含粵語字的詞 | {tiers.get('word', 0):,} | 600 | 2–4 |
| `sentence` | 真實粵語句子 | {tiers.get('sentence', 0):,} | 667 | 4–25 |

> **核心設計。** 如果模型在單字層把 `冇` 認對、在句子層卻讀成 `有`，
> 那就不是視覺問題，而是語言模型的先驗蓋過了字形。
> 只測句子的評測分不出這兩種情況。

15 組最小對立對以 `minimal_pair_with` 標記。書面語對照字（有、既、左、但等）
以 `role: counterpart` 收錄，作為比較的基準線。

單字層 Unicode 分區：基本區 {planes.get('BMP', 0):,}、
擴展 A 區 {planes.get('ExtA', 0):,}、擴展 B 區以上 {planes.get('ExtB+', 0):,}。
擴展區刻意超額配置：按語料頻率抽樣的話，常用字會塞滿評測集，罕見字一次都不會出現。

### 使用方式

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="test")
print(ds[0]["text"], ds[0]["tier"], ds[0]["difficulty"])

# 只取擴展區字元
ext = ds.filter(lambda r: r["plane"] in ("ExtA", "ExtB+"))

# 只取最小對立對，做單字層與句子層的對照
pairs = ds.filter(lambda r: r["minimal_pair_with"] != "")
```

### 基線跑分
{eval_note_zh}
| 模型 | CER ↓ | ACC ↑ | 1−NED ↑ | 單字層 ACC | 短句層 ACC | 擴展區 ACC |
|---|---|---|---|---|---|---|
{chr(10).join(rows_zh)}

指標採 OCR 慣例：CER（字元錯誤率）、ACC（整句完全正確率）、1−NED（正規化編輯相似度）。
prompt 刻意不提「繁體中文」，因為「粵語字被糾正成書面中文」正是要測的現象。

### 製作方式

語料：[粵語維基百科](https://huggingface.co/datasets/jed351/cantonese-wikipedia)
（CC BY-SA 4.0）、[HKCanCor](https://docs.pycantonese.org/stable/data.html#hkcancor)（CC BY 4.0）、
[rime-cantonese](https://github.com/rime/rime-cantonese)（CC BY 4.0）。

{cs.get('total', 0)} 個粵語特有字的字表，是從語料字頻扣掉兩岸港的標準字表與日文專用漢字
推導而來，再經人工複核。

字型只用 SIL OFL 1.1（{fonts.get('usable_instances', 0)} 個實例）：昭源黑體、昭源宋體、
Noto CJK、香港民間字集。不使用系統中文字型，因為其授權不允許用於產生公開散布的資料集，
且擴展區覆蓋率為 0。

增強包含旋轉、透視、彈性形變、筆畫粗細、模糊、雜訊、JPEG 壓縮痕跡，
分四個難度等級（{s.get('by_difficulty', {})}）。

每張影像都通過三道防豆腐字檢查：渲染前驗證字型 cmap、渲染後驗證墨水比例與對比、
最後再對整份資料集獨立複查字形覆蓋。

可重現：每張圖的亂數由 `(seed={report.get('seed')}, index)` 推導，與 worker 數量無關。

完整方法論見 [GitHub README]({GH_REPO})。

### 授權

**CC BY-SA 4.0。** 因粵語維基百科是 CC BY-SA 且會傳染，本資料集受 copyleft 約束，
衍生作品須以相同授權釋出。

請同時引用本資料集與三個上游語料來源。

生成程式碼為 MIT。字型為 SIL OFL 1.1，不隨資料集散布。

### 限制

- 全部是合成影像，真實粵語影像上的表現需另外驗證。
- 只有橫排文字行。香港招牌大量使用直排。
- 手寫只以彈性形變與筆畫粗細變化近似，並非真實手寫。
- `𠮶`（U+20BB6）與 `𤗲`（U+245F2）在字表中但產不出影像，因為沒有可用字型收錄。

### 聲明

由 Claude（Anthropic）輔助產出，所有數據、方法與敘述均經作者人工審核確認。
"""
    return "\n".join(y) + body


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="打包成 HuggingFace 格式")
    ap.add_argument("--out", default=config.OUT_DIR, help="benchmark 目錄")
    ap.add_argument("--dest", default=None, help="匯出目錄，預設 <out>/../hf_export")
    ap.add_argument("--repo", default=f"trickster-2005/{REPO_SLUG}",
                    help="HF repo id，只用於 dataset card 內的範例")
    ap.add_argument("--push", default=None,
                    help="直接上傳到這個 repo id（需要 huggingface_hub 與登入）")
    ap.add_argument("--with-eval", action="store_true",
                    help="把 eval_results.csv 的分數寫進 dataset card。"
                         "評測沒跑完就加這個旗標，會寫進不完整的平均值")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    dest = os.path.abspath(args.dest or os.path.join(out_dir, os.pardir, "hf_export"))
    repo_id = args.push or args.repo

    recs = load_records(out_dir)
    report = load_report(out_dir)
    evals = load_eval(out_dir) if args.with_eval else None
    if evals:
        note = "，含評測結果"
    elif os.path.exists(os.path.join(out_dir, "eval_results.csv")):
        note = "，跑分結果存在但未帶入（要帶請加 --with-eval），card 標 TBD"
    else:
        note = "，尚無評測結果（card 會標 TBD）"
    print(f"[hf] 讀入 {len(recs):,} 筆{note}")

    pq_path = os.path.join(dest, "data", "test-00000-of-00001.parquet")
    size, rows = build_parquet(recs, out_dir, pq_path)
    print(f"[hf] Parquet {rows:,} 列，{size/1024/1024:.1f} MB → {pq_path}")

    card = build_card(recs, report, evals, repo_id)
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(f"[hf] dataset card → {os.path.join(dest, 'README.md')}")

    # 授權檔一併帶上，讓 repo 自帶完整授權資訊
    src_lic = os.path.join(config.ROOT, "DATA_LICENSE.md")
    if os.path.exists(src_lic):
        import shutil
        shutil.copy2(src_lic, os.path.join(dest, "DATA_LICENSE.md"))

    if args.push:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            sys.exit("需要 huggingface_hub：pip install huggingface_hub")
        api = HfApi()
        api.create_repo(args.push, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=dest, repo_id=args.push, repo_type="dataset")
        print(f"[hf] 已上傳到 https://huggingface.co/datasets/{args.push}")
    else:
        print(f"\n檢查 {dest} 之後，用下列指令上傳：")
        print(f"  python to_huggingface.py --push <你的帳號>/{REPO_SLUG}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
