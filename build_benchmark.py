# -*- coding: utf-8 -*-
"""CantoBench 主程式：產生粵語 OCR 評測資料集。

三層結構，刻意讓同一個字同時出現在不同層：

    char      單一粵語特有字        測純字形辨識，沒有上下文可以依賴
    word      2-4 字粵語詞彙        測詞彙層
    sentence  4-25 字粵語短句        測真實情境，語言模型先驗會介入

這個設計是整個 benchmark 的核心實驗。同一個字（例如「冇」）在 char 層辨對、
在 sentence 層卻被辨成「有」，就直接證明錯誤來自語言模型先驗覆蓋視覺證據，
而不是看不清楚字形。單一層的 benchmark 無法區分這兩件事。

用法：
    python build_benchmark.py                    # 用 config.py 的預設規模
    python build_benchmark.py --total 500        # 快速試產，先看品質
    python build_benchmark.py --workers 8
    python build_benchmark.py --out /mnt/data/cantobench

產出（刻意對齊 TC-STR 格式，可直接餵給既有的評測管線）：
    output/images/*.jpg
    output/test_labels.txt      filename<TAB>label
    output/metadata.jsonl       每張圖的完整 metadata
    output/charset_review.tsv   字表人工複核檔
    output/dataset_card.md      HF dataset card
    output/build_report.json    產出統計與可重現性資訊
"""

import argparse
import collections
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import cv2

import config
import charset as charset_mod
import corpora
import fonts as fonts_mod
import render

# 每個 worker 各自持有一份字型池；用全域變數是為了避免每個任務都重新解析 cmap
_POOL = None
_POOL_BY_KEY = None


def _init_worker():
    global _POOL, _POOL_BY_KEY
    _POOL, _ = fonts_mod.load_font_pool(verbose=False)
    _POOL_BY_KEY = {e.key: e for e in _POOL}


def _render_task(task):
    """在 worker 中執行：算出一張圖並存檔。回傳 metadata 或錯誤。"""
    idx, text, tier, source, font_key, difficulty, out_dir, extra = task
    entry = _POOL_BY_KEY.get(font_key)
    if entry is None:
        return dict(ok=False, index=idx, text=text, error=f"字型不存在: {font_key}")

    img, meta = render.make_sample(text, entry, idx, difficulty=difficulty)
    if img is None:
        return dict(ok=False, index=idx, text=text, font=font_key,
                    error=meta.get("error", "未知"))

    fn = f"{tier}_{idx:06d}.jpg"
    path = os.path.join(out_dir, config.IMAGES_SUBDIR, fn)
    ok = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    if not ok:
        return dict(ok=False, index=idx, text=text, error="存檔失敗")

    rec = dict(ok=True, index=idx, file=fn, text=text, tier=tier, source=source,
               font=font_key, font_family=entry.family, font_license=entry.license,
               difficulty=meta.get("difficulty"), ink_ratio=meta.get("ink_ratio"),
               width=meta["size"][0], height=meta["size"][1],
               n_chars=len(text))
    rec.update(extra)
    return rec


# ---------------------------------------------------------------- 抽樣計畫

def _eligible_fonts(text, pool):
    """能完整畫出 text 的字型。這是防豆腐字的第一道防線。"""
    return [e for e in pool if e.covers(text)]


def _pick_fonts(text, pool, rng, n):
    """挑 n 個能畫出 text 的字型，盡量不重複、盡量跨家族。"""
    ok = _eligible_fonts(text, pool)
    if not ok:
        return []
    # 先按家族分組輪流取，讓同一個文字的多次渲染字型差異最大
    by_family = collections.defaultdict(list)
    for e in ok:
        by_family[e.family].append(e)
    families = sorted(by_family)
    rng.shuffle(families)
    picked, i = [], 0
    while len(picked) < n:
        fam = families[i % len(families)]
        cands = [e for e in by_family[fam] if e not in picked]
        if cands:
            picked.append(rng.choice(cands))
        elif len(picked) >= len(ok):
            break
        i += 1
        if i > len(families) * 8:
            break
    return picked[:n]


def plan_char_tier(charset, pool, rng, budget):
    """單字層。

    刻意「均勻」分配預算給每個字，而不是按語料頻率——長尾正是重點。
    按頻率抽的話 係/嘅 會塞滿整個 benchmark，而 𨋢/𠝹 一次都不出現，
    但後者才是真正能區分模型的樣本。
    """
    chars = sorted(charset)
    pairs = charset_mod.minimal_pair_lookup()

    # 最小對立對的「書面語對照字」也要收：要知道模型對 有/既/左 的基線正確率，
    # 才能判斷把 冇 讀成 有 是視覺問題還是語言模型先驗問題
    counterparts = sorted({o for _, (o, _) in pairs.items()})

    items = [(c, "cantonese") for c in chars] + \
            [(c, "counterpart") for c in counterparts]
    if not items:
        return []

    # 每個字至少一張，即使超出預算。benchmark 的價值在於涵蓋完整字表——
    # 少了某個字就等於那個字沒被測到。小預算試產時 char 層因此會超額，
    # 這是刻意的；正式規模（1200/167 ≈ 7 張每字）不會觸發。
    per = max(1, budget // len(items))
    if per * len(items) > budget:
        print(f"[plan] char 層保底每字 1 張：{len(items)} 個字，"
              f"超出預算 {budget} → 實際 {per * len(items)} 張")
    plan = []
    for text, role in items:
        d = charset.get(text, {})
        extra = dict(role=role,
                     plane=d.get("plane", charset_mod.char_plane(text)),
                     corpus_freq=d.get("freq", 0),
                     charset_source=d.get("source", "counterpart"))
        if text in pairs:
            extra["minimal_pair_with"] = pairs[text][0]
            extra["minimal_pair_note"] = pairs[text][1]
        for f in _pick_fonts(text, pool, rng, per):
            plan.append((text, "char", "charset", f.key, None, extra))
    return plan


def plan_word_tier(bundle, charset, pool, rng, budget):
    """詞彙層。只收「含至少一個粵語特有字」的詞——不然只是普通中文詞。"""
    cs = set(charset)
    cands = [(w, jp, wt) for w, jp, wt in bundle["rime_words"]
             if any(c in cs for c in w)]
    if not cands:
        return []
    rng.shuffle(cands)

    n_items = max(1, budget // config.RENDERS_PER_ITEM)
    plan = []
    for w, jp, wt in cands[:n_items * 2]:      # 多取一些，字型不足時可遞補
        if len(plan) >= budget:
            break
        extra = dict(role="word", jyutping=jp, rime_weight=wt,
                     canto_chars="".join(c for c in w if c in cs))
        for f in _pick_fonts(w, pool, rng, config.RENDERS_PER_ITEM):
            plan.append((w, "word", "rime-cantonese", f.key, None, extra))
    return plan[:budget]


def plan_sentence_tier(bundle, charset, pool, rng, budget):
    """句子層。只收含粵語特有字的句子，並平衡兩個語料來源。"""
    cs = set(charset)
    by_source = collections.defaultdict(list)
    for s, src in bundle["sentences"]:
        hits = [c for c in s if c in cs]
        if hits:
            by_source[src].append((s, src, hits))
    if not by_source:
        return []

    # 兩個來源各半，避免 Wikipedia 的量壓過 HKCanCor
    sources = sorted(by_source)
    share = budget // max(1, len(sources))
    plan = []
    for src in sources:
        pool_s = by_source[src]
        rng.shuffle(pool_s)
        n_items = max(1, share // config.RENDERS_PER_ITEM)
        taken = 0
        for s, source, hits in pool_s:
            if taken >= share:
                break
            extra = dict(role="sentence", canto_chars="".join(sorted(set(hits))),
                         n_canto_chars=len(set(hits)))
            picked = _pick_fonts(s, pool, rng, config.RENDERS_PER_ITEM)
            for f in picked:
                plan.append((s, "sentence", source, f.key, None, extra))
                taken += 1
            if len(plan) >= budget:
                break
    return plan[:budget]


def report_unrenderable(charset, pool):
    """找出「字表裡有、但沒有任何字型畫得出來」的字。

    這不是錯誤，是本專案要證明的核心論點之一：有些粵語字連開源字型都沒有收，
    模型當然學不會、也辨不出。實測 𠮶（U+20BB6）在 33 個字型實例中支援度是 0。

    必須明確報告而不是靜默丟棄——靜默丟棄的話，字表說有 154 字、實際只產出
    150 字，兩邊對不上又沒人知道為什麼。
    """
    missing = []
    for c in sorted(charset):
        n = sum(1 for e in pool if ord(c) in e.cmap)
        if n == 0:
            missing.append(dict(char=c, codepoint=f"U+{ord(c):05X}",
                                plane=charset_mod.char_plane(c),
                                corpus_freq=charset[c].get("freq", 0)))
    if missing:
        print(f"[plan] 有 {len(missing)} 個粵語字沒有任何字型支援，無法產生樣本：")
        for m in missing:
            print(f"        {m['char']}  {m['codepoint']}  {m['plane']}  "
                  f"語料出現 {m['corpus_freq']} 次")
        print("        （這本身就是資料點：連開源字型都缺的字，OCR 模型不可能辨得出）")
    return missing


def build_plan(bundle, charset, pool, sizes, seed):
    """組出完整抽樣計畫，並替每個樣本決定難度。"""
    rng = random.Random(seed)
    plan = []
    plan += plan_char_tier(charset, pool, rng, sizes["char"])
    plan += plan_word_tier(bundle, charset, pool, rng, sizes["word"])
    plan += plan_sentence_tier(bundle, charset, pool, rng, sizes["sentence"])

    # 難度在這裡決定（而不是在 worker 裡），確保難度分布可重現且可統計
    levels = list(config.DIFFICULTY_WEIGHTS)
    weights = [config.DIFFICULTY_WEIGHTS[k] for k in levels]
    out = []
    for i, (text, tier, source, font_key, _d, extra) in enumerate(plan):
        d = rng.choices(levels, weights=weights, k=1)[0]
        out.append((text, tier, source, font_key, d, extra))
    return out


# ---------------------------------------------------------------- 產出

def write_dataset_card(out_dir, bundle, charset_info, font_report, stats):
    """產生 HF dataset card。授權區塊是重點，不要手改成別的授權。"""
    lic = bundle["licenses"]
    lines = [
        "---",
        f"license: {'cc-by-sa-4.0' if config.DATASET_LICENSE.startswith('CC BY-SA') else 'cc-by-4.0'}",
        "task_categories:",
        "  - image-to-text",
        "language:",
        "  - yue",
        "tags:",
        "  - ocr",
        "  - cantonese",
        "  - synthetic",
        "  - hong-kong",
        "size_categories:",
        f"  - {'1K<n<10K' if stats['n_images'] < 10000 else '10K<n<100K'}",
        "---",
        "",
        f"# {config.DATASET_NAME} v{config.DATASET_VERSION}",
        "",
        "粵語（廣東話）OCR 評測資料集。目前 HuggingFace 上有 20 個以上的粵語資料集，",
        "全部是純文字或語音——這是第一個影像／OCR 的粵語資料集。",
        "",
        "## 為什麼需要這個資料集",
        "",
        "粵語書寫使用大量標準中文沒有的字（嘅、咗、冇、喺、𨋢），這些字在主流 OCR",
        "訓練資料中極少出現，而且部分連字型都缺。更麻煩的是其中不少與常用字形近但",
        "語義相反——`冇`（沒有）與 `有` 只差兩筆，辨錯會讓整句意思顛倒。",
        "",
        "## 三層結構",
        "",
        "| 層 | 內容 | 用途 |",
        "|---|---|---|",
        "| `char` | 單一粵語特有字 | 純字形辨識，無上下文可依賴 |",
        "| `word` | 2-4 字粵語詞彙 | 詞彙層 |",
        "| `sentence` | 4-25 字粵語短句 | 真實情境，語言模型先驗會介入 |",
        "",
        "同一個字會同時出現在不同層。**在 `char` 層辨對、在 `sentence` 層卻辨成對應的",
        "書面語字，即證明錯誤來自語言模型先驗覆蓋視覺證據，而非字形辨識失敗。**",
        "資料集另附最小對立對標記（`minimal_pair_with`）供此分析使用。",
        "",
        "## 統計",
        "",
        f"- 影像數：{stats['n_images']:,}",
        f"- 粵語特有字：{charset_info['total']:,}"
        f"（BMP {charset_info['by_plane'].get('BMP', 0)}、"
        f"Ext-A {charset_info['by_plane'].get('ExtA', 0)}、"
        f"Ext-B+ {charset_info['by_plane'].get('ExtB+', 0)}）",
        f"- 字型：{font_report['usable_instances']} 個實例／"
        f"{len(font_report['families'])} 個家族，全部 OFL-1.1",
        f"- 難度分布：{stats['by_difficulty']}",
        f"- 各層數量：{stats['by_tier']}",
        "",
        "## 授權",
        "",
        f"**{config.DATASET_LICENSE}**",
        "",
        "本資料集由下列語料衍生。因含 CC BY-SA 授權來源，整體受 copyleft 約束：",
        "",
        "| 來源 | 授權 | 出處 |",
        "|---|---|---|",
    ]
    for l in lic:
        lines.append(f"| {l['source']} | {l['license']} | {l['attribution']} |")
    lines += [
        "",
        "字型為 SIL OFL 1.1。OFL 允許散布用該字型製作的圖片，字型檔本身不隨資料集散布。",
        "",
        "| 字型家族 | 授權 |",
        "|---|---|",
    ]
    for src in config.FONT_SOURCES:
        lines.append(f"| {src['name']} | {src['license']} ({src['home']}) |")
    lines += [
        "",
        f"生成程式碼另以 {config.CODE_LICENSE} 授權。",
        "",
        "## 可重現性",
        "",
        f"- master seed：`{config.MASTER_SEED}`",
        "- 每張圖的隨機數由 `(seed, index)` 推導，與 worker 數量無關",
        "- 相同 seed ＋ 相同語料快照 ＝ 逐位元相同的產出",
        "",
        "## 已知限制",
        "",
        "- 全部為合成影像。真實粵語影像的表現需另行驗證。",
        "- 僅橫排文字行。香港常見的直排招牌未涵蓋。",
        "- 手寫僅以彈性形變與筆畫粗細變化近似，非真實手寫（粵語手寫字型幾乎不存在）。",
        "",
    ]
    path = os.path.join(out_dir, "dataset_card.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser(description="產生粵語 OCR 評測資料集")
    ap.add_argument("--out", default=config.OUT_DIR, help="輸出目錄")
    ap.add_argument("--total", type=int, default=None,
                    help="總張數；會依 config.TIER_SIZES 的比例分配。預設用 config 的絕對值")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=config.MASTER_SEED)
    ap.add_argument("--plan-only", action="store_true", help="只列印計畫，不產圖")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, config.IMAGES_SUBDIR), exist_ok=True)

    # 1. 語料
    print("=" * 62)
    print("1/5  載入語料")
    print("=" * 62)
    bundle = corpora.build_corpus_bundle()

    # 2. 字表
    print("\n" + "=" * 62)
    print("2/5  推導粵語特有字")
    print("=" * 62)
    cs, cs_info = charset_mod.derive_cantonese_charset(bundle["char_freq"])
    print(f"[charset] 粵語特有字 {cs_info['total']:,}（{cs_info['by_plane']}）")
    print(f"[charset] 濾除 {cs_info['filtered_out']}")
    charset_mod.write_review_file(
        cs, bundle["sentences"], os.path.join(out_dir, config.CHARSET_REVIEW_FILE))

    # 3. 字型
    print("\n" + "=" * 62)
    print("3/5  載入字型")
    print("=" * 62)
    pool, font_report = fonts_mod.load_font_pool(verbose=True)

    # 4. 計畫
    print("\n" + "=" * 62)
    print("4/5  建立抽樣計畫")
    print("=" * 62)
    sizes = dict(config.TIER_SIZES)
    if args.total:
        total = sum(sizes.values())
        sizes = {k: max(1, round(v * args.total / total)) for k, v in sizes.items()}
    print(f"[plan] 目標張數 {sizes}")

    unrenderable = report_unrenderable(cs, pool)
    plan = build_plan(bundle, cs, pool, sizes, args.seed)
    by_tier = collections.Counter(p[1] for p in plan)
    by_diff = collections.Counter(p[4] for p in plan)
    print(f"[plan] 實際 {len(plan):,} 張  各層 {dict(by_tier)}")
    print(f"[plan] 難度 {dict(by_diff)}")

    if args.plan_only:
        for text, tier, source, fk, d, extra in plan[:20]:
            print(f"  {tier:9} {d:7} {fk:32} {text[:28]}")
        return

    # 5. 產圖
    print("\n" + "=" * 62)
    print(f"5/5  產生影像（{args.workers} workers）")
    print("=" * 62)
    tasks = [(i, text, tier, source, fk, d, out_dir, extra)
             for i, (text, tier, source, fk, d, extra) in enumerate(plan)]

    records, failures = [], []
    done = 0
    with Pool(args.workers, initializer=_init_worker) as pool_p:
        for rec in pool_p.imap_unordered(_render_task, tasks, chunksize=32):
            done += 1
            if rec.get("ok"):
                records.append(rec)
            else:
                failures.append(rec)
            if done % 500 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done:>7,}/{len(tasks):,}  成功 {len(records):,}  "
                      f"失敗 {len(failures):,}  {done/max(el,1e-9):.0f} 張/秒")

    records.sort(key=lambda r: r["index"])

    # 寫標註檔（TC-STR 格式：檔名 TAB 標籤）
    labels_path = os.path.join(out_dir, config.LABELS_FILE)
    with open(labels_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r['file']}\t{r['text']}\n")

    # 寫完整 metadata
    meta_path = os.path.join(out_dir, config.METADATA_FILE)
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in records:
            d = {k: v for k, v in r.items() if k not in ("ok", "index")}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    stats = dict(
        n_images=len(records),
        n_failed=len(failures),
        by_tier=dict(collections.Counter(r["tier"] for r in records)),
        by_difficulty=dict(collections.Counter(r["difficulty"] for r in records)),
        by_source=dict(collections.Counter(r["source"] for r in records)),
        by_plane=dict(collections.Counter(r.get("plane", "-") for r in records
                                          if r["tier"] == "char")),
        n_distinct_texts=len({r["text"] for r in records}),
        n_fonts_used=len({r["font"] for r in records}),
        n_unrenderable_chars=len(unrenderable),
        elapsed_sec=round(time.time() - t0, 1),
    )

    card = write_dataset_card(out_dir, bundle, cs_info, font_report, stats)

    report = dict(
        dataset=config.DATASET_NAME, version=config.DATASET_VERSION,
        license=config.DATASET_LICENSE, seed=args.seed,
        stats=stats, charset=cs_info, unrenderable_chars=unrenderable,
        fonts=font_report, corpora=bundle["licenses"],
        image_height=config.IMAGE_HEIGHT,
        difficulty_weights=config.DIFFICULTY_WEIGHTS,
        failures=failures[:200],
    )
    with open(os.path.join(out_dir, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print(f"完成：{len(records):,} 張，失敗 {len(failures):,}，"
          f"耗時 {stats['elapsed_sec']:.0f} 秒")
    print("=" * 62)
    print(f"  影像      {os.path.join(out_dir, config.IMAGES_SUBDIR)}")
    print(f"  標註      {labels_path}")
    print(f"  metadata  {meta_path}")
    print(f"  card      {card}")
    print(f"  字表複核  {os.path.join(out_dir, config.CHARSET_REVIEW_FILE)}")
    if failures:
        print(f"\n前 5 個失敗原因：")
        for r in failures[:5]:
            print(f"  {r.get('text','')[:20]:22} {r.get('error')}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
