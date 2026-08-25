# -*- coding: utf-8 -*-
"""產出後的品質檢查。

合成資料集最大的風險是「看起來有東西、其實是錯的」，尤其是豆腐字——
字型缺該碼位時畫出空白方框，但標註檔仍寫著原本的字，等於汙染 ground truth。
產完一定要跑這支，並且看一眼 contact_sheet.jpg。

檢查項目：
    1. 標註與影像一一對應，沒有孤兒檔
    2. 影像可讀、尺寸合理
    3. 墨水比例分布——過低的就是豆腐字或空白
    4. 標註中的每個字，渲染它的字型是否真的有該字形
    5. 字元覆蓋與長尾分布——罕見字有沒有真的被產出來
    6. 難度／層級／字型分布是否符合設定
    7. 產出 contact_sheet.jpg 供人眼抽查

用法：
    python verify.py                      # 檢查 config.OUT_DIR
    python verify.py --out /mnt/data/cb   # 指定目錄
    python verify.py --sheet-rows 24      # 加大抽查表
"""

import argparse
import collections
import json
import os
import random
import sys

import numpy as np
import cv2

import config
import charset as charset_mod


def _load(out_dir):
    meta_path = os.path.join(out_dir, config.METADATA_FILE)
    if not os.path.exists(meta_path):
        sys.exit(f"找不到 {meta_path}，請先執行 build_benchmark.py")
    recs = [json.loads(l) for l in open(meta_path, encoding="utf-8") if l.strip()]
    labels = {}
    lp = os.path.join(out_dir, config.LABELS_FILE)
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8"):
            if "\t" in line:
                fn, txt = line.rstrip("\n").split("\t", 1)
                labels[fn] = txt
    return recs, labels


def check_pairing(recs, labels, out_dir):
    """標註、metadata、實體檔案三者要一致。"""
    img_dir = os.path.join(out_dir, config.IMAGES_SUBDIR)
    on_disk = {f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")}
    in_meta = {r["file"] for r in recs}

    problems = []
    if in_meta - on_disk:
        problems.append(f"metadata 有但磁碟沒有的檔案：{len(in_meta - on_disk)}")
    if on_disk - in_meta:
        problems.append(f"磁碟有但 metadata 沒有的孤兒檔：{len(on_disk - in_meta)}")
    mism = [r["file"] for r in recs
            if r["file"] in labels and labels[r["file"]] != r["text"]]
    if mism:
        problems.append(f"標註與 metadata 不一致：{len(mism)}")
    if set(labels) != in_meta:
        problems.append(f"標註檔與 metadata 的檔名集合不同："
                        f"{len(set(labels) ^ in_meta)} 個差異")

    print(f"[1] 配對檢查    影像 {len(on_disk):,}  metadata {len(recs):,}  "
          f"標註 {len(labels):,}")
    for p in problems:
        print(f"    !! {p}")
    if not problems:
        print("    三者完全一致")
    return problems


def check_images(recs, out_dir, sample=400):
    """抽樣讀圖，確認可解碼、尺寸合理、墨水比例正常。"""
    img_dir = os.path.join(out_dir, config.IMAGES_SUBDIR)
    rng = random.Random(config.MASTER_SEED)
    picks = recs if len(recs) <= sample else rng.sample(recs, sample)

    # 刻意直接呼叫 render 的實作，不要在這裡重寫一份：
    # 兩份指標一旦漂移，verify 就會對合格的樣本誤報，反而掩蓋真問題。
    import render

    bad_read, bad_size, low_ink, low_con = [], [], [], []
    ratios, contrasts = [], []
    for r in picks:
        p = os.path.join(img_dir, r["file"])
        img = cv2.imread(p)
        if img is None:
            bad_read.append(r["file"])
            continue
        h, w = img.shape[:2]
        if h != config.IMAGE_HEIGHT or w < 8 or w > 20000:
            bad_size.append((r["file"], w, h))
        ratio = render.ink_ratio(img)
        con = render.gray_contrast(img)
        ratios.append(ratio)
        contrasts.append(con)
        if ratio < config.MIN_INK_RATIO:
            low_ink.append((r["file"], r["text"], round(ratio, 4)))
        if con < config.MIN_CONTRAST:
            low_con.append((r["file"], r["text"], round(con, 1)))

    ratios = np.array(ratios) if ratios else np.array([0.0])
    contrasts = np.array(contrasts) if contrasts else np.array([0.0])
    print(f"[2] 影像檢查    抽樣 {len(picks):,} 張")
    print(f"    墨水比例  中位 {np.median(ratios):.3f}  "
          f"1% 分位 {np.percentile(ratios, 1):.4f}  最低 {ratios.min():.4f}"
          f"  （門檻 {config.MIN_INK_RATIO}）")
    print(f"    對比      中位 {np.median(contrasts):.0f}  "
          f"1% 分位 {np.percentile(contrasts, 1):.0f}  最低 {contrasts.min():.0f}"
          f"  （門檻 {config.MIN_CONTRAST}）")
    if bad_read:
        print(f"    !! 無法解碼 {len(bad_read)}：{bad_read[:3]}")
    if bad_size:
        print(f"    !! 尺寸異常 {len(bad_size)}：{bad_size[:3]}")
    if low_ink:
        print(f"    !! 墨水低於門檻 {len(low_ink)}：{low_ink[:3]}")
    if low_con:
        print(f"    !! 對比低於門檻 {len(low_con)}：{low_con[:3]}")
    if not (bad_read or bad_size or low_ink or low_con):
        print("    全部通過守門門檻")
    return bad_read, bad_size, low_ink + low_con


def check_glyph_coverage(recs, out_dir):
    """最重要的一項：確認渲染每張圖的字型，真的有標註中每個字的字形。

    build 時已經擋過一次，這裡是獨立複查——如果這裡出問題，
    代表 fonts.py 的 cmap 檢查有漏洞，整份資料集的 ground truth 都要重驗。
    """
    import fonts as fonts_mod
    try:
        pool, _ = fonts_mod.load_font_pool(verbose=False)
    except Exception as e:
        print(f"[4] 字形覆蓋    略過（無法載入字型池：{e}）")
        return []
    by_key = {e.key: e for e in pool}

    bad = []
    for r in recs:
        e = by_key.get(r.get("font"))
        if e is None:
            continue
        miss = e.missing(r["text"])
        if miss:
            bad.append((r["file"], r["text"], "".join(miss), r["font"]))

    print(f"[4] 字形覆蓋    檢查 {len(recs):,} 筆")
    if bad:
        print(f"    !! {len(bad)} 張的字型缺字形（ground truth 已汙染）")
        for b in bad[:5]:
            print(f"       {b[0]}  文字={b[1][:16]}  缺={b[2]}  字型={b[3]}")
    else:
        print("    每張圖的字型都涵蓋其標註的所有字元")
    return bad


def check_distribution(recs):
    """分布檢查：層級、難度、字型、字元長尾。"""
    print(f"[5] 分布檢查")
    for field in ("tier", "difficulty", "source", "role"):
        c = collections.Counter(r.get(field) for r in recs if r.get(field))
        if c:
            total = sum(c.values())
            parts = "  ".join(f"{k}={v:,}({v/total:.0%})"
                              for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
            print(f"    {field:11} {parts}")

    fam = collections.Counter(r.get("font_family") for r in recs)
    print(f"    字型家族    {len(fam)} 個，最多 {fam.most_common(1)[0][1]:,} 張／"
          f"最少 {min(fam.values()):,} 張")

    # 字元覆蓋：每個粵語字實際出現在多少張圖裡
    char_imgs = collections.Counter()
    for r in recs:
        for c in set(r["text"]):
            char_imgs[c] += 1
    ext = {c: n for c, n in char_imgs.items()
           if charset_mod.char_plane(c) in ("ExtA", "ExtB+")}
    print(f"    相異字元    {len(char_imgs):,}（其中擴展區 {len(ext)}）")
    if ext:
        print(f"    擴展區字元出現張數：" +
              "  ".join(f"{c}={n}" for c, n in sorted(ext.items(),
                                                      key=lambda kv: -kv[1])[:12]))
    rare = [c for c, n in char_imgs.items() if n < 3]
    if rare:
        print(f"    !! 出現少於 3 張的字元 {len(rare)} 個："
              f"{''.join(rare[:24])}")
    return char_imgs


def check_minimal_pairs(recs):
    """最小對立對的兩邊都要有足夠樣本，實驗才成立。"""
    pairs = charset_mod.minimal_pair_lookup()
    by_text = collections.Counter(r["text"] for r in recs if r["tier"] == "char")
    print(f"[6] 最小對立對  {len(pairs)} 組")
    weak = []
    for canto, (other, _why) in sorted(pairs.items()):
        a, b = by_text.get(canto, 0), by_text.get(other, 0)
        if a < 3 or b < 3:
            weak.append((canto, other, a, b))
    if weak:
        print(f"    !! {len(weak)} 組樣本不足（char 層任一邊少於 3 張）：")
        for canto, other, a, b in weak[:8]:
            print(f"       {canto}({a}) vs {other}({b})")
    else:
        print("    每組兩邊在 char 層都有 3 張以上樣本")
    return weak


def contact_sheet(recs, out_dir, rows=16, cols=4, seed=None):
    """把隨機抽樣的圖拼成一張大圖，供人眼快速抽查。

    自動化檢查抓不到「字型雖然有字形、但字形長得怪」這類問題，
    所以人眼看一眼是必要的。
    """
    img_dir = os.path.join(out_dir, config.IMAGES_SUBDIR)
    rng = random.Random(seed if seed is not None else config.MASTER_SEED)

    # 各層各難度都抽到，不要只抽到最多的那類
    buckets = collections.defaultdict(list)
    for r in recs:
        buckets[(r.get("tier"), r.get("difficulty"))].append(r)
    keys = sorted(buckets)
    picks = []
    n = rows * cols
    while len(picks) < n and keys:
        for k in keys:
            if buckets[k] and len(picks) < n:
                picks.append(buckets[k].pop(rng.randrange(len(buckets[k]))))
        keys = [k for k in keys if buckets[k]]

    cell_h = config.IMAGE_HEIGHT + 22
    cell_w = 520
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for i, r in enumerate(picks[:n]):
        img = cv2.imread(os.path.join(img_dir, r["file"]))
        if img is None:
            continue
        ry, cx = divmod(i, cols)
        y0, x0 = ry * cell_h, cx * cell_w
        h, w = img.shape[:2]
        # 太寬的整句要等比縮小，不能直接裁掉——裁掉的話複核時看不到句尾，
        # 而句尾正是最容易出問題的地方（增強的邊界效應）
        if w > cell_w - 8:
            scale = (cell_w - 8) / w
            img = cv2.resize(img, (cell_w - 8, max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        sheet[y0 + 20:y0 + 20 + h, x0 + 4:x0 + 4 + w] = img
        cv2.putText(sheet, f"{r['tier'][:4]}/{r['difficulty'][:5]}",
                    (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (90, 90, 90), 1, cv2.LINE_AA)

    path = os.path.join(out_dir, "contact_sheet.jpg")
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[7] 抽查表      {path}（{len(picks[:n])} 張，"
          f"標籤為 層級/難度，中文標註請對照 metadata）")
    return path


def main():
    ap = argparse.ArgumentParser(description="檢查產出的資料集品質")
    ap.add_argument("--out", default=config.OUT_DIR)
    ap.add_argument("--sample", type=int, default=400, help="影像抽樣檢查張數")
    ap.add_argument("--sheet-rows", type=int, default=16)
    ap.add_argument("--skip-glyph", action="store_true",
                    help="略過字形覆蓋複查（需要載入全部字型，較慢）")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    recs, labels = _load(out_dir)
    print(f"檢查 {out_dir}\n{'=' * 62}")

    problems = check_pairing(recs, labels, out_dir)
    bad_read, bad_size, low_ink = check_images(recs, out_dir, args.sample)

    print(f"[3] 文字長度    " + "  ".join(
        f"{t}={np.median([r['n_chars'] for r in recs if r['tier'] == t]):.0f}字(中位)"
        for t in sorted({r["tier"] for r in recs})))

    bad_glyph = [] if args.skip_glyph else check_glyph_coverage(recs, out_dir)
    check_distribution(recs)
    weak_pairs = check_minimal_pairs(recs)
    contact_sheet(recs, out_dir, rows=args.sheet_rows)

    print("=" * 62)
    fatal = len(problems) + len(bad_read) + len(bad_glyph)
    if fatal:
        print(f"發現 {fatal} 類嚴重問題，請先修正再發布。")
        sys.exit(1)
    warn = len(bad_size) + len(low_ink) + len(weak_pairs)
    print(f"通過。{'有 %d 項警告，建議看一下抽查表。' % warn if warn else '無警告。'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
