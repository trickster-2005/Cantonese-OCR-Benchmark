# -*- coding: utf-8 -*-
"""挑一批有代表性的樣本給網站的互動瀏覽器用。

挑選原則：三層都要有、四種難度都要有、Unicode 三個分區都要有、
最小對立對要成對出現（不然網站上沒辦法示範核心實驗）。

產出：
    docs/assets/samples/*.jpg
    docs/assets/samples.json    每張圖的標籤與 metadata
"""

import argparse
import collections
import json
import os
import random
import shutil
import sys

import config
import charset as charset_mod


# 網站瀏覽器只需要「看得出設計意圖」的量，不是全覆蓋。
# 單字卡片又小又多，放太多會把詞彙與短句的樣本擠到看不見。
N_PAIRS = 7        # 最小對立對取幾組（每組兩張：粵語字＋書面語對照字）
N_EXT = 9          # 擴展區字元取幾個


def pick(recs, n_per_bucket=1, seed=20260825):
    """分層抽樣：先確保最小對立對成對，再補滿各層各難度。"""
    rng = random.Random(seed)
    chosen, seen_files = [], set()

    def take(rs, k):
        rs = [r for r in rs if r["file"] not in seen_files]
        rng.shuffle(rs)
        for r in rs[:k]:
            seen_files.add(r["file"])
            chosen.append(r)

    # 1. 最小對立對：粵語字與其書面語對照字要成對出現，優先取 clean 難度。
    #    只取前 N_PAIRS 組，順序照 MINIMAL_PAIRS 的定義（最有代表性的排前面）。
    pairs = charset_mod.MINIMAL_PAIRS[:N_PAIRS]
    for canto, other, _note in pairs:
        for t in (canto, other):
            cands = [r for r in recs if r["tier"] == "char" and r["text"] == t]
            clean = [r for r in cands if r.get("difficulty") == "clean"] or cands
            take(clean, 1)

    # 2. 擴展區字元：專案重點，但取樣即可，不必全列。
    #    優先取語料頻率高的，那些才是真的有人在用的字。
    ext = [r for r in recs if r["tier"] == "char"
           and r.get("plane") in ("ExtA", "ExtB+")]
    by_char = collections.defaultdict(list)
    for r in ext:
        by_char[r["text"]].append(r)
    ranked = sorted(by_char, key=lambda t: -max(
        r.get("corpus_freq", 0) for r in by_char[t]))
    for t in ranked[:N_EXT]:
        take(by_char[t], 1)

    # 3. 各層 × 各難度補滿
    for tier in ("char", "word", "sentence"):
        for diff in config.DIFFICULTY_PARAMS:
            take([r for r in recs if r["tier"] == tier
                  and r.get("difficulty") == diff], n_per_bucket)

    # 4. 句層兩個來源各補一些
    for src in ("wikipedia", "hkcancor"):
        take([r for r in recs if r["tier"] == "sentence"
              and r.get("source") == src], 3)

    return chosen


def main():
    ap = argparse.ArgumentParser(description="匯出網站用的樣本")
    ap.add_argument("--out", default=config.OUT_DIR)
    ap.add_argument("--docs", default=os.path.join(config.ROOT, "docs"))
    ap.add_argument("--per-bucket", type=int, default=2)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    recs = [json.loads(l) for l in
            open(os.path.join(out_dir, config.METADATA_FILE), encoding="utf-8")]

    chosen = pick(recs, args.per_bucket)
    dest = os.path.join(args.docs, "assets", "samples")
    os.makedirs(dest, exist_ok=True)
    for f in os.listdir(dest):
        if f.endswith(".jpg"):
            os.remove(os.path.join(dest, f))

    pairs = charset_mod.minimal_pair_lookup()
    # 反查：書面語對照字 -> 對應的粵語字
    counterpart_of = {o: c for c, (o, _) in pairs.items()}

    items = []
    for r in chosen:
        shutil.copy2(os.path.join(out_dir, config.IMAGES_SUBDIR, r["file"]),
                     os.path.join(dest, r["file"]))
        item = dict(file=r["file"], text=r["text"], tier=r["tier"],
                    difficulty=r.get("difficulty", ""),
                    plane=r.get("plane", ""), source=r.get("source", ""),
                    font=r.get("font_family", ""), n_chars=r.get("n_chars", 0),
                    role=r.get("role", ""), width=r.get("width", 0))
        if r.get("minimal_pair_with"):
            item["pair_with"] = r["minimal_pair_with"]
            item["pair_role"] = "cantonese"
        elif r["text"] in counterpart_of:
            item["pair_with"] = counterpart_of[r["text"]]
            item["pair_role"] = "standard"
        items.append(item)

    items.sort(key=lambda x: (["char", "word", "sentence"].index(x["tier"]),
                              x["file"]))
    with open(os.path.join(args.docs, "assets", "samples.json"), "w",
              encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

    size = sum(os.path.getsize(os.path.join(dest, i["file"])) for i in items)
    print(f"匯出 {len(items)} 張樣本 → {dest}  ({size/1024:.0f} KB)")
    print("  分層:", dict(collections.Counter(i["tier"] for i in items)))
    print("  難度:", dict(collections.Counter(i["difficulty"] for i in items)))
    print("  分區:", dict(collections.Counter(i["plane"] for i in items if i["plane"])))
    print("  對立對:", sum(1 for i in items if i.get("pair_with")))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
