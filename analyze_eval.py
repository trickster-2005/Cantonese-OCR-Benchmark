# -*- coding: utf-8 -*-
"""彙總 evaluate.py 的輸出，算出網站要的分層指標與核心實驗結果。

evaluate.py 的 FIELDS 沒有把 canto_chars 寫進 eval_results.csv（只有 file 能
回頭對應），所以句子層「這句含哪些粵語字」要從 build_benchmark.py 原本產出的
metadata.jsonl 用 file 當 key join 回來，而不是重跑一次評測。
"""
import csv
import json
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

meta_by_file = {}
for line in open("output/metadata.jsonl", encoding="utf-8"):
    if line.strip():
        m = json.loads(line)
        meta_by_file[m["file"]] = m

rows = list(csv.DictReader(open("output/eval_results.csv", encoding="utf-8")))
for r in rows:
    r["canto_chars"] = meta_by_file.get(r["file"], {}).get("canto_chars", "")

by_model = defaultdict(list)
for r in rows:
    by_model[r["model"]].append(r)


def agg(rs, keyfn=lambda r: True):
    sub = [r for r in rs if keyfn(r)]
    if not sub:
        return None

    def avg(k):
        return sum(float(r[k]) for r in sub) / len(sub)

    return dict(n=len(sub), cer=avg("cer_clipped"), acc=avg("acc"),
                ned_sim=avg("ned_sim"), f1=avg("f1"))


results = {}
for m in ("qwen3vl4b", "internvl35_4b"):
    rs = by_model[m]
    print(f"=== {m} (n={len(rs)}) ===")
    o = agg(rs)
    print(f"  整體   CER={o['cer']:.4f} ACC={o['acc']:.4f} "
          f"1-NED={o['ned_sim']:.4f} F1={o['f1']:.4f}")
    tiers = {}
    for tier in ("char", "word", "sentence"):
        t = agg(rs, lambda r, tier=tier: r["tier"] == tier)
        tiers[tier] = t
        if t:
            print(f"  {tier:9} ACC={t['acc']:.4f}  CER={t['cer']:.4f}  (n={t['n']})")
    ext = agg(rs, lambda r: r.get("plane") in ("ExtA", "ExtB+"))
    if ext:
        print(f"  擴展區字  ACC={ext['acc']:.4f}  (n={ext['n']})")
    print()
    results[m] = dict(overall=o, tiers=tiers, ext=ext)

print("=== 核心實驗：最小對立對粵語字，單字層 vs 句子層命中率 ===")
core = {}
for m in ("qwen3vl4b", "internvl35_4b"):
    rs = by_model[m]
    pair_chars = set(r["gt"] for r in rs
                      if r["tier"] == "char" and r.get("minimal_pair_with"))
    char_acc = agg(rs, lambda r: r["tier"] == "char" and r["gt"] in pair_chars)

    hit, tot = 0, 0
    for r in rs:
        if r["tier"] != "sentence":
            continue
        cc = r.get("canto_chars", "")
        for c in cc:
            if c in pair_chars:
                tot += 1
                if c in r["prediction"]:
                    hit += 1
    sent_rate = hit / max(tot, 1)
    core[m] = dict(char_acc=char_acc["acc"], char_n=char_acc["n"],
                    sent_rate=sent_rate, sent_n=tot)
    print(f"{m:16} 單字層對立字ACC={char_acc['acc']:.4f} (n={char_acc['n']})  "
          f"句子層對立字命中率={sent_rate:.4f} (n={tot})")

print()
print("=== 逐字：冇 在單字層 vs 句子層的命中狀況（招牌案例）===")
for m in ("qwen3vl4b", "internvl35_4b"):
    rs = by_model[m]
    char_rows = [r for r in rs if r["tier"] == "char" and r["gt"] == "冇"]
    if char_rows:
        n_ok = sum(1 for r in char_rows if r["acc"] == "1.0")
        print(f"{m:16} 冇-單字層: {n_ok}/{len(char_rows)} 正確")
        for r in char_rows[:3]:
            print(f"    font={r['font_family']:20} pred={r['prediction']!r}")
    sent_rows = [r for r in rs if r["tier"] == "sentence" and "冇" in r.get("canto_chars", "")]
    if sent_rows:
        n_hit = sum(1 for r in sent_rows if "冇" in r["prediction"])
        print(f"{m:16} 冇-句子層: {n_hit}/{len(sent_rows)} 命中")
        for r in sent_rows[:3]:
            print(f"    gt={r['gt'][:24]!r}  pred={r['prediction'][:24]!r}")
    print()

print("=== 15 組最小對立對逐字拆解：單字層（完全匹配）vs 句子層（任意位置出現）===")
print("這張表格就是 README「基線分數」段落裡貼的那張，數字對不上要先查這裡\n")
for m in ("qwen3vl4b", "internvl35_4b"):
    rs = by_model[m]
    pairs = sorted(
        set((r["gt"], r["minimal_pair_with"]) for r in rs
            if r["tier"] == "char" and r.get("minimal_pair_with")),
        key=lambda p: -sum(1 for r in rs if r["tier"] == "char" and r["gt"] == p[0]
                           and p[0] in r["prediction"]))
    print(f"--- {m} ---")
    print(f"{'對立對':8} {'單字層':>10} {'句子層':>12}")
    for canto, std in pairs:
        cr = [r for r in rs if r["tier"] == "char" and r["gt"] == canto]
        c_hit = sum(1 for r in cr if canto in r["prediction"])
        sh, st = 0, 0
        for r in rs:
            if r["tier"] != "sentence":
                continue
            if canto in r.get("canto_chars", ""):
                st += 1
                if canto in r["prediction"]:
                    sh += 1
        print(f"{canto}→{std:6} {c_hit:>3}/{len(cr):<3}  {sh:>5}/{st:<5}")
    print()
