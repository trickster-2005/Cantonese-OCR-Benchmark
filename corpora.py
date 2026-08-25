# -*- coding: utf-8 -*-
"""三個語料來源的抓取、清理與切句。

來源與授權（這決定了整個資料集的授權，不要隨便加來源）：

  粵語 Wikipedia   CC BY-SA 4.0   ← copyleft，會傳染給整個資料集
  HKCanCor         CC BY 4.0
  rime-cantonese   CC BY 4.0      （只取 CC BY 的三個檔，maps 是 ODbL 不收）

刻意排除的來源：
  raptorkwok/cantonese_sentences  無授權聲明、內容為論壇貼文，著作權不在上傳者手上
  words.hk                        NC ODL 1.0，與 Wikipedia 的 CC BY-SA 法律上互斥
"""

import os
import re
import sys
import collections
import urllib.request

import config
import charset as charset_mod

# 句末標點，用來切句
SENT_END = "。！？!?；;\n"
# 允許留在句子裡的標點
KEEP_PUNCT = "，、。！？：；「」『』（）,.!?:;()《》〈〉—…·～~%"


def _cache_path(*parts):
    p = os.path.join(config.CACHE_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _clean_text(s):
    """統一清理：全形空白、控制字元、引用標記、網址。"""
    s = s.replace("　", " ").replace("\xa0", " ")
    s = re.sub(r"[\u0000-\u001f\u007f]", "", s)
    s = re.sub(r"\[\d+\]", "", s)          # 維基的引用標記
    s = re.sub(r"https?://\S+", "", s)     # 網址
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _acceptable(s, lo, hi, canto_chars=None, require_canto=False):
    """判斷一個句子能不能用。

    擋掉：長度不對、漢字比例太低（多半是外文或數字表格）、
    重複字元太多（爬蟲雜訊）、含私隱風險的長數字串。
    """
    if not (lo <= len(s) <= hi):
        return False
    han = sum(1 for c in s if charset_mod.is_han(c))
    if han / max(1, len(s)) < 0.6:          # 漢字要占六成以上
        return False
    if re.search(r"(.)\1{4,}", s):          # 同一字元連續 5 次以上
        return False
    if re.search(r"\d{6,}", s):             # 長數字串：電話／證件號風險
        return False
    if any(c in s for c in "{}[]|=<>"):     # 殘留的標記語言
        return False
    if require_canto and canto_chars:
        if not any(c in canto_chars for c in s):
            return False
    return True


def _keep_only_allowed(s):
    """只保留漢字與允許的標點，並修掉頭尾的孤立標點。"""
    s = "".join(c for c in s if charset_mod.is_han(c) or c in KEEP_PUNCT)
    return s.strip("，、：；,.;: ")


# ---------------------------------------------------------------- Wikipedia

def load_wikipedia(limit_sentences=200000):
    """從粵語維基百科切出短句。

    用 streaming 模式，不會把整份資料集抓下來。只掃前 max_articles 篇，
    以 5,000 張規模的 benchmark 來說遠遠夠用。
    """
    cfg = config.CORPORA["wikipedia"]
    if not cfg.get("enabled"):
        return []

    cache = _cache_path("corpus", "wikipedia_sents.txt")
    if os.path.exists(cache):
        sents = open(cache, encoding="utf-8").read().splitlines()
        print(f"[corpora] wikipedia: 讀取快取 {len(sents):,} 句")
        return sents

    try:
        from datasets import load_dataset
    except ImportError:
        print("[corpora] 需要 datasets 套件：pip install datasets", file=sys.stderr)
        return []

    print(f"[corpora] wikipedia: streaming {cfg['hf_dataset']} ...")
    ds = load_dataset(cfg["hf_dataset"], split=cfg["split"], streaming=True)

    lo, hi = config.SENTENCE_LEN
    sents, seen = [], set()
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > cfg["max_articles"] or len(sents) >= limit_sentences:
            break
        text = _clean_text(str(row.get(cfg["field"], "")))
        for raw in re.split(f"[{re.escape(SENT_END)}]", text):
            s = _keep_only_allowed(raw.strip())
            if _acceptable(s, lo, hi) and s not in seen:
                seen.add(s)
                sents.append(s)

    with open(cache, "w", encoding="utf-8") as f:
        f.write("\n".join(sents))
    print(f"[corpora] wikipedia: {len(sents):,} 句（掃過 {scanned:,} 篇）")
    return sents


# ---------------------------------------------------------------- HKCanCor

def load_hkcancor():
    """香港粵語語料庫。口語轉寫，核心粵字密度是三個來源中最高的（實測 7.8%）。

    轉寫慣例會留下痕跡要清掉：句末的句點、修補標記的連字號、
    以及少量拉丁字母的語碼混用。
    """
    cfg = config.CORPORA["hkcancor"]
    if not cfg.get("enabled"):
        return []

    cache = _cache_path("corpus", "hkcancor_sents.txt")
    if os.path.exists(cache):
        sents = open(cache, encoding="utf-8").read().splitlines()
        print(f"[corpora] hkcancor: 讀取快取 {len(sents):,} 句")
        return sents

    try:
        import pycantonese
    except ImportError:
        print("[corpora] 需要 pycantonese：pip install pycantonese", file=sys.stderr)
        return []

    corpus = pycantonese.hkcancor()
    lo, hi = config.SENTENCE_LEN
    sents, seen = [], set()
    for utt in corpus.utterances():
        s = "".join(tok.word for tok in utt.tokens)
        s = _clean_text(s).replace("-", "")      # 連字號是轉寫的修補標記
        s = _keep_only_allowed(s)
        if _acceptable(s, lo, hi) and s not in seen:
            seen.add(s)
            sents.append(s)

    with open(cache, "w", encoding="utf-8") as f:
        f.write("\n".join(sents))
    print(f"[corpora] hkcancor: {len(sents):,} 句")
    return sents


# ---------------------------------------------------------------- rime-cantonese

def _parse_rime_dict(text):
    """Rime 詞庫格式是 YAML 標頭、三點分隔線、之後接 TSV：

        <字>  <粵拼>  <頻率百分比，可省略>

    第三欄的百分比可以拿來做長尾加權，沒有的視為 0。
    """
    entries = []
    parts = text.split("\n...\n", 1)
    body = parts[1] if len(parts) > 1 else text
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if not cols or not cols[0]:
            continue
        word = cols[0].strip()
        jyutping = cols[1].strip() if len(cols) > 1 else ""
        weight = 0.0
        if len(cols) > 2 and cols[2].strip().endswith("%"):
            try:
                weight = float(cols[2].strip().rstrip("%"))
            except ValueError:
                weight = 0.0
        entries.append((word, jyutping, weight))
    return entries


def load_rime():
    """回傳 (chars, words)。chars 是單字條目，words 是多字詞條目。"""
    cfg = config.CORPORA["rime"]
    if not cfg.get("enabled"):
        return [], []

    all_entries = []
    for fn in cfg["files"]:
        cache = _cache_path("corpus", fn)
        if os.path.exists(cache):
            text = open(cache, encoding="utf-8").read()
        else:
            url = cfg["base_url"] + fn
            print(f"[corpora] rime: 下載 {fn} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "cantobench/0.1"})
            with urllib.request.urlopen(req, timeout=180) as r:
                text = r.read().decode("utf-8")
            with open(cache, "w", encoding="utf-8") as f:
                f.write(text)
        all_entries += _parse_rime_dict(text)

    chars = [(w, j, wt) for w, j, wt in all_entries
             if len(w) == 1 and charset_mod.is_han(w)]
    lo, hi = config.WORD_LEN
    words = [(w, j, wt) for w, j, wt in all_entries
             if lo <= len(w) <= hi and all(charset_mod.is_han(c) for c in w)]

    # 同字不同讀音會有多筆，去重時保留權重最大的
    def dedup(items):
        best = {}
        for w, j, wt in items:
            if w not in best or wt > best[w][2]:
                best[w] = (w, j, wt)
        return list(best.values())

    chars, words = dedup(chars), dedup(words)
    print(f"[corpora] rime: {len(chars):,} 單字條目、{len(words):,} 詞彙條目")
    return chars, words


# ---------------------------------------------------------------- 整合

def build_corpus_bundle():
    """把三個來源整合成建資料集需要的全部素材。

    回傳 dict：
      sentences   [(句子, 來源)]
      rime_chars  [(字, 粵拼, 權重)]
      rime_words  [(詞, 粵拼, 權重)]
      char_freq   Counter，語料字頻，拿去推導粵語特有字
      licenses    實際用到的來源與授權，寫進 dataset card
    """
    wiki = load_wikipedia()
    hkc = load_hkcancor()

    sentences = [(s, "wikipedia") for s in wiki] + [(s, "hkcancor") for s in hkc]
    rime_chars, rime_words = load_rime()

    # 字頻只從真實句子統計；詞表是字典，不反映實際使用頻率
    char_freq = collections.Counter()
    for s, _ in sentences:
        char_freq.update(c for c in s if charset_mod.is_han(c))

    licenses = []
    for key, name in (("wikipedia", "wikipedia"), ("hkcancor", "hkcancor"),
                      ("rime", "rime-cantonese")):
        used = (wiki if key == "wikipedia" else
                hkc if key == "hkcancor" else (rime_chars or rime_words))
        if used:
            c = config.CORPORA[key]
            licenses.append(dict(source=name, license=c["license"],
                                 attribution=c["attribution"]))

    print(f"[corpora] 合計 {len(sentences):,} 句、字頻表 {len(char_freq):,} 個相異字")
    return dict(sentences=sentences, rime_chars=rime_chars, rime_words=rime_words,
                char_freq=char_freq, licenses=licenses)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    b = build_corpus_bundle()
    cs, info = charset_mod.derive_cantonese_charset(b["char_freq"])
    print("\n=== 粵語特有字推導結果 ===")
    print(f"總數 {info['total']:,}")
    print(f"分區 {info['by_plane']}")
    print(f"來源 {info['by_source']}")
    print(f"與 HKSCS 交集 {info['hkscs_overlap']:,}")
    top = sorted(cs.items(), key=lambda kv: -kv[1]["freq"])[:40]
    print("\n語料中最高頻的粵語特有字：")
    print("  " + " ".join(f"{c}({d['freq']})" for c, d in top))
