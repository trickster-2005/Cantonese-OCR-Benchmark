# -*- coding: utf-8 -*-
"""界定「粵語特有字」，以及設計用來探測失敗模式的最小對立對。

推導方式是頻率差分加三道濾網：

    候選 = 語料實際出現的字 − 標準通用字表 − 港式異體字 − 日文專用漢字
    字表 = 候選 ∪ 人工核心表 ∪ 擴展區粵語字 ∪ 最小對立對 − 人工排除表

為什麼需要這麼多層：

  單靠差分會同時撈到三種噪音——港式異體字（裏衞佈麪牀羣峯，是寫法差異不是
  粵語字）、日文漢字（維基的日本條目帶進來的 鱇栃峠）、以及罕見專名字。
  三道濾網都是實測驗證過才加的，見 config.CHARSET_SOURCES 的註解。

  反過來，差分也會誤刪 睇諗佢靚 這類「收在標準字表裡、但粵語用法不同」的字，
  所以要用人工核心表強制補回。

  最後仍會有殘留噪音。自動化只能做到候選清單，最後一哩要人眼過——
  benchmark 的字表只有 200-300 字，複核一次約 20 分鐘。
  build_benchmark.py 會產出 charset_review.tsv 供複核。
"""

import os
import sys
import urllib.request

import config

# ---------------------------------------------------------------- 人工核心表

# 這些字要嘛收在標準字表裡（會被差分誤刪），要嘛是粵語書寫的絕對核心，
# 無論如何都必須進 benchmark。只收有把握的，其餘交給差分自動撈。
CORE_CANTONESE_CHARS = (
    # 最高頻的功能字：助詞、代詞、否定、指示
    "嘅咗冇佢哋喺乜嘢嚟啲嗰唔咩咁冚係噉"
    # 常見動詞／形容詞
    "睇諗攰嗌嘥搵揸攞掟瞓踎躝閂慳靚黐劏呃唞啖啱喐喎嗒嘈掂揀揈搞搣撳甩焗煲燶"
    # 名詞／量詞／其他實詞
    "佬仔乸叻嚿罅餸軚鈪鎅脷腍膶舐舖蒲褸踭趷跣氹孭冧奀孻屙岌廿晏樖甴曱睩篤"
    # 語氣詞
    "嘞嗮咪囉㗎嚡譖囖嚹啩唓"
)

# Unicode 擴展區的粵語字。這些是最能區分模型好壞的樣本——多數字型根本沒有，
# 主流 OCR 訓練資料幾乎不含，但在香港日常書寫中真實存在。
EXT_CANTONESE_CHARS = [
    "㗎",      # 㗎  Ext-A  語氣詞
    "\U000210C1",  # 𡃁  Ext-B  𡃁仔（細路）
    "\U000210C9",  # 𡃉  Ext-B  語氣詞，HKCanCor 中出現 1,459 次
    "\U00020BB6",  # 𠮶  Ext-B  嗰
    "\U000282E2",  # 𨋢  Ext-B  電梯，香港日常用字
    "\U00020779",  # 𠝹  Ext-B  割
]

# 人工排除表：差分撈到但確定不是粵語特有字的。
# 分三類註記，方便日後複核時判斷。
EXCLUDE_CHARS = (
    # 港台異體字（是寫法差異，不是粵語用字）
    "綫爲敎凈卽蹟艷蔴菓囘繙滙鷄攷墻畧荳竈啓衹髙簒舋諡謚"
    # 日文漢字（維基的日本條目帶進來的，JIS 濾網抓不到 Big5 也收的那些）
    "鮟鱇栃咲嶋峠瀬鯛鵐魨鯥蟇"
    # 其他方言（不是粵語）
    "\U0002028E"   # 𠊎 客家話的「我」
    # 罕見專名字與技術用字（頻率低、與粵語無關）
    "勳驍鄺婭蕎綬撾瀧灝煒謖飆硤拏迾蟎鶡鉉銫嶠薊蚚峇堀躄銨讕鮓烴冪佮逳埐卌"
)

# ---------------------------------------------------------------- 最小對立對

# benchmark 最鋒利的探針：視覺相近、或語義對應的粵語／書面語字對。
#
# 設計意圖：同一個字會同時出現在「字」層和「句」層。如果模型在單字層辨對、
# 在句子層卻把同一個字辨成對應的書面語字，就直接證明了錯誤來自語言模型
# 先驗覆蓋視覺證據，而不是看不清楚字形。這是這個 benchmark 的核心實驗。
#
# 格式：(粵語字, 容易被誤判成的字, 說明)
MINIMAL_PAIRS = [
    ("冇", "有", "字形僅差兩筆，語義完全相反——辨錯會讓句意顛倒"),
    ("嘅", "既", "最常見的粵語 OCR 混淆"),
    ("咗", "左", "右半部相同，左偏旁易被忽略"),
    ("佢", "但", "字形極近"),
    ("係", "系", "多一個人字旁"),
    ("哋", "地", "多一個口字旁"),
    ("喺", "係", "兩個粵語字互相混淆"),
    ("嗰", "個", "多一個口字旁"),
    ("啲", "的", "粵語量詞 vs 書面語助詞"),
    ("嚟", "來", "粵語 vs 書面語同義字"),
    ("睇", "看", "語義替換，非字形相近——測語言模型先驗"),
    ("諗", "想", "語義替換"),
    ("乜", "什", "語義替換"),
    ("唔", "不", "語義替換，粵語否定詞"),
    ("嘢", "野", "字形近，常被誤寫成野"),
]


# ---------------------------------------------------------------- 工具

def is_han(c):
    """判斷是不是漢字（含擴展區 A/B/C…）。"""
    o = ord(c)
    return (0x4E00 <= o <= 0x9FFF or      # 基本區
            0x3400 <= o <= 0x4DBF or      # 擴展 A
            0xF900 <= o <= 0xFAFF or      # 相容漢字
            0x20000 <= o <= 0x323AF)      # 擴展 B 以後


def char_plane(c):
    """回報字元屬於哪個區，用來在 metadata 裡標記難度來源。"""
    o = ord(c)
    if 0x4E00 <= o <= 0x9FFF:
        return "BMP"
    if 0x3400 <= o <= 0x4DBF:
        return "ExtA"
    if o >= 0x20000:
        return "ExtB+"
    if 0xF900 <= o <= 0xFAFF:
        return "Compat"
    return "other"


def _download(url, dest):
    """抓一個純文字檔並快取。失敗時回傳 None 而不是拋例外，
    讓呼叫端可以在沒有網路時退回只用人工核心表。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return open(dest, encoding="utf-8").read()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cantobench/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            text = r.read().decode("utf-8")
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        print(f"[charset] 下載失敗 {url}: {e}", file=sys.stderr)
        return None


def _parse_charlist(text):
    """標準字表是一行一個字，'#' 開頭是分級標題。只取漢字。"""
    chars = set()
    if not text:
        return chars
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for c in line:
            if is_han(c):
                chars.add(c)
    return chars


def _read_user_charfile(path):
    """讀人工複核後的 include/exclude 檔。一行一個字，'#' 開頭是註解。"""
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        for c in line:
            if is_han(c):
                out.add(c)
    return out


def load_standard_charsets():
    """下載所有標準字表。回傳 dict[name] = set(chars)。"""
    out = {}
    for key, url in config.CHARSET_SOURCES.items():
        dest = os.path.join(config.CACHE_DIR, "charsets", key + ".txt")
        out[key] = _parse_charlist(_download(url, dest))
        print(f"[charset] {key}: {len(out[key]):,} 字")
    return out


# ---------------------------------------------------------------- 主推導

def derive_cantonese_charset(corpus_char_freq, min_freq=None):
    """從語料字頻推導粵語特有字。

    corpus_char_freq: collections.Counter，語料中每個字出現次數
    min_freq:         差分候選的最低出現次數，None = 用 config 的設定

    回傳 (charset, info)：
      charset — dict[char] = dict(freq=, plane=, source=)
      info    — 推導過程的統計，寫進 metadata 供論文引用
    """
    if min_freq is None:
        min_freq = config.CHARSET_MIN_FREQ

    std = load_standard_charsets()

    # 濾網 1：兩岸港三份標準通用字表
    standard = (std.get("putonghua_standard", set())
                | std.get("tw_common", set())
                | std.get("hk_primary", set()))

    # 濾網 2：JIS 專用字＝日文漢字。在 JIS 裡但 Big5 系列都沒有的字。
    jp_only = std.get("jis", set()) - std.get("big5", set()) - std.get("big5e", set())

    # 濾網 3：人工排除表（程式內建 ＋ 使用者複核後補的）
    excluded = set(EXCLUDE_CHARS) | _read_user_charfile(config.EXCLUDE_CHARS_FILE)

    charset = {}
    n_filtered = dict(standard=0, japanese=0, excluded=0, low_freq=0)

    # 來源 1：差分。語料裡出現、但通過三道濾網的漢字。
    for c, n in corpus_char_freq.items():
        if not is_han(c):
            continue
        if n < min_freq:
            n_filtered["low_freq"] += 1
            continue
        if c in standard:
            n_filtered["standard"] += 1
            continue
        if c in jp_only:
            n_filtered["japanese"] += 1
            continue
        if c in excluded:
            n_filtered["excluded"] += 1
            continue
        charset[c] = dict(freq=n, plane=char_plane(c), source="diff")

    def force_add(chars, tag):
        """強制補進字表。人工排除表優先權最高，被排除的不補。"""
        for c in chars:
            if not is_han(c) or c in excluded:
                continue
            if c in charset:
                charset[c]["source"] += "+" + tag
            else:
                charset[c] = dict(freq=corpus_char_freq.get(c, 0),
                                  plane=char_plane(c), source=tag)

    # 來源 2：人工核心表。補回被標準字表「吃掉」的粵語常用字。
    force_add(CORE_CANTONESE_CHARS, "core")
    # 來源 3：擴展區粵語字。即使語料沒出現也要收——它們最能拉開模型差距。
    force_add(EXT_CANTONESE_CHARS, "ext")
    # 來源 4：最小對立對裡的粵語字一定要在
    force_add([p[0] for p in MINIMAL_PAIRS], "pair")
    # 來源 5：使用者複核後手動加回的
    force_add(_read_user_charfile(config.INCLUDE_CHARS_FILE), "manual")

    planes, sources = {}, {}
    for c, d in charset.items():
        planes[d["plane"]] = planes.get(d["plane"], 0) + 1
        sources[d["source"]] = sources.get(d["source"], 0) + 1

    info = dict(
        total=len(charset),
        by_plane=planes,
        by_source=sources,
        filtered_out=n_filtered,
        standard_charset_size=len(standard),
        japanese_only_size=len(jp_only),
        excluded_size=len(excluded),
        min_freq=min_freq,
        hkscs_overlap=len(set(charset) & std.get("hkscs", set())),
    )
    return charset, info


def write_review_file(charset, corpus_sentences, path):
    """產出人工複核用的 TSV：字、頻率、分區、來源、例句。

    複核完把不要的字填進 data/exclude_chars.txt、漏掉的填進
    data/include_chars.txt，重跑一次 build_benchmark.py 即可。
    """
    # 為每個字找一個例句，讓複核時能看到實際用法
    example = {}
    for s, _src in corpus_sentences:
        for c in set(s):
            if c in charset and c not in example:
                example[c] = s
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = sorted(charset.items(), key=lambda kv: (-kv[1]["freq"], kv[0]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("字\t碼位\t語料頻率\t分區\t來源\t例句\n")
        for c, d in rows:
            f.write(f"{c}\tU+{ord(c):05X}\t{d['freq']}\t{d['plane']}\t{d['source']}\t"
                    f"{example.get(c, '')}\n")
    print(f"[charset] 人工複核檔已寫入 {path}（{len(rows)} 字）")
    return path


def minimal_pair_lookup():
    """回傳 dict[粵語字] = (易混字, 說明)，供 metadata 標記使用。"""
    return {c: (o, why) for c, o, why in MINIMAL_PAIRS}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import corpora

    bundle = corpora.build_corpus_bundle()
    cs, info = derive_cantonese_charset(bundle["char_freq"])

    print("\n=== 粵語特有字推導結果 ===")
    print(f"總數 {info['total']:,}")
    print(f"分區 {info['by_plane']}")
    print(f"來源 {info['by_source']}")
    print(f"濾除 {info['filtered_out']}")
    print(f"與 HKSCS 交集 {info['hkscs_overlap']:,}")

    top = sorted(cs.items(), key=lambda kv: -kv[1]["freq"])[:40]
    print("\n語料中最高頻的粵語特有字：")
    print("  " + " ".join(f"{c}({d['freq']})" for c, d in top))

    out = os.path.join(config.OUT_DIR, config.CHARSET_REVIEW_FILE)
    write_review_file(cs, bundle["sentences"], out)
