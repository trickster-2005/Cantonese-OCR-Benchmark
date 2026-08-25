# -*- coding: utf-8 -*-
"""字型載入與覆蓋率驗證。

這個模組存在的唯一理由是防止「豆腐字」——字型缺該字時 PIL 會畫出一個空白
方框，但標註檔裡寫的還是原本的字，等於直接汙染 ground truth。合成中文
OCR 資料集最常見、也最難察覺的錯誤就是這個，因為圖看起來「有東西」。

防線有兩層：
  1. 渲染前查 cmap，該字型沒有的碼位就換字型（fonts.py，這裡）
  2. 渲染後查墨水比例，低於門檻直接丟棄（render.py）

只收 SIL OFL 1.1 授權的字型。系統字型除非白名單命中，否則只列出不使用——
Windows / macOS 內建的中文字型不能拿來產生公開散布的資料集。
"""

import os
import sys
import glob

import config

try:
    from fontTools.ttLib import TTFont, TTCollection
except ImportError:
    print("[fonts] 需要 fonttools：pip install fonttools", file=sys.stderr)
    raise

from PIL import ImageFont

# 系統字型目錄裡，只有檔名命中這些樣式的才會被採用。
# 全部是 SIL OFL 1.1，可自由散布、可商用。
SAFE_FONT_PATTERNS = (
    "notosanshk", "notoserifhk", "notosanstc", "notoseriftc",
    "notosanscjk", "notoserifcjk",
    "chironhei", "chironsung",
    "sourcehansans", "sourcehanserif",
    "hkcs",
    "cwtexq", "cwtexming", "cwtexhei", "cwtexkai",
    "lxgw", "wenkai",
)

# 用來評估字型「粵語適用度」的探針字。覆蓋率低的字型會被降權但不一定排除，
# 因為只渲染 BMP 常用字時它們仍然有用（也提供了字型多樣性）。
PROBE_BMP = "嘅咗冇佢哋喺乜嘢睇諗嚟冚咁啲嗰唔咩"
PROBE_EXT = "㗎\U000210C1\U000210C9\U00020BB6\U000282E2\U00020779"


class FontEntry:
    """一個可用的字型實例（同一個檔案的不同 index / 不同字重算不同實例）。"""

    def __init__(self, path, index=0, variation=None, family=None, license="OFL-1.1"):
        self.path = path
        self.index = index
        self.variation = variation       # 變體字型的字重，None = 靜態字型
        self.family = family or os.path.splitext(os.path.basename(path))[0]
        self.license = license
        self.cmap = set()
        self.coverage_bmp = 0.0
        self.coverage_ext = 0.0

    @property
    def key(self):
        v = f"@{self.variation}" if self.variation else ""
        i = f"#{self.index}" if self.index else ""
        return f"{self.family}{i}{v}"

    def covers(self, text):
        """這個字型能不能畫出 text 的每一個字元。"""
        return all(ord(c) in self.cmap for c in text)

    def missing(self, text):
        return [c for c in text if ord(c) not in self.cmap]

    def load_pil(self, size):
        """建立 PIL 字型物件。變體字型會套用指定字重。"""
        f = ImageFont.truetype(self.path, size, index=self.index)
        if self.variation is not None:
            try:
                f.set_variation_by_axes([self.variation])
            except Exception:
                pass   # FreeType 太舊不支援變體軸，就用預設字重
        return f

    def to_dict(self):
        return dict(key=self.key, family=self.family, path=os.path.basename(self.path),
                    index=self.index, variation=self.variation, license=self.license,
                    n_glyphs=len(self.cmap),
                    coverage_bmp=round(self.coverage_bmp, 3),
                    coverage_ext=round(self.coverage_ext, 3))


def _read_cmaps(path):
    """回傳 [(index, cmap set)]。TTC 會展開成多個 face。"""
    out = []
    try:
        if path.lower().endswith((".ttc", ".otc")):
            coll = TTCollection(path, lazy=True)
            for i, f in enumerate(coll.fonts):
                cm = f.getBestCmap()
                if cm:
                    out.append((i, set(cm.keys())))
        else:
            f = TTFont(path, fontNumber=0, lazy=True)
            cm = f.getBestCmap()
            if cm:
                out.append((0, set(cm.keys())))
    except Exception as e:
        print(f"[fonts] 略過無法解析的字型 {os.path.basename(path)}: {e}", file=sys.stderr)
    return out


def _is_variable(path, index=0):
    """判斷是不是變體字型（有 fvar 表）。"""
    try:
        if path.lower().endswith((".ttc", ".otc")):
            f = TTCollection(path, lazy=True).fonts[index]
        else:
            f = TTFont(path, fontNumber=index, lazy=True)
        return "fvar" in f
    except Exception:
        return False


def _weight_axis_range(path, index=0):
    """取變體字型 wght 軸的範圍，用來過濾超出範圍的字重設定。"""
    try:
        if path.lower().endswith((".ttc", ".otc")):
            f = TTCollection(path, lazy=True).fonts[index]
        else:
            f = TTFont(path, fontNumber=index, lazy=True)
        for a in f["fvar"].axes:
            if a.axisTag == "wght":
                return a.minValue, a.maxValue
    except Exception:
        pass
    return None


FONT_EXTS = ("*.ttf", "*.otf", "*.ttc", "*.otc")


def _candidate_files():
    """蒐集所有候選字型檔：先是 fetch_fonts.py 下載的，再掃系統目錄。"""
    files = []

    # 1. 專案自己下載的 OFL 字型，全部採用
    for pat in FONT_EXTS:
        files += glob.glob(os.path.join(config.FONTS_DIR, "**", pat), recursive=True)

    # 2. 系統字型，只採用白名單命中的。
    #    白名單是必要的：系統上可能裝著授權不允許用於公開資料集的中文字型，
    #    誤用會讓整份資料集不能發布。
    for d in config.EXTRA_FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for pat in FONT_EXTS:
            for p in glob.glob(os.path.join(d, "**", pat), recursive=True):
                base = os.path.basename(p).lower().replace("-", "").replace("_", "")
                if any(s in base for s in SAFE_FONT_PATTERNS):
                    files.append(p)

    # 同一個檔可能透過 symlink 被掃到兩次
    seen, out = set(), []
    for p in files:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def load_font_pool(verbose=True):
    """建立可用字型池。

    回傳 (entries, report)：
      entries — [FontEntry]，已經載好 cmap，可以直接查覆蓋率
      report  — 給 metadata / dataset card 用的統計
    """
    entries = []
    for path in _candidate_files():
        for index, cmap in _read_cmaps(path):
            variations = [None]
            if _is_variable(path, index):
                rng = _weight_axis_range(path, index)
                if rng:
                    lo, hi = rng
                    variations = [w for w in config.VARIABLE_FONT_WEIGHTS if lo <= w <= hi]
                    if not variations:
                        variations = [None]

            for var in variations:
                e = FontEntry(path, index=index, variation=var)
                e.cmap = cmap
                e.coverage_bmp = sum(1 for c in PROBE_BMP if ord(c) in cmap) / len(PROBE_BMP)
                e.coverage_ext = sum(1 for c in PROBE_EXT if ord(c) in cmap) / len(PROBE_EXT)
                entries.append(e)

    # 至少要能畫出一半的常用粵字才有意義，否則多半是日文/韓文字型
    usable = [e for e in entries if e.coverage_bmp >= 0.5]
    usable.sort(key=lambda e: (-e.coverage_ext, -e.coverage_bmp, e.key))

    report = dict(
        scanned_files=len({e.path for e in entries}),
        total_instances=len(entries),
        usable_instances=len(usable),
        with_full_ext=sum(1 for e in usable if e.coverage_ext >= 0.99),
        families=sorted({e.family for e in usable}),
        fonts=[e.to_dict() for e in usable],
    )

    if verbose:
        print(f"[fonts] 掃到 {report['scanned_files']} 個字型檔、"
              f"{len(entries)} 個實例，可用 {len(usable)} 個")
        print(f"[fonts] 其中 {report['with_full_ext']} 個完整覆蓋擴展區粵語字")
        for e in usable[:15]:
            print(f"        {e.key:34} BMP {e.coverage_bmp:5.0%}  Ext {e.coverage_ext:5.0%}"
                  f"  {len(e.cmap):>6,} 字")
        if len(usable) > 15:
            print(f"        ...（另外 {len(usable)-15} 個）")

    if not usable:
        raise RuntimeError(
            "找不到任何可用的中文字型。請先執行 python fetch_fonts.py 下載 OFL 字型。")

    return usable, report


def fonts_for(text, pool):
    """回傳能完整畫出 text 的字型子集。這是防豆腐字的第一道防線。"""
    return [e for e in pool if e.covers(text)]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pool, rep = load_font_pool()
    print(f"\n字型家族：{', '.join(rep['families'])}")

    print("\n=== 擴展區粵語字的字型支援度（這決定哪些字進得了 benchmark）===")
    for c in PROBE_EXT:
        ok = [e for e in pool if ord(c) in e.cmap]
        print(f"  {c}  U+{ord(c):05X}  {len(ok):>3}/{len(pool)} 個字型支援")
