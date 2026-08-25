# -*- coding: utf-8 -*-
"""CantoBench 合成資料集共用設定。

所有可調參數集中在這裡，其他模組只讀不寫。
在 server 上要改規模、難度分布或字型來源，改這個檔案就好。
"""

import os

# ---------------------------------------------------------------- 路徑

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("CANTOBENCH_CACHE", os.path.join(ROOT, ".cache"))
FONTS_DIR = os.environ.get("CANTOBENCH_FONTS", os.path.join(ROOT, "fonts"))
OUT_DIR = os.environ.get("CANTOBENCH_OUT", os.path.join(ROOT, "output"))

# 產出結構（刻意對齊 TC-STR，讓 eval/run_eval.py 可以直接吃）：
#   output/
#     images/            *.jpg
#     test_labels.txt    filename\tlabel
#     metadata.jsonl     每張圖的完整 metadata
#     dataset_card.md    自動產生的 HF dataset card
IMAGES_SUBDIR = "images"
LABELS_FILE = "test_labels.txt"
METADATA_FILE = "metadata.jsonl"

# ---------------------------------------------------------------- 隨機性

# benchmark 必須可重現：同一個 seed + 同一份語料 = 完全相同的圖片。
# 每張圖的 rng 由 MASTER_SEED 和該圖 index 推導，所以 worker 數量改變
# 不會影響產出結果。
MASTER_SEED = 20260825

# ---------------------------------------------------------------- 規模

# 三層結構。value = 要產生的圖片張數。
# 預設總量約 5,000 張，這是「8 個模型跑得完」的量級（L40S 上約 17 小時）。
# 未來要做訓練集，把數字調大即可，生成本身很快（實測約 313 張/秒）。
TIER_SIZES = {
    "char": 1200,      # 單一粵語特有字
    "word": 1800,      # 2-4 字粵語詞彙
    "sentence": 2000,  # 4-25 字粵語短句
}

# 每個文字項目最多重複渲染幾次（用不同字型/增強）。
# 重複渲染可以讓「同一個字的辨識率」有統計意義，而不是單一樣本的運氣。
RENDERS_PER_ITEM = 3

# 句子長度範圍（字元數，含標點）
SENTENCE_LEN = (4, 25)
WORD_LEN = (2, 4)

# ---------------------------------------------------------------- 難度分層

# benchmark 的價值在於能看出模型「在哪裡開始壞掉」，所以刻意做難度分層。
# 每個 level 的權重 = 該難度在資料集中的比例。
DIFFICULTY_WEIGHTS = {
    "clean": 0.25,   # 幾乎無增強：測純字形辨識能力的上限
    "light": 0.35,   # 輕微：模擬乾淨的翻拍/掃描
    "medium": 0.25,  # 中等：模擬一般手機拍攝
    "hard": 0.15,    # 困難：模擬低光、壓縮、傾斜的真實街拍
}

# 各難度的增強參數上界。實際值在區間內隨機抽。
DIFFICULTY_PARAMS = {
    "clean":  dict(rotate=0.5, blur=0.3, noise=2,  persp=0.004, elastic=0.0, jpeg=(92, 98), contrast=(0.95, 1.05)),
    "light":  dict(rotate=2.0, blur=0.8, noise=5,  persp=0.012, elastic=1.5, jpeg=(80, 95), contrast=(0.85, 1.15)),
    "medium": dict(rotate=4.0, blur=1.4, noise=9,  persp=0.025, elastic=3.0, jpeg=(60, 88), contrast=(0.70, 1.30)),
    "hard":   dict(rotate=7.0, blur=2.2, noise=14, persp=0.045, elastic=5.0, jpeg=(35, 75), contrast=(0.55, 1.45)),
}

# ---------------------------------------------------------------- 影像

IMAGE_HEIGHT = 48          # OCR 文字行的慣用高度
RENDER_FONT_SIZE = 64      # 先用大字級渲染再縮小，可以得到比較自然的抗鋸齒
PADDING = (24, 16)         # (水平, 垂直) 邊距，單位 px（渲染時的尺度）
JPEG_QUALITY = 95          # 最終存檔品質（增強裡的 jpeg 參數是「模擬壓縮痕跡」，跟這個不同）

# 可讀性守門的兩個獨立門檻。任一不過就丟棄該樣本。
# 墨水比例抓「什麼都沒有」（豆腐字、渲染失敗）；
# 對比抓「有東西但淡到看不見」——細字重加重度增強時的典型結果。
# 兩個都需要：實測只看墨水比例會漏掉對比只剩 11 階的樣本。
MIN_INK_RATIO = 0.004      # 實測：真實樣本 1% 分位 0.0061，純背景對照最高 0.0005
MIN_CONTRAST = 45          # 灰階 99% 與 1% 分位差

# 渲染時文字與背景的最低亮度差。設太低的話樣本會在增強後被守門丟掉，
# 而且被丟掉的多半是高難度樣本，等於偷偷把資料集變簡單。
MIN_TEXT_BG_GAP = 85

# ---------------------------------------------------------------- 字型

# 全部是 SIL OFL 1.1，可自由散布、可商用，且對粵語字/HKSCS 覆蓋最好。
# fetch_fonts.py 會把這些抓到 FONTS_DIR。
FONT_SOURCES = [
    # 昭源黑體：現代黑體，香港字形，收錄 HKSCS-2016 之後的補充字。
    # 實測 47,724 字元、Ext-B 粵語字覆蓋 3/4，是所有候選中最好的。
    dict(name="ChironHeiHK", license="OFL-1.1",
         home="https://github.com/chiron-fonts/chiron-hei-hk",
         url="https://raw.githubusercontent.com/chiron-fonts/chiron-hei-hk/release/STATIC_TTF/{f}",
         files=["ChironHeiHK-EL.ttf", "ChironHeiHK-L.ttf", "ChironHeiHK-N.ttf",
                "ChironHeiHK-M.ttf", "ChironHeiHK-B.ttf", "ChironHeiHK-H.ttf"]),
    # 昭源宋體：明體／宋體，同樣的香港字形基礎
    dict(name="ChironSungHK", license="OFL-1.1",
         home="https://github.com/chiron-fonts/chiron-sung-hk",
         url="https://raw.githubusercontent.com/chiron-fonts/chiron-sung-hk/release/STATIC_TTF/{f}",
         files=["ChironSungHK-EL.ttf", "ChironSungHK-L.ttf", "ChironSungHK-N.ttf",
                "ChironSungHK-M.ttf", "ChironSungHK-B.ttf", "ChironSungHK-H.ttf"]),
    # Noto CJK 變體字型。一個檔可以展開成多個字重，字型多樣性 CP 值最高。
    dict(name="NotoCJK", license="OFL-1.1",
         home="https://github.com/notofonts/noto-cjk",
         url="https://github.com/notofonts/noto-cjk/raw/main/{f}",
         files=["Sans/Variable/TTF/Subset/NotoSansHK-VF.ttf",
                "Sans/Variable/TTF/Subset/NotoSansTC-VF.ttf",
                "Serif/Variable/TTF/Subset/NotoSerifHK-VF.ttf",
                "Serif/Variable/TTF/Subset/NotoSerifTC-VF.ttf"]),
    # 香港民間字集：專門補 HKSCS 與粵語用字的字形
    dict(name="HKCS", license="OFL-1.1",
         home="https://github.com/hfhchan/hkcs",
         url="https://raw.githubusercontent.com/hfhchan/hkcs/master/{f}",
         files=["hkcs-traditional-bmp-canonicalonly.otf"]),
]

# 額外掃描這些目錄裡既有的字型（server 上如果已經裝了 Noto CJK 就會自動用上）。
# 只會採用「授權明確可散布」的，其餘僅列出不使用——見 fonts.py 的 SAFE_FONT_PATTERNS。
EXTRA_FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
]

# 變體字型（VF）要展開成哪幾個字重。PIL 支援時才會生效。
VARIABLE_FONT_WEIGHTS = [300, 400, 500, 700, 900]

# ---------------------------------------------------------------- 語料

# 三個來源，授權都乾淨。
# 注意：含 Wikipedia 之後整個資料集受 CC BY-SA 4.0 的 copyleft 約束。
CORPORA = {
    "wikipedia": dict(
        enabled=True,
        hf_dataset="jed351/cantonese-wikipedia",
        split="train",
        field="text",
        license="CC BY-SA 4.0",
        attribution="粵語維基百科 Cantonese Wikipedia (zh-yue.wikipedia.org)",
        max_articles=20000,   # 只取前 N 篇來切句，夠用且省下載時間
    ),
    "hkcancor": dict(
        enabled=True,
        source="pycantonese",
        license="CC BY 4.0",
        attribution="Hong Kong Cantonese Corpus (HKCanCor), Luke & Wong (2015)",
    ),
    "rime": dict(
        enabled=True,
        base_url="https://raw.githubusercontent.com/rime/rime-cantonese/main/",
        # 只取 CC BY 4.0 的檔案。jyut6ping3.maps 是 ODbL-1.0，刻意不收。
        files=["jyut6ping3.chars.dict.yaml",
               "jyut6ping3.words.dict.yaml",
               "jyut6ping3.phrase.dict.yaml"],
        license="CC BY 4.0",
        attribution="rime-cantonese (CanCLID)",
    ),
}

# 標準字表，用來反推「粵語特有字」＝ 語料出現字 − 標準通用字。
# 這些是各地政府公告的標準字表（事實性資料），只用於推導，不隨資料集散布。
#
# 每一份的角色（都是實測驗證過才加進來的，不要隨便增刪）：
#   putonghua_standard  大陸通用規範漢字表，主力濾網
#   tw_common           台灣常用國字表，補大陸表沒有的正體字
#   hk_primary          香港小學學習字詞表。實測能濾掉 9/15 港式異體字
#                       （裏衞佈麪牀羣峯恆着）而只誤殺 1 個真粵字，
#                       而那個字已經在人工核心表裡會被補回來
#   jis / big5 / big5e  用來算「JIS 專用字」＝ 日文漢字，濾掉維基日本條目
#                       帶進來的 鱇栃峠 這類字。實測零誤殺
#   hkscs               只用於統計交集，不參與濾除
CHARSET_SOURCES = {
    "hkscs": "https://raw.githubusercontent.com/rime-aca/character_set/master/HKSCS.txt",
    "putonghua_standard": "https://raw.githubusercontent.com/rime-aca/character_set/master/%E9%80%9A%E7%94%A8%E8%A6%8F%E7%AF%84%E6%BC%A2%E5%AD%97%E8%A1%A8.txt",
    "tw_common": "https://raw.githubusercontent.com/rime-aca/character_set/master/%E5%B8%B8%E7%94%A8%E5%9C%8B%E5%AD%97%E8%A1%A8.txt",
    "hk_primary": "https://raw.githubusercontent.com/rime-aca/character_set/master/%E9%A6%99%E6%B8%AF%E5%B0%8F%E5%AD%B8%E5%AD%B8%E7%BF%92%E5%AD%97%E8%A9%9E%E8%A1%A8.txt",
    "jis": "https://raw.githubusercontent.com/rime-aca/character_set/master/JIS.txt",
    "big5": "https://raw.githubusercontent.com/rime-aca/character_set/master/Big5.txt",
    "big5e": "https://raw.githubusercontent.com/rime-aca/character_set/master/Big5E.txt",
}

# 人工複核。自動推導只能做到「候選清單」，最後一哩必須人眼過。
# benchmark 的字表只有 200-300 個字，人工複核約 20 分鐘，換來的品質提升很划算。
#
# build_benchmark.py 會產出 output/charset_review.tsv（含字頻、分區、例句），
# 複核完把不要的字填進 data/exclude_chars.txt、漏掉的填進 data/include_chars.txt，
# 重跑一次即可。兩個檔都是一行一個字，'#' 開頭是註解。
REVIEW_DIR = os.path.join(ROOT, "data")
EXCLUDE_CHARS_FILE = os.path.join(REVIEW_DIR, "exclude_chars.txt")
INCLUDE_CHARS_FILE = os.path.join(REVIEW_DIR, "include_chars.txt")
CHARSET_REVIEW_FILE = "charset_review.tsv"

# 差分候選的最低語料出現次數。低於此值多半是錯字或一次性專名，
# 但擴展區粵語字和人工核心表不受這個門檻限制。
CHARSET_MIN_FREQ = 3

# ---------------------------------------------------------------- 授權

DATASET_LICENSE = "CC BY-SA 4.0"
CODE_LICENSE = "MIT"
DATASET_NAME = "CantoBench-Synth"
DATASET_VERSION = "0.1.0"
