# -*- coding: utf-8 -*-
"""文字行渲染與影像增強。

流程：
    大字級渲染 → 筆畫粗細變化 → 彈性形變 → 透視 → 旋轉
    → 色彩/對比 → 模糊 → 雜訊 → JPEG 壓縮痕跡 → 縮到目標高度

幾個刻意的設計決定：

  先用大字級（RENDER_FONT_SIZE）畫再縮到 IMAGE_HEIGHT，而不是直接畫小字。
  直接畫小字的抗鋸齒是字型引擎給的，長得跟真實拍攝/掃描不一樣；先大後小
  才會產生接近真實重採樣的邊緣。

  「模擬手寫」不是把邊緣鋸齒化。鋸齒是低解析度和壓縮的產物，不是手寫。
  真正接近手寫的是筆畫粗細不均（stroke）＋ 非剛性的局部位移（elastic），
  這裡兩者都做。真手寫還需要手寫字型，粵語手寫字型幾乎不存在，留給 v2。

  每張圖的隨機數都由 (MASTER_SEED, index) 推導，所以 worker 數量改變不會
  影響產出。benchmark 必須可重現。
"""

import random

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import config


def rng_for(index):
    """由主 seed 和樣本序號推導出該樣本專屬的隨機數產生器。

    刻意不用全域 random：多進程下全域狀態不可控，會讓結果無法重現。
    """
    return random.Random(config.MASTER_SEED * 1000003 + index)


def pick_difficulty(rng):
    """依 config 的權重抽一個難度等級。"""
    levels = list(config.DIFFICULTY_WEIGHTS)
    weights = [config.DIFFICULTY_WEIGHTS[k] for k in levels]
    return rng.choices(levels, weights=weights, k=1)[0]


# ---------------------------------------------------------------- 背景

def _make_background(w, h, rng):
    """產生紙張／招牌感的背景：底色 ＋ 緩慢漸層 ＋ 細微紋理。

    純白背景會讓模型學到「白底黑字」這個捷徑，真實場景不是這樣，
    所以底色、漸層方向和紋理強度都隨機。
    """
    base = rng.randint(170, 252)
    bg = np.full((h, w, 3), base, dtype=np.float32)

    # 輕微色偏，模擬泛黃紙張或有色招牌
    tint = np.array([rng.uniform(-12, 6), rng.uniform(-8, 8), rng.uniform(-6, 12)],
                    dtype=np.float32)
    bg += tint

    # 線性漸層，模擬打光不均
    if rng.random() < 0.7:
        amp = rng.uniform(5, 28)
        gx = np.linspace(-1, 1, w, dtype=np.float32) * rng.uniform(-1, 1)
        gy = np.linspace(-1, 1, h, dtype=np.float32) * rng.uniform(-1, 1)
        grad = (gy[:, None] + gx[None, :]) * amp
        bg += grad[:, :, None]

    # 細微紋理。注意 GaussianBlur 會把 (h,w,1) 壓成 (h,w)，要補回通道軸才廣播得起來
    noise = np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0, 4, (h, w))
    noise = cv2.GaussianBlur(noise.astype(np.float32), (0, 0), 2.0)
    bg += noise[:, :, None]

    return np.clip(bg, 0, 255).astype(np.uint8)


def _text_color(bg_mean, rng, min_gap=None):
    """挑一個跟背景有足夠對比的文字顏色。

    對比要相對於「實際背景亮度」算，不能用固定區間。用固定區間的話，
    背景 145、文字 95 只差 50 階，再經過模糊和對比調整就低於可讀門檻，
    樣本會被守門丟掉——而且丟掉的多半是難度高的，等於偷偷把資料集變簡單。

    不強制純黑：真實文字有各種深色，對比在下限之上仍然隨機，
    才測得出模型在低對比下的表現。
    """
    if min_gap is None:
        min_gap = config.MIN_TEXT_BG_GAP
    if bg_mean > 140:                     # 淺底深字
        v = rng.randint(0, max(0, int(bg_mean - min_gap)))
    else:                                 # 深底淺字（反白招牌）
        v = rng.randint(min(255, int(bg_mean + min_gap)), 255)
    jitter = lambda: max(0, min(255, v + rng.randint(-18, 18)))
    return (jitter(), jitter(), jitter())


# ---------------------------------------------------------------- 渲染

def render_text(text, font_entry, rng):
    """把一行文字畫成 RGB numpy array（尚未增強）。

    回傳 (image, meta)；字型畫不出來時回傳 (None, 原因)。
    """
    size = config.RENDER_FONT_SIZE
    try:
        font = font_entry.load_pil(size)
    except Exception as e:
        return None, f"字型載入失敗: {e}"

    try:
        bbox = font.getbbox(text)
    except Exception as e:
        return None, f"取字框失敗: {e}"
    if bbox is None:
        return None, "取不到字框"

    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        return None, "字框為零"

    padx, pady = config.PADDING
    padx = int(padx * rng.uniform(0.6, 1.8))
    pady = int(pady * rng.uniform(0.6, 1.8))
    w, h = tw + padx * 2, th + pady * 2

    bg = _make_background(w, h, rng)
    img = Image.fromarray(bg)
    color = _text_color(float(bg.mean()), rng)

    # 字距微調：真實排版不會每個字都等距
    draw = ImageDraw.Draw(img)
    if len(text) > 1 and rng.random() < 0.35:
        x = padx
        for ch in text:
            draw.text((x, pady - bbox[1]), ch, font=font, fill=color)
            adv = font.getlength(ch)
            x += adv + rng.uniform(-adv * 0.04, adv * 0.08)
    else:
        draw.text((padx - bbox[0], pady - bbox[1]), text, font=font, fill=color)

    return np.asarray(img), dict(bg_mean=float(bg.mean()), text_color=color,
                                 render_size=(w, h))


# ---------------------------------------------------------------- 增強

def _stroke_variation(a, rng, amount):
    """用形態學運算改變筆畫粗細。

    這是「模擬手寫／不同書寫工具」的關鍵一步：同一個字用細筆和粗麥克筆寫出來，
    筆畫粗細差很多，而 OCR 模型對這個很敏感。
    """
    if amount <= 0 or rng.random() < 0.4:
        return a
    k = rng.choice([2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # 深字用 erode 變粗、dilate 變細（因為文字是暗的）
    return cv2.erode(a, kernel) if rng.random() < 0.5 else cv2.dilate(a, kernel)


def _elastic(a, rng, strength):
    """彈性形變：平滑的隨機位移場。

    這才是接近手寫的變形——筆畫整體位置和曲度有機地偏移，
    而不是邊緣出現鋸齒。
    """
    if strength <= 0:
        return a
    h, w = a.shape[:2]
    st = np.random.RandomState(rng.randint(0, 2**31 - 1))
    sigma = max(4.0, min(h, w) / 8.0)
    dx = cv2.GaussianBlur(st.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(st.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    # 高斯模糊會大幅衰減振幅，正規化回來才控制得住強度
    for d in (dx, dy):
        m = np.abs(d).max()
        if m > 1e-6:
            d *= strength / m
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(a, xx + dx, yy + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _perspective(a, rng, amount):
    """透視變換，模擬非正面拍攝的角度。"""
    if amount <= 0:
        return a
    h, w = a.shape[:2]
    d = np.array([w, h], dtype=np.float32) * amount
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-d[0], d[0]), rng.uniform(-d[1], d[1])]
                            for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _rotate(a, rng, degrees):
    """小角度旋轉。用邊緣複製填補，避免出現不自然的純色角落。"""
    if degrees <= 0:
        return a
    ang = rng.uniform(-degrees, degrees)
    if abs(ang) < 0.05:
        return a
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _color_contrast(a, rng, contrast_range):
    """對比與亮度調整，模擬曝光差異。

    對比要繞著中間灰縮放，不能直接 a*alpha——後者是往黑色壓縮，
    alpha 小的時候會把整張圖推暗並同時抹平明暗差。實測用錯公式時，
    細字重（ExtraLight）的筆畫會被壓到只剩 11 階對比，等於整個字消失，
    但標註還在，ground truth 就被汙染了。
    """
    lo, hi = contrast_range
    alpha = rng.uniform(lo, hi)
    beta = rng.uniform(-18, 18)
    mid = 128.0
    out = (a.astype(np.float32) - mid) * alpha + mid + beta
    # 輕微的通道獨立增益，模擬白平衡偏差
    if rng.random() < 0.4:
        gain = np.array([rng.uniform(0.94, 1.06) for _ in range(3)], dtype=np.float32)
        out = (out - mid) * gain + mid
    return np.clip(out, 0, 255).astype(np.uint8)


def _blur(a, rng, sigma_max):
    """失焦／動態模糊。"""
    if sigma_max <= 0:
        return a
    if rng.random() < 0.25 and sigma_max > 0.8:
        # 動態模糊：單方向的線性核
        k = rng.choice([3, 5, 7])
        kern = np.zeros((k, k), dtype=np.float32)
        if rng.random() < 0.5:
            kern[k // 2, :] = 1.0
        else:
            kern[:, k // 2] = 1.0
        return cv2.filter2D(a, -1, kern / kern.sum())
    s = rng.uniform(0, sigma_max)
    return cv2.GaussianBlur(a, (0, 0), s) if s > 0.05 else a


def _noise(a, rng, level):
    """感測器雜訊。"""
    if level <= 0:
        return a
    st = np.random.RandomState(rng.randint(0, 2**31 - 1))
    out = a.astype(np.float32) + st.normal(0, level, a.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg_artifacts(a, rng, q_range):
    """先低品質編碼再解碼，留下真實的 JPEG 區塊痕跡。

    這一步很重要：真實世界的圖片幾乎都經過壓縮，
    只在乾淨圖上訓練/評測會高估模型能力。
    """
    lo, hi = q_range
    q = rng.randint(lo, hi)
    ok, enc = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return a
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def augment(a, difficulty, rng):
    """套用整條增強管線。回傳 (image, params)。"""
    p = config.DIFFICULTY_PARAMS[difficulty]
    a = _stroke_variation(a, rng, p["elastic"])
    a = _elastic(a, rng, p["elastic"])
    a = _perspective(a, rng, p["persp"])
    a = _rotate(a, rng, p["rotate"])
    a = _color_contrast(a, rng, p["contrast"])
    a = _blur(a, rng, p["blur"])
    a = _noise(a, rng, p["noise"])
    a = _jpeg_artifacts(a, rng, p["jpeg"])
    return a, dict(difficulty=difficulty, **{k: (list(v) if isinstance(v, tuple) else v)
                                             for k, v in p.items()})


def resize_to_height(a, height=None):
    """縮到 OCR 慣用的文字行高度。縮小用 INTER_AREA 品質最好。"""
    height = height or config.IMAGE_HEIGHT
    h, w = a.shape[:2]
    if h == height:
        return a
    nw = max(8, int(round(w * height / h)))
    interp = cv2.INTER_AREA if h > height else cv2.INTER_CUBIC
    return cv2.resize(a, (nw, height), interpolation=interp)


def gray_contrast(a):
    """穩健的整體對比估計：99% 與 1% 分位差。

    直接用 max-min 會被單一雜訊點帶偏，用分位數才反映實際可讀性。
    """
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    return float(np.percentile(g, 99) - np.percentile(g, 1))


def ink_ratio(a, thresh=20.0, median=0):
    """估計「墨水」占比：偏離**局部**背景的像素比例。

    這是防豆腐字的第二道防線。字型 cmap 有該碼位、但實際 glyph 是空的
    （或渲染因故失敗）時，cmap 檢查攔不到，這裡會攔到。

    用高通濾波而不是「偏離直方圖眾數」：背景有漸層時眾數根本不代表背景，
    同一張圖會量出 0.00 或 0.88 這種毫無意義的值，把好樣本當成豆腐字丟掉。
    文字是高頻、漸層背景是低頻，相減之後只剩筆畫，對深字淺字都成立。

    參數是實測掃出來的。雜訊也是高頻，本來想用中值濾波去掉，但實測發現
    3x3 中值濾波會連細字重的筆畫一起抹平——48px 高的圖上細筆畫只有 1-2px，
    正好是中值濾波的獵物，真實樣本的 1% 分位從 0.0061 掉到 0.0006。
    改成不濾波、把振幅門檻拉到 20 反而乾淨：雜訊振幅小過不了門檻，
    筆畫振幅大留得下來。實測 500 張真實樣本 1% 分位 0.0061，
    500 張純背景對照最高 0.0005，分離度足夠。
    """
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    if median:
        g = cv2.medianBlur(g, median)
    g = g.astype(np.float32)
    sigma = max(2.0, min(g.shape[:2]) / 6.0)
    lowpass = cv2.GaussianBlur(g, (0, 0), sigma)
    return float(np.mean(np.abs(g - lowpass) > thresh))


def make_sample(text, font_entry, index, difficulty=None):
    """產生一張樣本。

    回傳 (image_bgr, meta) 或 (None, 失敗原因)。
    呼叫端負責存檔與寫 metadata。
    """
    rng = rng_for(index)
    difficulty = difficulty or pick_difficulty(rng)

    rgb, meta = render_text(text, font_entry, rng)
    if rgb is None:
        return None, dict(error=meta)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr, aug = augment(bgr, difficulty, rng)
    bgr = resize_to_height(bgr)

    # 可讀性守門：兩個獨立指標都要過。
    # 墨水比例抓「什麼都沒有」，對比抓「有東西但淡到看不見」——
    # 後者是細字重加上重度增強時的典型結果，只看墨水比例會漏掉。
    ratio = ink_ratio(bgr)
    if ratio < config.MIN_INK_RATIO:
        return None, dict(error=f"墨水比例過低 {ratio:.4f}（疑似豆腐字或空白）")
    contrast = gray_contrast(bgr)
    if contrast < config.MIN_CONTRAST:
        return None, dict(error=f"對比過低 {contrast:.0f}（文字被增強抹平）")

    meta.update(aug)
    meta["ink_ratio"] = round(ratio, 4)
    meta["contrast"] = round(contrast, 1)
    meta["size"] = [int(bgr.shape[1]), int(bgr.shape[0])]
    return bgr, meta
