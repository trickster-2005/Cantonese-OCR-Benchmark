# -*- coding: utf-8 -*-
"""下載 OFL 授權字型。

為什麼要自己下載，不用系統字型：

  1. 系統上不一定有中文字型（乾淨的 Ubuntu server 通常一個都沒有）
  2. Windows／macOS 內建的中文字型（微軟正黑、新細明體、標楷體）授權不允許
     拿來產生公開散布的資料集
  3. 實測 Windows 內建 CJK 字型對擴展區粵語字的覆蓋是 0/4，
     而 OFL 的昭源黑體是 3/4——授權乾淨的選擇剛好品質也最好

這裡下載的全部是 SIL OFL 1.1，可自由使用、散布、商用。
OFL 允許散布用該字型製作的文件與圖片，所以產出的資料集沒有問題；
但字型檔本身不會打包進資料集，只在文件裡標註來源。

用法：
    python fetch_fonts.py            # 下載到 config.FONTS_DIR
    python fetch_fonts.py --verify   # 只檢查已下載的字型，不重新下載
"""

import argparse
import os
import sys
import urllib.request

import config


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_one(url, dest):
    """下載單一字型檔。已存在且非空就跳過。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return "skip", os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "cantobench/0.1"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    size = os.path.getsize(tmp)
    if size < 1024:
        os.remove(tmp)
        raise RuntimeError(f"下載結果過小（{size} bytes），可能是錯誤頁面")
    os.replace(tmp, dest)
    return "ok", size


def fetch_all():
    """依 config.FONT_SOURCES 下載全部字型。回傳 (成功數, 失敗清單)。"""
    os.makedirs(config.FONTS_DIR, exist_ok=True)
    ok = 0
    failed = []
    for src in config.FONT_SOURCES:
        print(f"\n[fonts] {src['name']}  ({src['license']})  {src['home']}")
        for f in src["files"]:
            url = src["url"].format(f=f)
            dest = os.path.join(config.FONTS_DIR, src["name"], os.path.basename(f))
            try:
                status, size = download_one(url, dest)
                mark = "已存在" if status == "skip" else "下載完成"
                print(f"        {mark:8} {os.path.basename(f):32} {_human(size):>10}")
                ok += 1
            except Exception as e:
                print(f"        失敗     {os.path.basename(f):32} {e}", file=sys.stderr)
                failed.append((src["name"], f, str(e)))
    return ok, failed


def write_font_licenses():
    """在字型目錄寫一份授權說明，方便日後查證與寫 dataset card。"""
    path = os.path.join(config.FONTS_DIR, "FONT_LICENSES.md")
    lines = [
        "# 字型授權",
        "",
        "本目錄的字型全部為 **SIL Open Font License 1.1 (OFL-1.1)**。",
        "",
        "OFL 允許自由使用、修改與散布，唯一限制是不得單獨販售字型檔本身。",
        "用 OFL 字型渲染出來的圖片不受限制，因此本資料集的影像可以自由散布。",
        "",
        "字型檔本身**不隨資料集散布**，使用者請自行從下列來源取得。",
        "",
        "| 字型 | 授權 | 來源 |",
        "|---|---|---|",
    ]
    for src in config.FONT_SOURCES:
        lines.append(f"| {src['name']} | {src['license']} | {src['home']} |")
    lines += ["", "取得方式：", "", "```bash", "python fetch_fonts.py", "```", ""]
    os.makedirs(config.FONTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[fonts] 授權說明已寫入 {path}")


def verify():
    """載入字型池並回報覆蓋率。這是下載後該做的健康檢查。"""
    import fonts
    pool, report = fonts.load_font_pool(verbose=True)

    print(f"\n[fonts] 字型家族：{', '.join(report['families'])}")
    print("\n=== 擴展區粵語字支援度（決定哪些字進得了 benchmark）===")
    for c in fonts.PROBE_EXT:
        supp = [e for e in pool if ord(c) in e.cmap]
        bar = "#" * int(20 * len(supp) / max(1, len(pool)))
        print(f"  {c}  U+{ord(c):05X}  {len(supp):>3}/{len(pool):<3} {bar}")

    if report["with_full_ext"] == 0:
        print("\n[fonts] 警告：沒有任何字型完整覆蓋擴展區粵語字。"
              "benchmark 仍可產生，但 Ext-B 樣本會很少。", file=sys.stderr)
    return pool, report


def main():
    ap = argparse.ArgumentParser(description="下載 OFL 授權中文字型")
    ap.add_argument("--verify", action="store_true",
                    help="只檢查已下載的字型覆蓋率，不重新下載")
    args = ap.parse_args()

    if not args.verify:
        ok, failed = fetch_all()
        write_font_licenses()
        print(f"\n[fonts] 完成：{ok} 個檔案就緒，{len(failed)} 個失敗")
        if failed:
            print("[fonts] 失敗清單：", file=sys.stderr)
            for name, f, err in failed:
                print(f"        {name}/{f}: {err}", file=sys.stderr)

    print()
    verify()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
