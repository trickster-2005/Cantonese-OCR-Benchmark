# -*- coding: utf-8 -*-
"""從 site/content.yaml + site/template.html 產生 docs/index.html 與 docs/en/index.html。

編輯網站文字的正確流程：

    1. 改 site/content.yaml 裡的文字
    2. 執行 python site/build_site.py
    3. 打開 docs/index.html 看結果（或用瀏覽器開 docs/ 目錄起個本機伺服器）
    4. 滿意的話照平常方式 git add / commit / push

不要直接手改 docs/index.html 或 docs/en/index.html —— 這兩個檔案是自動產生的，
下次執行這支腳本會被整個覆蓋掉，手改的東西會消失。

版面結構（章節順序、CSS class）要改的話動 site/template.html；
顏色、字體、卡片排版動 docs/assets/style.css；這支腳本只管文字代換。

設計上的考量：
    content.yaml 裡的值可以再引用 {{shared.xxx}}（例如 footer 那幾行會嵌入
    {{shared.licence_dataset}}），所以替換要跑多輪直到不再變化，而不是只替換
    模板一次就結束——不然像 "資料集：{{shared.licence_dataset}}" 這種值
    代入模板後，畫面上會直接看到沒被解開的 {{shared.licence_dataset}} 字樣。

    最後一定要檢查產出的 HTML 裡有沒有殘留 {{...}}——這通常代表 content.yaml
    漏了某個 key，或者 template.html 裡的 token 名字打錯字。兩種情況都不該
    悄悄產生一個壞掉的網頁，而是直接報錯、列出缺哪些 key，逼你立刻修正。
"""

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("需要 pyyaml：pip install pyyaml（或 pip install -r requirements.txt）")

ROOT = Path(__file__).resolve().parent.parent
CONTENT_FILE = ROOT / "site" / "content.yaml"
TEMPLATE_FILE = ROOT / "site" / "template.html"

TOKEN_RE = re.compile(r"\{\{([a-zA-Z0-9_.]+)\}\}")

# 每個語言版本的結構性設定：資產路徑前綴、hreflang、語言切換鈕目標等。
# 這些是「網站骨架」的一部分，不是文字內容，所以寫死在這裡而不是 content.yaml——
# 一般編輯文字不需要碰這裡；真的要新增第三個語言版本才需要改。
PAGES = [
    dict(
        lang="zh", html_lang="zh-Hant-TW", root="",
        zh_href="./", en_href="en/", xdefault_href="./",
        lang_float_href="en/", lang_float_hreflang="en",
        eval_href="eval/",
        out=ROOT / "docs" / "index.html",
    ),
    dict(
        lang="en", html_lang="en", root="../",
        zh_href="../", en_href="./", xdefault_href="../",
        lang_float_href="../", lang_float_hreflang="zh-Hant",
        eval_href="../eval/en/",
        out=ROOT / "docs" / "en" / "index.html",
    ),
]


def flatten(d, prefix=""):
    """把巢狀 dict 攤平成 dotted-key 的一層 dict。

    shared.urls.hf 這種巢狀結構，攤平後 key 就是字面上的 "shared.urls.hf"，
    跟 template.html 裡寫的 {{shared.urls.hf}} 直接對應，不需要真的做屬性
    路徑查找。
    """
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            # YAML 的 "|" 區塊純量結尾會帶一個換行，那是格式而不是內容，去掉。
            # 中間的換行保留——瀏覽器渲染 HTML 時本來就會把換行當空白處理，
            # 跟原本手寫在 <p> 標籤裡的多行原始碼視覺效果一樣。
            out[key] = str(v).rstrip("\n")
    return out


def substitute_once(text, ctx):
    return TOKEN_RE.sub(lambda m: ctx.get(m.group(1), m.group(0)), text)


def resolve_content_values(ctx, max_rounds=5):
    """content.yaml 裡的值本身可能還內嵌 {{shared.xxx}} token（例如 footer 那幾行），
    要先把這些值自己解析乾淨，模板代換那一步才不會留下沒展開的 token。
    """
    resolved = dict(ctx)
    for _ in range(max_rounds):
        changed = False
        for k, v in list(resolved.items()):
            nv = substitute_once(v, resolved)
            if nv != v:
                resolved[k] = nv
                changed = True
        if not changed:
            break
    else:
        sys.exit("[build_site] content.yaml 裡的值互相循環引用了，解不開，請檢查 shared 段落")
    return resolved


def build_page(page, shared_flat, lang_flat, template_text):
    ctx = {**shared_flat, **{k: v for k, v in page.items() if k not in ("lang", "out")}}
    ctx.update(lang_flat)
    ctx = resolve_content_values(ctx)

    html = substitute_once(template_text, ctx)

    leftover = sorted(set(TOKEN_RE.findall(html)))
    if leftover:
        sys.exit(
            f"[build_site] {page['lang']} 版缺少下列內容，"
            f"請在 site/content.yaml 的 shared/{page['lang']} 區塊補上：\n  "
            + "\n  ".join(leftover)
        )

    page["out"].parent.mkdir(parents=True, exist_ok=True)
    page["out"].write_text(html, encoding="utf-8", newline="\n")
    return len(html.encode("utf-8"))


def main():
    if not CONTENT_FILE.exists():
        sys.exit(f"找不到 {CONTENT_FILE}")
    if not TEMPLATE_FILE.exists():
        sys.exit(f"找不到 {TEMPLATE_FILE}")

    data = yaml.safe_load(CONTENT_FILE.read_text(encoding="utf-8"))
    template_text = TEMPLATE_FILE.read_text(encoding="utf-8")

    shared_flat = flatten(data.get("shared", {}), "shared")

    # 順手檢查一下 content.yaml 裡有沒有 template.html 完全用不到的 key——
    # 不是錯誤，但通常代表改版面時忘記把舊文字一起清掉，提醒一下比較好。
    used_tokens = set(TOKEN_RE.findall(template_text))

    for page in PAGES:
        lang_data = data.get(page["lang"])
        if lang_data is None:
            sys.exit(f"[build_site] content.yaml 沒有 {page['lang']}: 區塊")
        lang_flat = flatten(lang_data)

        unused = sorted(set(lang_flat) - used_tokens)
        if unused:
            print(f"[build_site] 提醒：{page['lang']} 版 content.yaml 裡這些 key "
                  f"template.html 沒有用到，可能是改版面時忘記清掉：")
            for k in unused:
                print(f"    {k}")

        size = build_page(page, shared_flat, lang_flat, template_text)
        print(f"[build_site] {page['lang']:2}  {page['out'].relative_to(ROOT)}  "
              f"{size/1024:.1f} KB")

    print("[build_site] 完成。打開 docs/index.html 看結果，"
          "或在 docs/ 目錄跑 python -m http.server 之後用瀏覽器開 http://localhost:8000/")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
