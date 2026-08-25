# 資料集授權

本專案**產出的資料集**（影像與標註）授權為
**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**。

https://creativecommons.org/licenses/by-sa/4.0/

## 為什麼是 CC BY-SA 而不是更寬鬆的授權

資料集的文字內容衍生自粵語維基百科，其授權為 CC BY-SA 4.0。
CC BY-SA 具有 copyleft（傳染性）：任何衍生作品都必須以相同授權釋出。
因此整個資料集受此約束，無法改用 CC BY 4.0 或 CC0。

若需要非 copyleft 的版本，可在 `config.py` 中關閉 wikipedia 來源
（`CORPORA["wikipedia"]["enabled"] = False`），僅使用 HKCanCor 與
rime-cantonese（皆為 CC BY 4.0），產出的資料集即可用 CC BY 4.0 釋出。
代價是句層樣本會少很多，且語域侷限於口語轉寫。

## 語料出處與歸屬

| 來源 | 授權 | 歸屬 |
|---|---|---|
| 粵語維基百科 | CC BY-SA 4.0 | Cantonese Wikipedia contributors, zh-yue.wikipedia.org |
| HKCanCor | CC BY 4.0 | Hong Kong Cantonese Corpus, Luke & Wong (2015) |
| rime-cantonese | CC BY 4.0 | CanCLID, github.com/rime/rime-cantonese |

## 字型

影像以 SIL Open Font License 1.1 授權的字型渲染：
昭源黑體、昭源宋體、Noto CJK、香港民間字集。

OFL 明確允許散布使用該字型製作的文件與圖片，因此本資料集的影像不受字型授權限制。
字型檔本身**不隨資料集散布**，使用者請自行從原始來源取得（見 `fetch_fonts.py`）。

## 程式碼

生成程式碼以 MIT 授權釋出，見 `LICENSE`。程式碼與資料集的授權是分開的。

## 使用建議

引用時請同時標註本資料集與上述三個上游語料來源。
