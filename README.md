# CantoBench-Synth

A synthetic OCR benchmark for written Cantonese (廣東話), and the pipeline that
generates it.

**[English](#english) · [繁體中文](#繁體中文)**

Project page: `https://trickster-2005.github.io/Cantonese-OCR-Benchmark/`
Dataset: `https://huggingface.co/datasets/trickster-2005/cantonese-benchmark-synth`

OCF AI Research Internship 2026.

---

## English

### Why this exists

Everyday writing in Hong Kong uses many characters absent from Standard Written
Chinese: 嘅, 咗, 冇, 喺, 𨋢. Surveying what was already available turned up
three gaps:

- **No image dataset.** There are twenty-odd Cantonese datasets on HuggingFace;
  all of them are text or audio. We did not find one providing Cantonese OCR data
  as images.
- **Existing benchmarks measure language understanding.** CantoNLU, HKCanto-Eval
  and similar work evaluate comprehension and cultural knowledge, not visual
  recognition.
- **Some characters are missing from fonts entirely.** Across the 33 open-licensed
  font instances this pipeline uses, `𠮶` (U+20BB6) is supported by none, and
  `𤗲` (U+245F2) by none either. A character no font carries is one a model never
  had a chance to learn.

The harder problem is not rare characters but ones that resemble common ones.
`冇` ("to not have") differs from `有` ("to have") by two strokes and means the
opposite.

Measured font coverage, for the same 17 common Cantonese characters and 6
extension-block characters:

| Font | Common Cantonese | Extension blocks | Glyphs |
|---|---|---|---|
| Chiron Hei HK (OFL) | 17/17 | 5/6 | 47,724 |
| Noto Sans HK (OFL) | 17/17 | 5/6 | 20,755 |
| MingLiU, MS JhengHei, SimSun, DFKai (Windows built-ins) | 17/17 | **0/6** | ~29,000 |

### Quick start

```bash
pip install -r requirements.txt
```

```bash
python fetch_fonts.py
```

```bash
python build_benchmark.py --total 500
```

```bash
python verify.py
```

Inspect `output/contact_sheet.jpg`, then run the full scale:

```bash
python build_benchmark.py
```

On a server, `./run_server.sh` does all of the above in one step.

Roughly 4,930 images by default. That size is deliberate: generation is not the
bottleneck, **inference is**. Generating 4,930 images takes 25 seconds on an
8-core machine; evaluating them across two VLMs takes about 2.7 hours.

| Images | Generation | 2 models | 8 models |
|---|---|---|---|
| 5,000 | ~25 s | ~2.7 h | ~11 h |
| 20,000 | ~2 min | ~11 h | ~44 h |
| 1,000,000 | ~53 min | infeasible | infeasible |

For a training set, raise `TIER_SIZES` in `config.py`. Measured throughput is
about 200 images/s on 6 workers (AMD EPYC 7R13) and 313/s on 14 workers
(i5-1240P).

### Output

```
output/
├── images/*.jpg          48 px tall text lines
├── test_labels.txt       filename <TAB> label  (TC-STR format)
├── metadata.jsonl        full metadata, one JSON object per line
├── charset_review.tsv    character set for human review
├── contact_sheet.jpg     stratified sample montage
├── dataset_card.md       generated HuggingFace dataset card
└── build_report.json     statistics and reproducibility record
```

Filenames encode the tier, and indices are allocated in contiguous blocks:

| Tier | Prefix | Index range | Images | Distinct texts | Renders each | Length |
|---|---|---|---|---|---|---|
| Character | `char_` | 000000–001168 | 1,162 | 166 | 7.0 | 1 |
| Word | `word_` | 001169–002968 | 1,781 | 600 | 3.0 | 2–4 |
| Sentence | `sentence_` | 002969–004968 | 1,987 | 667 | 3.0 | 4–25 |

Each text is rendered several times with different fonts and difficulty levels,
so per-character accuracy is a statistic rather than a coin flip.

#### metadata.jsonl fields

Shared: `file`, `text`, `tier`, `source`, `font`, `font_family`, `font_license`,
`difficulty`, `ink_ratio`, `contrast`, `width`, `height`, `n_chars`, `role`.

Tier-specific:

| Tier | Extra fields |
|---|---|
| `char` | `plane` (BMP / ExtA / ExtB+), `corpus_freq`, `charset_source`, `minimal_pair_with`, `minimal_pair_note` |
| `word` | `jyutping`, `rime_weight`, `canto_chars` |
| `sentence` | `canto_chars`, `n_canto_chars` |

`role` is `cantonese` or `counterpart` in the character tier. Counterparts are the
Standard Chinese characters of each minimal pair (有, 既, 左, 但 …), included so
there is a baseline: without them, a wrong reading of `冇` cannot be attributed to
either vision or language prior.

### The core experimental design

The same character appears in both the character tier and the sentence tier.

> If a model reads `冇` correctly in isolation but returns `有` inside a sentence,
> the failure is not visual. The language-model prior overrode the glyph.

A sentence-only benchmark cannot separate those two cases. Fifteen minimal pairs
are tagged in the metadata via `minimal_pair_with`:

`冇`/`有`, `嘅`/`既`, `咗`/`左`, `佢`/`但`, `係`/`系`, `哋`/`地`, `喺`/`係`,
`嗰`/`個`, `啲`/`的`, `嚟`/`來`, `睇`/`看`, `諗`/`想`, `乜`/`什`, `唔`/`不`,
`嘢`/`野`.

### Method

#### Corpora

| Source | Licence | Role | Measured core-Cantonese density |
|---|---|---|---|
| [Cantonese Wikipedia](https://huggingface.co/datasets/jed351/cantonese-wikipedia) | **CC BY-SA 4.0** | sentences | 5.53% |
| [HKCanCor](https://pycantonese.org/data.html) | CC BY 4.0 | sentences | 7.80% |
| [rime-cantonese](https://github.com/rime/rime-cantonese) | CC BY 4.0 | characters, words | lexicon |

Density was measured over the characters 嘅咗冇哋喺乜嘢嚟啲嗰咩㗎, which do not
occur in Standard Written Chinese.

**Deliberately excluded:**

| Source | Reason |
|---|---|
| `raptorkwok/cantonese_sentences` | **No licence declared** (no `license` field in the HF card metadata, no licence tag), and the content is forum posts, so copyright rests with the individual posters rather than the uploader. Its measured core-Cantonese density is 4.19%, *lower* than Cantonese Wikipedia, so there is no quality argument for taking the risk. |
| words.hk | "Non-Commercial Open Data License 1.0". **Legally incompatible with CC BY-SA 4.0**: CC BY-SA requires derivatives to permit commercial use, words.hk requires them to forbid it. A combined dataset would have no valid licence at all. |
| rime-cantonese `jyut6ping3.maps` | ODbL-1.0 rather than CC BY 4.0, unlike the other files in that repository. |

#### Character set derivation

```
candidates = corpus characters
           − standard character lists (PRC, Taiwan, Hong Kong)
           − Japanese-only kanji
           − curated exclusions
charset    = candidates ∪ curated core ∪ extension-block set ∪ minimal pairs
           − user exclusions
```

Each filter was validated before being added:

- **Hong Kong primary-school list** removes 9 of 15 Hong Kong variant forms
  (裏, 衞, 佈, 麪, 牀, 羣, 峯, 恆, 着) while removing only one genuine Cantonese
  character, which the curated core list adds back.
- **JIS-minus-Big5** removes Japanese kanji that Japan-related Wikipedia articles
  drag in (鱇, 栃, 峠) with zero false positives on Cantonese characters.
- **Curated exclusions** cover the residue: variant forms (綫, 爲, 敎, 卽),
  Japanese kanji that Big5 also contains, other topolects (`𠊎` is Hakka), and
  rare proper-noun or technical characters.

Subtraction alone would also wrongly delete 睇, 諗, 佢, 靚 and similar characters,
which *are* in the standard lists but carry different Cantonese usage. The curated
core list forces them back in.

The result is 154 characters (128 BMP, 4 Ext-A, 22 Ext-B+; 75 overlapping HKSCS).

**Automation stops at a candidate list.** The final pass is human review:
`output/charset_review.tsv` lists every character with its corpus frequency,
Unicode block, derivation source and a real example sentence. Put unwanted
characters in `data/exclude_chars.txt` and missing ones in
`data/include_chars.txt`, then re-run. At 154 characters this takes about twenty
minutes and is worth it.

#### Fonts

SIL OFL 1.1 only, fetched by `fetch_fonts.py`:
[Chiron Hei HK](https://github.com/chiron-fonts/chiron-hei-hk),
[Chiron Sung HK](https://github.com/chiron-fonts/chiron-sung-hk),
[Noto CJK](https://github.com/notofonts/noto-cjk),
[Hong Kong Character Set](https://github.com/hfhchan/hkcs).
Variable fonts are expanded into five weights each, giving 33 instances from 17
files.

Do not substitute system Chinese fonts. The Windows and macOS built-ins are not
licensed for generating a publicly distributed dataset, and their extension-block
coverage is zero. Font files are never bundled into the dataset; OFL permits
distributing images made with a font, which is what the dataset contains.

#### Augmentation

Rotation, perspective, elastic distortion, stroke-weight variation, blur, noise
and JPEG artefacts, across four difficulty levels (clean / light / medium / hard,
weighted 25 / 35 / 25 / 15).

### Design decisions worth knowing

**Three defences against tofu characters.** The most common and least visible
error in synthetic CJK OCR data is a font lacking a glyph: a blank box is
rendered while the label still claims the character, silently corrupting ground
truth, and the image still "looks like it has something in it". So the font's
cmap is checked before rendering (`fonts.py`), ink ratio and contrast are checked
after (`render.py`), and `verify.py` re-checks glyph coverage independently. In
the 4,930-image release, all 4,930 pass the independent re-check.

**Ink ratio is a high-pass filter, not deviation from the histogram mode.** With a
gradient background the mode does not represent the background at all; the same
image measured 0.00 or 0.88 depending on the draw, discarding good samples as
tofu. Thresholds were swept empirically: over 500 real samples the 1st percentile
is 0.0061, and over 500 background-only controls the maximum is 0.0005.

**Contrast is a second, independent gate.** A thin weight plus heavy augmentation
produces samples that contain something but are too faint to read. One observed
case retained only 11 grey levels of contrast and passed the ink check.

**A 3×3 median filter destroys thin strokes.** It was added to suppress noise
before the high-pass, then removed: at 48 px tall, light-weight strokes are 1–2 px
wide and the median filter erases exactly those. The real-sample 1st percentile
fell from 0.0061 to 0.0006. Raising the amplitude threshold instead is cleaner.

**Contrast scaling must pivot on mid-grey.** `a * alpha` compresses toward black
and flattens light and dark together. Using the wrong formula crushed ExtraLight
strokes to 11 levels of contrast.

**Text colour guarantees a minimum gap from the background.** Sampling colour from
a fixed range gave background 145 against text 95, only 50 levels apart, which
falls below the legibility gate after blur. Rejections then skew toward hard
samples, quietly making the dataset easier than configured.

**The long tail is flattened on purpose.** The character tier allocates its budget
evenly across characters rather than by corpus frequency. Frequency sampling would
fill the benchmark with 係 and 嘅 while `𨋢` and `𠝹` never appear, and those are
the samples that actually separate models.

**Jaggedness is not handwriting.** Aliasing is a product of low resolution and
compression. What approximates handwriting is stroke-weight variation plus
non-rigid local displacement, both of which are implemented. Real handwriting also
needs handwriting fonts, and Cantonese handwriting fonts barely exist, so that is
left for a future version.

**Render large, then shrink.** Text is drawn at 64 px and resampled to 48 px
rather than drawn at 48 px directly. Antialiasing produced by the font engine at
small sizes does not look like real capture or scanning.

**Reproducible.** Each image's RNG derives from `(MASTER_SEED, index)`,
independent of worker count. Verified: the same seed at 4 workers and at 2 workers
produced 252 images with identical SHA-256 hashes.

### Evaluation

```bash
python evaluate.py --limit 20        # smoke test
python evaluate.py                   # full run, resumable
```

Metrics follow OCR convention: **CER** (character error rate, the primary metric
in the literature), **ACC** (exact sequence match), **1−NED** (ICDAR convention),
and character-level **F1**.

Three deliberate choices:

- **The prompt does not mention Traditional Chinese.** Saying so pushes models
  toward Standard Written Chinese, and "Cantonese characters get corrected into
  Standard Chinese" is precisely the phenomenon under test.
- **Post-processing removes format noise only**, never anything model-specific.
  Raw output is kept in `prediction_raw` so anyone can rescore with different
  rules. This is the single largest reason OCR evaluation results fail to
  replicate across groups.
- **Results are written per row and the run is resumable.** A full two-model run
  is hours of work.

### Licensing

| Component | Licence |
|---|---|
| Generated dataset | **CC BY-SA 4.0** |
| Code in this repository | MIT |
| Fonts | SIL OFL 1.1 (not redistributed) |

The dataset is copyleft because Cantonese Wikipedia is CC BY-SA and that
propagates. For a non-copyleft variant, set
`CORPORA["wikipedia"]["enabled"] = False` in `config.py`; the remaining sources
are CC BY 4.0, at the cost of far fewer sentences and a narrower register.

Do not license a dataset under GPL. GPL is a software licence and maps poorly onto
data. Use the Creative Commons family.

### Known limitations

- All images are synthetic. Performance on real Cantonese images needs separate
  validation; a small hand-proofread real evaluation set is the natural next step.
- Horizontal text lines only. Hong Kong signage is frequently vertical.
- Handwriting is approximated by elastic distortion and stroke-weight variation
  rather than being genuine.
- `𠮶` (U+20BB6) and `𤗲` (U+245F2) are in the character set but produce no images,
  because no available font contains them. `build_benchmark.py` reports this
  explicitly rather than dropping them silently.

### Files

| File | Purpose |
|---|---|
| `config.py` | All parameters: scale, difficulty, fonts, corpora, licensing |
| `corpora.py` | Fetch, clean and segment the three corpora |
| `charset.py` | Character-set derivation, minimal pairs, review file |
| `fonts.py` | Font loading and cmap coverage validation |
| `render.py` | Rendering and augmentation |
| `fetch_fonts.py` | Download OFL fonts |
| `build_benchmark.py` | Main entry point |
| `verify.py` | Post-build quality checks |
| `evaluate.py` | Run VLMs over the benchmark |
| `export_samples.py` | Export samples for the project page |
| `run_server.sh` | One-step server workflow |
| `docs/` | Bilingual project page (GitHub Pages), generated from `site/` — **do not hand-edit** |
| `site/content.yaml` | All page text. Edit this. |
| `site/template.html` | Page structure; one template renders both language versions |
| `site/build_site.py` | Compiles `content.yaml` + `template.html` into `docs/*.html` |

### Editing the project page's text

```bash
python site/build_site.py
```

Edit the text in `site/content.yaml`, run the command above, and both
`docs/index.html` and `docs/en/index.html` regenerate. Preview in a browser,
then commit and push as usual.

Do not hand-edit `docs/index.html` or `docs/en/index.html` directly — they are
generated output and get overwritten on the next build. Both language versions
share one template, so a structural change (new section, reordered content)
only needs to be made once.

The build checks that every `{{...}}` placeholder has matching text; a missing
or misspelled key fails the build with the exact key name rather than silently
producing a page with a raw `{{some_key}}` showing on screen.

### Citation

```bibtex
@misc{cantobench2026,
  title  = {CantoBench-Synth: A Synthetic OCR Benchmark for Written Cantonese},
  author = {Yu, Yu-Tang},
  year   = {2026},
  note   = {OCF AI Research Internship},
  url    = {https://github.com/trickster-2005/Cantonese-OCR-Benchmark}
}
```

Please also cite the three upstream corpora.

### Acknowledgement

The dataset generation code and project page were produced with assistance from
Claude (Anthropic). All figures, methods and claims were reviewed and verified by
the author.

---

## 繁體中文

### 為什麼做這個

香港的日常書寫大量使用標準中文沒有的字：嘅、咗、冇、喺、𨋢。盤點現有資源後發現三件事：

- **沒有影像資料集。** HuggingFace 上搜得到的粵語資料集有二十多個，內容都是純文字
  或語音，目前未找到以影像形式提供的粵語 OCR 資料集。
- **既有評測都在測文字理解。** CantoNLU、HKCanto-Eval 等評估的是語言理解與文化知識，
  不是視覺辨識。
- **部分字連字型都缺。** 本管線使用的 33 個開源字型實例中，`𠮶`（U+20BB6）與
  `𤗲`（U+245F2）的支援數都是 0。字型沒收錄的字，模型沒有機會學會。

真正棘手的不是罕見字，而是長得像常用字的那些。`冇`（沒有）和 `有` 只差兩筆，
語義完全相反。

實測字型覆蓋率（17 個常用粵語字、6 個擴展區字）：

| 字型 | 常用粵語字 | 擴展區 | 字符數 |
|---|---|---|---|
| 昭源黑體（OFL） | 17/17 | 5/6 | 47,724 |
| Noto Sans HK（OFL） | 17/17 | 5/6 | 20,755 |
| 新細明體、微軟正黑、宋體、標楷體（Windows 內建） | 17/17 | **0/6** | 約 29,000 |

### 快速開始

```bash
pip install -r requirements.txt
```

```bash
python fetch_fonts.py
```

```bash
python build_benchmark.py --total 500
```

```bash
python verify.py
```

看過 `output/contact_sheet.jpg` 確認品質後，再跑完整規模：

```bash
python build_benchmark.py
```

在 server 上用 `./run_server.sh` 可以一次做完上述所有步驟。

預設約 4,930 張。這個量級是刻意選的：瓶頸不是生成，**是推論**。生成 4,930 張在
8 核機器上只要 25 秒，但兩個 VLM 跑完要約 2.7 小時。

| 張數 | 生成 | 2 個模型 | 8 個模型 |
|---|---|---|---|
| 5,000 | 約 25 秒 | 約 2.7 小時 | 約 11 小時 |
| 20,000 | 約 2 分鐘 | 約 11 小時 | 約 44 小時 |
| 100 萬 | 約 53 分鐘 | 不可行 | 不可行 |

要做訓練集，改 `config.py` 的 `TIER_SIZES` 即可。實測吞吐量：6 workers 約 200 張/秒
（AMD EPYC 7R13）、14 workers 約 313 張/秒（i5-1240P）。

### 產出格式

```
output/
├── images/*.jpg          高度 48px 的文字行
├── test_labels.txt       檔名 <TAB> 標籤（TC-STR 格式）
├── metadata.jsonl        完整 metadata，一行一筆
├── charset_review.tsv    字表人工複核檔
├── contact_sheet.jpg     分層抽樣拼貼圖
├── dataset_card.md       自動產生的 HuggingFace dataset card
└── build_report.json     統計與可重現性記錄
```

檔名前綴就是層級，index 以連續區塊配置：

| 層級 | 前綴 | index 區間 | 張數 | 相異文字 | 每文字張數 | 字數 |
|---|---|---|---|---|---|---|
| 單字 | `char_` | 000000–001168 | 1,162 | 166 | 7.0 | 1 |
| 詞彙 | `word_` | 001169–002968 | 1,781 | 600 | 3.0 | 2–4 |
| 短句 | `sentence_` | 002969–004968 | 1,987 | 667 | 3.0 | 4–25 |

同一個文字會用不同字型與難度重複渲染，這樣「某個字的辨識率」才有統計意義。

#### metadata.jsonl 欄位

共通：`file`、`text`、`tier`、`source`、`font`、`font_family`、`font_license`、
`difficulty`、`ink_ratio`、`contrast`、`width`、`height`、`n_chars`、`role`。

各層獨有：

| 層 | 額外欄位 |
|---|---|
| `char` | `plane`（BMP / ExtA / ExtB+）、`corpus_freq`、`charset_source`、`minimal_pair_with`、`minimal_pair_note` |
| `word` | `jyutping`、`rime_weight`、`canto_chars` |
| `sentence` | `canto_chars`、`n_canto_chars` |

單字層的 `role` 是 `cantonese` 或 `counterpart`。counterpart 是最小對立對的書面語
對照字（有、既、左、但等），刻意收錄作為基準線：沒有它，就無法判斷 `冇` 辨錯究竟是
視覺問題還是語言先驗問題。

### 核心實驗設計

同一個字會同時出現在單字層與句子層。

> 如果模型在單字層把 `冇` 認對、在句子層卻讀成 `有`，那就不是視覺問題，
> 而是語言模型的先驗蓋過了實際看到的字形。

只測句子的評測分不出這兩種情況。metadata 用 `minimal_pair_with` 標記了 15 組對立對：

`冇`/`有`、`嘅`/`既`、`咗`/`左`、`佢`/`但`、`係`/`系`、`哋`/`地`、`喺`/`係`、
`嗰`/`個`、`啲`/`的`、`嚟`/`來`、`睇`/`看`、`諗`/`想`、`乜`/`什`、`唔`/`不`、
`嘢`/`野`。

### 製作方式

#### 語料

| 來源 | 授權 | 角色 | 實測核心粵字密度 |
|---|---|---|---|
| [粵語維基百科](https://huggingface.co/datasets/jed351/cantonese-wikipedia) | **CC BY-SA 4.0** | 句子 | 5.53% |
| [HKCanCor](https://pycantonese.org/data.html) | CC BY 4.0 | 句子 | 7.80% |
| [rime-cantonese](https://github.com/rime/rime-cantonese) | CC BY 4.0 | 字、詞 | 詞庫 |

密度以 嘅咗冇哋喺乜嘢嚟啲嗰咩㗎 這組字測量，它們不出現在書面中文裡。

**刻意排除的來源：**

| 來源 | 理由 |
|---|---|
| `raptorkwok/cantonese_sentences` | **完全沒有授權聲明**（HF card metadata 沒有 `license` 欄位，也沒有 license tag），且內容是論壇貼文，著作權屬於個別發文者而非上傳者。實測核心粵字密度 4.19%，*低於*粵語維基百科，所以連品質理由都不成立。 |
| words.hk | 「Non-Commercial Open Data License 1.0」。**與 CC BY-SA 4.0 法律上不相容**：CC BY-SA 要求衍生作品允許商業使用，words.hk 要求禁止。混用的話沒有任何授權能合法套用。 |
| rime-cantonese `jyut6ping3.maps` | 該檔是 ODbL-1.0，與同一個 repo 的其他檔案（CC BY 4.0）不同。 |

#### 字表推導

```
候選 = 語料出現字
     − 標準字表（大陸、台灣、香港）
     − 日文專用漢字
     − 內建排除表
字表 = 候選 ∪ 人工核心表 ∪ 擴展區字表 ∪ 最小對立對
     − 使用者排除表
```

每道濾網都經實測驗證才加入：

- **香港小學學習字詞表**能濾掉 15 個港式異體字中的 9 個（裏、衞、佈、麪、牀、羣、峯、
  恆、着），只誤殺 1 個真粵字，而該字已在人工核心表中會被補回。
- **JIS 減 Big5** 濾掉維基日本條目帶進來的日文漢字（鱇、栃、峠），對粵語字零誤殺。
- **內建排除表**處理殘留：異體字（綫、爲、敎、卽）、Big5 也收錄的日文漢字、
  其他方言（`𠊎` 是客家話）、罕見專名與技術用字。

單靠差分也會誤刪 睇、諗、佢、靚 這類「收在標準字表裡、但粵語用法不同」的字，
所以人工核心表會強制把它們補回來。

結果是 154 個字（BMP 128、Ext-A 4、Ext-B+ 22；與 HKSCS 交集 75）。

**自動化只能做到候選清單。** 最後一哩是人工複核：`output/charset_review.tsv`
列出每個字的語料頻率、Unicode 分區、推導來源與真實例句。要排除的填進
`data/exclude_chars.txt`，漏掉的填進 `data/include_chars.txt`，重跑即可。
154 個字約 20 分鐘可複核完，很值得。

#### 字型

只用 SIL OFL 1.1，由 `fetch_fonts.py` 下載：
[昭源黑體](https://github.com/chiron-fonts/chiron-hei-hk)、
[昭源宋體](https://github.com/chiron-fonts/chiron-sung-hk)、
[Noto CJK](https://github.com/notofonts/noto-cjk)、
[香港民間字集](https://github.com/hfhchan/hkcs)。
變體字型各展開成五個字重，17 個檔案產生 33 個實例。

不要改用系統中文字型。Windows 與 macOS 內建字型的授權不允許用於產生公開散布的
資料集，而且擴展區覆蓋率是 0。字型檔本身不會打包進資料集；OFL 允許散布用該字型
製作的圖片，而資料集的內容正是圖片。

#### 增強

旋轉、透視、彈性形變、筆畫粗細、模糊、雜訊、JPEG 壓縮痕跡，
分四個難度等級（乾淨／輕微／中等／困難，權重 25／35／25／15）。

### 值得一提的設計決定

**防豆腐字有三道防線。** 合成中文 OCR 資料最常見也最難察覺的錯誤，是字型缺該字形時
畫出空白方框、但標註仍宣稱是那個字，ground truth 被靜默汙染，而圖看起來「有東西」。
所以渲染前查字型 cmap（`fonts.py`）、渲染後查墨水比例與對比（`render.py`）、
`verify.py` 再獨立複查一次字形覆蓋。4,930 張的正式版本，全部 4,930 筆通過獨立複查。

**墨水比例用高通濾波，不是「偏離直方圖眾數」。** 背景有漸層時眾數根本不代表背景，
同一張圖會量出 0.00 或 0.88，把好樣本當豆腐字丟掉。門檻是實測掃出來的：
500 張真實樣本的 1% 分位是 0.0061，500 張純背景對照的最大值是 0.0005。

**對比是獨立的第二道守門。** 細字重加上重度增強會產生「有東西但淡到看不見」的樣本。
實測出現過對比只剩 11 階、卻通過墨水檢查的案例。

**3×3 中值濾波會殺掉細筆畫。** 原本加它來在高通前去雜訊，後來拿掉：48px 高的圖上，
細字重筆畫只有 1–2px 寬，正好是中值濾波的獵物。真實樣本 1% 分位從 0.0061 掉到
0.0006。改成拉高振幅門檻乾淨得多。

**對比縮放必須繞中間灰。** `a * alpha` 是往黑色壓縮，會把明暗一起抹平。
用錯公式時 ExtraLight 字重的筆畫被壓到只剩 11 階對比。

**文字顏色要保證與背景的最低差距。** 用固定區間抽色會出現背景 145 配文字 95，
只差 50 階，經模糊後就低於可讀門檻。而被丟掉的樣本會偏向高難度，
等於偷偷把資料集變簡單。

**長尾是刻意打平的。** 單字層把預算平均分配給每個字，不按語料頻率。按頻率抽的話，
係 和 嘅 會塞滿整個評測集，而 `𨋢`、`𠝹` 一次都不會出現，而後者才是真正能區分模型的
樣本。

**鋸齒不等於手寫。** 鋸齒是低解析度與壓縮的產物。接近手寫的是筆畫粗細變化加上
非剛性的局部位移，兩者都已實作。真手寫還需要手寫字型，而粵語手寫字型幾乎不存在，
留給後續版本。

**先大後小渲染。** 文字以 64px 繪製再重採樣到 48px，而不是直接畫 48px。
字型引擎在小字級產生的抗鋸齒，長得跟真實拍攝或掃描不一樣。

**可重現。** 每張圖的亂數由 `(MASTER_SEED, index)` 推導，與 worker 數量無關。
已驗證：同一 seed 分別以 4 workers 與 2 workers 產出 252 張，SHA-256 完全相同。

### 評測

```bash
python evaluate.py --limit 20        # 小量測試
python evaluate.py                   # 完整執行，可續跑
```

指標採 OCR 慣例：**CER**（字元錯誤率，文獻主指標）、**ACC**（整句完全正確）、
**1−NED**（ICDAR 慣例）、字元級 **F1**。

三個刻意的決定：

- **prompt 不提「繁體中文」。** 提了會把模型推向書面中文，而「粵語字被糾正成書面語字」
  正是要測的現象。
- **後處理只清格式雜訊**，不做任何模型專屬處理。原始輸出保留在 `prediction_raw`，
  任何人都能用別的規則重算。這是 OCR 評測結果無法互相比較的最大單一原因。
- **逐筆寫檔、可續跑。** 兩個模型跑完是數小時的工作。

### 授權

| 項目 | 授權 |
|---|---|
| 產出的資料集 | **CC BY-SA 4.0** |
| 本 repo 的程式碼 | MIT |
| 字型 | SIL OFL 1.1（不隨資料集散布） |

資料集是 copyleft，因為粵語維基百科是 CC BY-SA 且會傳染。若需要非 copyleft 的版本，
在 `config.py` 設 `CORPORA["wikipedia"]["enabled"] = False`；其餘來源都是 CC BY 4.0，
代價是句子數量大幅減少、語域也較窄。

不要用 GPL 授權資料集。GPL 是軟體授權，套到資料上語意不清，應使用 Creative Commons
家族。

### 已知限制

- 全部是合成影像。真實粵語影像上的表現需要另外驗證，下一步應是建立一個小型的
  人工校對真實評測集。
- 只有橫排文字行。香港招牌大量使用直排。
- 手寫只以彈性形變與筆畫粗細變化近似，並非真實手寫。
- `𠮶`（U+20BB6）與 `𤗲`（U+245F2）在字表中但產不出影像，因為沒有任何可用字型收錄。
  `build_benchmark.py` 會明確報告，而不是靜默丟棄。

### 檔案

| 檔案 | 用途 |
|---|---|
| `config.py` | 所有參數：規模、難度、字型、語料、授權 |
| `corpora.py` | 三個語料的抓取、清理、切句 |
| `charset.py` | 字表推導、最小對立對、複核檔 |
| `fonts.py` | 字型載入與 cmap 覆蓋驗證 |
| `render.py` | 渲染與增強 |
| `fetch_fonts.py` | 下載 OFL 字型 |
| `build_benchmark.py` | 主程式 |
| `verify.py` | 產出後品質檢查 |
| `evaluate.py` | 在評測集上跑 VLM |
| `export_samples.py` | 匯出專案頁用的樣本 |
| `run_server.sh` | server 一鍵流程 |
| `docs/` | 雙語專案頁（GitHub Pages），由 `site/` 自動產生，**不要手改** |
| `site/content.yaml` | 網站上所有文字，改這個 |
| `site/template.html` | 網站版面結構，一份模板同時產生中英兩版 |
| `site/build_site.py` | 把 `content.yaml` + `template.html` 編譯成 `docs/*.html` |

### 改網站文字

```bash
python site/build_site.py
```

流程：改 `site/content.yaml` 裡的文字 → 執行上面這行 → `docs/index.html` 和
`docs/en/index.html` 自動重新產生 → 瀏覽器打開確認 → 照平常方式 commit + push。

`docs/index.html` 和 `docs/en/index.html` 兩個檔案本身不要手改——它們是產出物，
下次跑腳本會被整個覆蓋掉。中英文共用同一份 `template.html`，改版面結構（章節順序、
新增區塊）只要改一個地方，兩個語言版本會一起更新，不會再有改了中文忘記改英文的問題。

`content.yaml` 開頭有詳細的編輯說明。腳本會檢查每個 `{{...}}` 標記都有對應的文字，
少填或打錯字會直接報錯並告訴你缺哪一個，不會產生一個看起來正常、但畫面上出現
`{{some_key}}` 這種沒解開的網頁。

### 引用

見上方英文段落的 BibTeX。請同時引用三個上游語料來源。

### 聲明

資料集生成程式與專案頁由 Claude（Anthropic）輔助產出，
所有數據、方法與敘述均經作者人工審核確認。
