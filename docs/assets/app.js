/* 側邊目錄的捲動高亮。與樣本瀏覽器獨立，任一段失敗都不影響另一段。 */
(function () {
  "use strict";
  var links = [].slice.call(document.querySelectorAll(".toc-link"));
  if (!links.length || !("IntersectionObserver" in window)) return;

  var map = Object.create(null);
  var targets = [];
  links.forEach(function (a) {
    var id = (a.getAttribute("href") || "").replace(/^#/, "");
    var el = id && document.getElementById(id);
    if (el) { map[id] = a; targets.push(el); }
  });

  // 記錄每個區塊目前是否在視窗內，取最上面那個當作 active。
  // 只看「最後一個進入視窗的」會在快速捲動時跳來跳去。
  var visible = Object.create(null);
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { visible[e.target.id] = e.isIntersecting; });
    var active = null;
    for (var i = 0; i < targets.length; i++) {
      if (visible[targets[i].id]) { active = targets[i].id; break; }
    }
    if (!active) return;
    links.forEach(function (a) { a.classList.remove("is-active"); });
    if (map[active]) map[active].classList.add("is-active");
  }, { rootMargin: "-15% 0px -70% 0px", threshold: 0 });

  targets.forEach(function (t) { io.observe(t); });
})();

/* 樣本瀏覽器。中英文兩版共用，字串由頁面的 window.CB_I18N 提供。 */
(function () {
  "use strict";

  var T = window.CB_I18N || {};
  var root = document.getElementById("browser");
  if (!root) return;

  // 英文版在 /en/ 底下，資產仍在網站根目錄的 /assets/，
  // 所以路徑前綴必須由頁面指定，不能寫死相對路徑。
  var BASE = T.assetBase || "assets/";

  var grid = root.querySelector(".grid");
  var countEl = root.querySelector(".count");
  var state = { tier: "all", difficulty: "all", plane: "all", zoom: false };
  var items = [];

  // 粵語特有字：用來在 ground truth 裡標示哪些字是重點。
  // 由 samples.json 裡 char 層的樣本反推，不必另外維護一份清單。
  var cantoChars = Object.create(null);

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function highlight(text) {
    var out = "";
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      // 代理對：擴展區字元佔兩個 UTF-16 單位，要一起取
      var cp = text.codePointAt(i);
      if (cp > 0xffff) { ch = text.substr(i, 2); i++; }
      out += cantoChars[ch]
        ? '<span class="hl">' + esc(ch) + "</span>"
        : esc(ch);
    }
    return out;
  }

  function matches(it) {
    if (state.tier !== "all" && it.tier !== state.tier) return false;
    if (state.difficulty !== "all" && it.difficulty !== state.difficulty) return false;
    if (state.plane === "ext") {
      if (it.plane !== "ExtA" && it.plane !== "ExtB+") return false;
    } else if (state.plane === "pair") {
      if (!it.pair_with) return false;
    } else if (state.plane !== "all" && it.plane !== state.plane) return false;
    return true;
  }

  function card(it) {
    var wide = it.tier === "sentence" || it.width > 320;
    var tags = [
      '<span class="tag">' + esc(T.tier[it.tier] || it.tier) + "</span>",
      '<span class="tag">' + esc(T.diff[it.difficulty] || it.difficulty) + "</span>",
    ];
    if (it.plane === "ExtA" || it.plane === "ExtB+") {
      tags.push('<span class="tag ext">' + esc(it.plane) + "</span>");
    }
    if (it.pair_with) {
      var lbl = it.pair_role === "cantonese"
        ? T.pairCanto.replace("%s", it.pair_with)
        : T.pairStd.replace("%s", it.pair_with);
      tags.push('<span class="tag pair">' + esc(lbl) + "</span>");
    }
    if (it.source && it.source !== "charset") {
      tags.push('<span class="tag">' + esc(it.source) + "</span>");
    }
    tags.push('<span class="tag">' + esc(it.font) + "</span>");

    return '<figure class="card' + (wide ? " wide" : "") +
      (state.zoom ? " zoom" : "") + '">' +
      '<div class="shot"><img loading="lazy" src="' + BASE + 'samples/' +
      esc(it.file) + '" alt="' + esc(it.text) + '" width="' +
      (it.width || 200) + '" height="48"></div>' +
      '<figcaption class="meta"><div class="gt">' + highlight(it.text) +
      '</div><div class="tags">' + tags.join("") + "</div></figcaption></figure>";
  }

  function render() {
    var shown = items.filter(matches);
    grid.innerHTML = shown.length
      ? shown.map(card).join("")
      : '<p class="empty">' + esc(T.noMatch) + "</p>";
    countEl.textContent = T.showing
      .replace("%1", shown.length).replace("%2", items.length);
  }

  root.addEventListener("click", function (e) {
    var b = e.target.closest(".chip");
    if (!b) return;
    var key = b.dataset.filter, val = b.dataset.value;
    if (key === "zoom") {
      state.zoom = !state.zoom;
      b.setAttribute("aria-pressed", state.zoom ? "true" : "false");
      render();
      return;
    }
    state[key] = val;
    root.querySelectorAll('.chip[data-filter="' + key + '"]').forEach(function (o) {
      o.setAttribute("aria-pressed", o === b ? "true" : "false");
    });
    render();
  });

  fetch(BASE + "samples.json")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      items = data;
      data.forEach(function (it) {
        if (it.tier === "char" && it.role === "cantonese") cantoChars[it.text] = 1;
      });
      render();
    })
    .catch(function () {
      grid.innerHTML = '<p class="empty">' + esc(T.loadFail) + "</p>";
    });
})();
