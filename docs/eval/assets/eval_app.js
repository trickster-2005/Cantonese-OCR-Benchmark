/* 評測結果瀏覽器。9,860 筆資料，分頁渲染避免一次塞爆 DOM。
   中英文兩版共用，字串由頁面的 window.EV_I18N 提供。 */
(function () {
  "use strict";

  var T = window.EV_I18N || {};
  var root = document.getElementById("eval-browser");
  if (!root) return;

  // 英文版在 docs/eval/en/ 底下，資源實際在 docs/eval/assets/，
  // 路徑深度跟中文版（docs/eval/）不一樣，所以前綴由頁面指定，不能寫死。
  var BASE = T.assetBase || "assets/";

  var PAGE_SIZE = 60;

  var grid = root.querySelector(".egrid");
  var countEl = root.querySelector(".ecount");
  var pagerEl = root.querySelector(".epager");
  var searchEl = root.querySelector(".esearch");

  var state = {
    model: "all", tier: "all", difficulty: "all", correctness: "all",
    pair: "all", sort: "default", q: "", page: 1,
  };

  var items = [];
  var modelDots = { qwen3vl4b: "qwen", internvl35_4b: "internvl" };
  var modelLabel = { qwen3vl4b: "Qwen3-VL 4B", internvl35_4b: "InternVL3.5 4B" };

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function matches(it) {
    if (state.model !== "all" && it.model !== state.model) return false;
    if (state.tier !== "all" && it.tier !== state.tier) return false;
    if (state.difficulty !== "all" && it.difficulty !== state.difficulty) return false;
    if (state.correctness === "correct" && it.acc !== 1) return false;
    if (state.correctness === "incorrect" && it.acc === 1) return false;
    if (state.pair === "pair" && !it.minimal_pair_with) return false;
    if (state.q) {
      var q = state.q;
      if (it.gt.indexOf(q) === -1 && it.prediction.indexOf(q) === -1) return false;
    }
    return true;
  }

  function sortItems(list) {
    var s = state.sort;
    if (s === "default") return list;
    var copy = list.slice();
    var key, dir;
    if (s === "cer_desc") { key = "cer_clipped"; dir = -1; }
    else if (s === "cer_asc") { key = "cer_clipped"; dir = 1; }
    else if (s === "acc_desc") { key = "acc"; dir = -1; }
    else if (s === "acc_asc") { key = "acc"; dir = 1; }
    else if (s === "len_desc") { key = "n_chars"; dir = -1; }
    else if (s === "len_asc") { key = "n_chars"; dir = 1; }
    else return list;
    copy.sort(function (a, b) { return (a[key] - b[key]) * dir; });
    return copy;
  }

  function card(it) {
    var wide = it.tier === "sentence" || it.n_chars > 8;
    var dotCls = modelDots[it.model] || "";
    var ok = it.acc === 1;
    var tags = [
      '<span class="tag' + (ok ? " ok" : " bad") + '">' +
        (ok ? esc(T.correct) : esc(T.incorrect)) + "</span>",
      '<span class="tag">' + esc(T.tier[it.tier] || it.tier) + "</span>",
      '<span class="tag">' + esc(T.diff[it.difficulty] || it.difficulty) + "</span>",
    ];
    if (it.plane === "ExtA" || it.plane === "ExtB+") {
      tags.push('<span class="tag ext">' + esc(it.plane) + "</span>");
    }
    if (it.minimal_pair_with) {
      tags.push('<span class="tag pair">' +
        T.pairWith.replace("%s", esc(it.minimal_pair_with)) + "</span>");
    }
    tags.push('<span class="tag num">CER ' + (it.cer_clipped * 100).toFixed(0) + "%</span>");

    var predHtml = it.prediction
      ? '<span class="' + (ok ? "pred-ok" : "pred-bad") + '">' + esc(it.prediction) + "</span>"
      : '<span class="pred-empty">' + esc(T.emptyOutput) + "</span>";

    return '<figure class="ecard' + (wide ? " wide" : "") + '">' +
      '<div class="eshot"><img loading="lazy" src="' + BASE + 'images/' +
      esc(it.file) + '" alt="' + esc(it.gt) + '" width="' +
      Math.max(60, it.n_chars * 26) + '" height="48"></div>' +
      '<figcaption class="emeta">' +
      '<div class="erow"><span class="elabel">' + esc(T.gt) + '</span>' +
      '<span class="egt">' + esc(it.gt) + '</span></div>' +
      '<div class="erow"><span class="elabel">' + esc(T.pred) + '</span>' +
      predHtml + '</div>' +
      '<div class="etags"><span class="dot ' + dotCls + '"></span>' +
      esc(modelLabel[it.model] || it.model) + tags.join("") + '</div>' +
      '</figcaption></figure>';
  }

  function render() {
    var filtered = sortItems(items.filter(matches));
    var total = filtered.length;
    var pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    if (state.page > pages) state.page = pages;
    if (state.page < 1) state.page = 1;

    var start = (state.page - 1) * PAGE_SIZE;
    var pageItems = filtered.slice(start, start + PAGE_SIZE);

    grid.innerHTML = pageItems.length
      ? pageItems.map(card).join("")
      : '<p class="empty">' + esc(T.noMatch) + "</p>";

    countEl.textContent = T.showing
      .replace("%1", total === 0 ? 0 : start + 1)
      .replace("%2", Math.min(start + PAGE_SIZE, total))
      .replace("%3", total);

    renderPager(pages);
  }

  function renderPager(pages) {
    if (pages <= 1) { pagerEl.innerHTML = ""; return; }
    var p = state.page;
    var html = [];
    html.push(pagerBtn("«", 1, p === 1));
    html.push(pagerBtn("‹", p - 1, p === 1));

    var lo = Math.max(1, p - 2), hi = Math.min(pages, p + 2);
    if (lo > 1) html.push('<span class="epage-ellipsis">…</span>');
    for (var i = lo; i <= hi; i++) {
      html.push('<button class="epage' + (i === p ? " current" : "") +
        '" data-page="' + i + '"' + (i === p ? ' aria-current="true"' : "") +
        ">" + i + "</button>");
    }
    if (hi < pages) html.push('<span class="epage-ellipsis">…</span>');

    html.push(pagerBtn("›", p + 1, p === pages));
    html.push(pagerBtn("»", pages, p === pages));
    pagerEl.innerHTML = html.join("");
  }

  function pagerBtn(label, page, disabled) {
    return '<button class="epage epage-nav" data-page="' + page + '"' +
      (disabled ? " disabled" : "") + ">" + label + "</button>";
  }

  root.addEventListener("click", function (e) {
    var chip = e.target.closest(".chip");
    if (chip) {
      var key = chip.dataset.filter, val = chip.dataset.value;
      state[key] = val;
      state.page = 1;
      root.querySelectorAll('.chip[data-filter="' + key + '"]').forEach(function (o) {
        o.setAttribute("aria-pressed", o === chip ? "true" : "false");
      });
      render();
      return;
    }
    var pageBtn = e.target.closest(".epage");
    if (pageBtn && !pageBtn.disabled) {
      state.page = parseInt(pageBtn.dataset.page, 10);
      render();
      root.scrollIntoView({ block: "start" });
    }
  });

  root.querySelector(".esort").addEventListener("change", function (e) {
    state.sort = e.target.value;
    state.page = 1;
    render();
  });

  var searchTimer = null;
  searchEl.addEventListener("input", function (e) {
    clearTimeout(searchTimer);
    var val = e.target.value.trim();
    searchTimer = setTimeout(function () {
      state.q = val;
      state.page = 1;
      render();
    }, 200);
  });

  fetch(BASE + "eval_data.json")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      items = data;
      render();
    })
    .catch(function () {
      grid.innerHTML = '<p class="empty">' + esc(T.loadFail) + "</p>";
    });
})();
