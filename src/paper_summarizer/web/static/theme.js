"use strict";
// テーマ切り替え: システム (OS の設定に従う) / ライト / ダーク。選択はブラウザごとに保存する
(function () {
  const sel = document.getElementById("theme");
  if (!sel) return;
  let saved = "system";
  try { saved = localStorage.getItem("ps-theme") || "system"; } catch (e) {}
  sel.value = ["light", "dark"].includes(saved) ? saved : "system";
  sel.addEventListener("change", () => {
    const v = sel.value;
    if (v === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = v;
    try { localStorage.setItem("ps-theme", v); } catch (e) {}
  });
})();
