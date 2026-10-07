"use strict";
const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";

function toast(msg) {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg; t.hidden = false;
  clearTimeout(t._timer); t._timer = setTimeout(() => (t.hidden = true), 4000);
}

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "X-CSRF-Token": csrf }, body });
  const data = await r.json().catch(() => ({ ok: false, error: `HTTP ${r.status}` }));
  if (r.status === 401) location.href = "/login";
  return data;
}

function renderMath(root) {
  if (window.renderMathInElement && root) {
    renderMathInElement(root, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }],
                                throwOnError: false });
  }
}

// ダッシュボードの自動更新
const statusBox = document.getElementById("status");
if (statusBox) {
  setInterval(async () => {
    const r = await fetch(statusBox.dataset.src);
    if (r.status === 401) { location.href = "/login"; return; }
    if (r.ok) statusBox.innerHTML = await r.text();
  }, 30000);
}

// 今すぐ処理
document.getElementById("run-now")?.addEventListener("click", async (e) => {
  e.target.disabled = true;
  const d = await post("/api/run");
  toast(d.ok ? "処理を開始しました" : (d.error || "開始できませんでした"));
  setTimeout(() => (e.target.disabled = false), 5000);
});

// 再試行・再処理
document.querySelectorAll("button[data-action]").forEach((b) => b.addEventListener("click", async () => {
  if (b.dataset.confirm && !confirm(b.dataset.confirm)) return;
  b.disabled = true;
  const d = await post(`/api/paper/${b.dataset.id}/${b.dataset.action}`);
  toast(d.ok ? d.message : (d.error || "失敗しました"));
  if (!d.ok) b.disabled = false;
}));

// 投稿 (ドラッグ & ドロップ)
const drop = document.getElementById("drop");
if (drop) {
  const input = document.getElementById("files");
  const results = document.getElementById("upload-results");
  const project = document.getElementById("project");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => send(e.dataTransfer.files));
  input.addEventListener("change", () => { send(input.files); input.value = ""; });

  async function send(files) {
    if (!files.length) return;
    if (!project.reportValidity()) return;
    for (const f of files) {
      const li = document.createElement("li");
      li.className = "wait"; li.textContent = `${f.name} — 送信中…`;
      results.prepend(li);
      const fd = new FormData();
      fd.append("project", project.value.trim());
      fd.append("files", f);
      const d = await post("/api/upload", fd);
      if (!d.ok) { li.className = "ng"; li.textContent = `${f.name} — ${d.error}`; continue; }
      const r = d.results[0];
      li.className = r.ok ? "ok" : "ng";
      li.textContent = `${r.ok ? "✓" : "✕"} ${r.project} / ${r.name} — ${r.message}`;
    }
  }
}

// 要約の数式描画と、根拠表示 [§…] の原文表示
const doc = document.querySelector(".doc");
renderMath(doc);
const ev = document.getElementById("evidence");
if (doc && ev) {
  const body = document.getElementById("ev-body");
  document.getElementById("ev-close").addEventListener("click", () => (ev.hidden = true));
  doc.addEventListener("click", async (e) => {
    const a = e.target.closest("a.cite");
    if (!a) return;
    e.preventDefault();
    const r = await fetch(`/api/paper/${doc.dataset.paper}/evidence?ids=${encodeURIComponent(a.dataset.ids)}`);
    const d = await r.json();
    body.replaceChildren();
    if (!d.ok) { body.textContent = d.error || "取得できませんでした"; ev.hidden = false; return; }
    for (const it of d.items) {
      const div = document.createElement("div"); div.className = "evitem";
      const id = document.createElement("div"); id.className = "id";
      id.textContent = `${it.id} · §${it.section} ${it.section_title}`;
      const p = document.createElement("div"); p.textContent = it.text;
      div.append(id, p); body.append(div);
    }
    ev.hidden = false;
    renderMath(body);
  });
}
