// groundwork-rag web page.
// Security note: answers and chunks come from documents, which are untrusted input.
// This file never uses innerHTML; all text is inserted with textContent / DOM nodes,
// so text that looks like HTML or <script> stays plain text (no XSS).

const $ = (id) => document.getElementById(id);

const EXAMPLES = [
  "How does LoRA reduce the number of trainable parameters?",
  "What is QLoRA?",
  "What is dark knowledge in distillation?",
  "Can attackers plant malicious text in the documents a chatbot reads?",
  "Give me a recipe for chocolate cake.",
];

const REFUSALS = {
  low_similarity: (data) => {
    const best = (key) => Math.max(0, ...data.retrieved.map((h) => h[key] ?? 0));
    let text = `No chunk reached the meaning threshold (best ${fmt(best("score"))} < ${data.min_similarity.toFixed(3)})`;
    if (data.retrieved.some((h) => h.keyword != null)) {
      text += ` or the keyword threshold (best ${best("keyword").toFixed(2)} < ${data.min_keyword.toFixed(2)})`;
    }
    return `${text}, so the model was not called.`;
  },
  no_answer_in_sources: () =>
    "The retrieved chunks were on topic, but the model found no answer in them.",
};

// Small helper to build elements without innerHTML.
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child != null) node.append(child);
  }
  return node;
}

const fmt = (n) => n.toFixed(3);

// --- Answer rendering ------------------------------------------------------

// Inline formatting: **bold**, $math$ (shown as monospace) and [n] citations.
// Everything else is plain text.
function renderInline(text, validNumbers) {
  const frag = document.createDocumentFragment();
  const parts = text.split(/(\*\*[^*]+\*\*|\$[^$\n]+\$|\[\d+(?:\s*,\s*\d+)*\])/g);
  for (const part of parts) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      frag.append(el("strong", { text: part.slice(2, -2) }));
    } else if (part.startsWith("$") && part.endsWith("$") && part.length > 2) {
      frag.append(el("code", { class: "math", text: part.slice(1, -1) }));
    } else if (/^\[\d/.test(part)) {
      for (const n of part.slice(1, -1).split(",").map((s) => s.trim())) {
        if (validNumbers.has(Number(n))) {
          const link = el("a", { class: "cite", href: `#src-${n}`, text: n, "aria-label": `source ${n}` });
          link.addEventListener("click", (e) => { e.preventDefault(); focusSource(n); });
          frag.append(link);
        } else {
          frag.append(`[${n}]`);
        }
      }
    } else {
      frag.append(part);
    }
  }
  return frag;
}

// Blocks: paragraphs separated by blank lines, and "- " / "1. " lists.
function renderAnswer(text, validNumbers) {
  const container = document.createDocumentFragment();
  for (const block of text.trim().split(/\n\s*\n/)) {
    const lines = block.split("\n").filter((l) => l.trim());
    const bullet = /^\s*([-*•]|\d+[.)])\s+/;
    if (lines.length && lines.every((l) => bullet.test(l))) {
      const ordered = /^\s*\d/.test(lines[0]);
      const list = el(ordered ? "ol" : "ul");
      for (const line of lines) list.append(el("li", {}, renderInline(line.replace(bullet, ""), validNumbers)));
      container.append(list);
    } else {
      const p = el("p");
      lines.forEach((line, i) => {
        if (i) p.append(el("br"));
        p.append(renderInline(line, validNumbers));
      });
      container.append(p);
    }
  }
  return container;
}

function focusSource(n) {
  const card = $(`src-${n}`);
  if (!card) return;
  card.querySelector("details")?.setAttribute("open", "");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1200);
}

function sourceCard(source) {
  const c = source.chunk;
  return el("li", { class: "source", id: `src-${source.number}` },
    el("div", { class: "source-head" },
      el("span", { class: "cite", text: String(source.number) }),
      el("div", { class: "source-meta" },
        el("span", { class: "source-title", text: c.title }),
        el("span", { class: "source-loc", text: `${c.location} · similarity ${fmt(source.score)}` }),
      ),
    ),
    el("details", {},
      el("summary", { text: "Show passage" }),
      el("p", { class: "passage", text: c.text }),
    ),
  );
}

// --- Retrieval inspector ---------------------------------------------------

function scoreBar(label, value, threshold, display) {
  const pct = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;
  return el("div", { class: "signal" },
    el("span", { class: "signal-label", text: label }),
    el("div", { class: "bar", role: "img", "aria-label": `${label} ${display}, threshold ${threshold}` },
      el("div", { class: "fill", style: `width:${pct(value)}` }),
      el("div", { class: "thr", style: `left:${pct(threshold)}` }),
    ),
    el("span", { class: "score", text: display }),
  );
}

function hitRow(hit, data) {
  const c = hit.chunk;
  const signals = [scoreBar("meaning", hit.score, data.min_similarity, fmt(hit.score))];
  if (hit.keyword != null) {
    signals.push(scoreBar("keywords", hit.keyword, data.min_keyword, hit.keyword.toFixed(2)));
  }
  return el("li", { class: `hit ${hit.used ? "used" : "below"}` },
    el("div", { class: "hit-head" },
      el("span", { class: "rank", text: String(hit.rank) }),
      el("span", { class: "hit-src", title: c.title, text: c.title }),
      el("span", { class: "hit-loc", title: c.location, text: c.location }),
    ),
    ...signals,
    el("details", {},
      el("summary", { text: hit.used ? "Show chunk (sent to the model)" : "Show chunk (not used)" }),
      el("p", { class: "passage", text: c.text }),
    ),
  );
}

function renderInspector(data) {
  $("inspector-empty").hidden = true;
  $("inspector").hidden = false;
  $("retrieved").replaceChildren(...data.retrieved.map((h) => hitRow(h, data)));

  const t = data.timings;
  const rows = [
    ["Model", data.model ?? "not called"],
    ["Retrieval", `${Math.round(t.retrieval_ms)} ms`],
  ];
  if (t.generation_ms != null) rows.push(["Generation", `${(t.generation_ms / 1000).toFixed(1)} s`]);
  $("timings").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));
}

// --- Asking ----------------------------------------------------------------

function render(data) {
  const valid = new Set(data.sources.map((s) => s.number));
  $("answer").hidden = false;
  $("answer-question").textContent = data.question;
  $("answer-body").replaceChildren(renderAnswer(data.answer, valid));

  const refusal = $("refusal");
  refusal.hidden = !data.refused;
  if (data.refused) {
    refusal.textContent = REFUSALS[data.refusal_reason](data);
  }
  $("sources-title").hidden = data.sources.length === 0;
  $("sources").replaceChildren(...data.sources.map(sourceCard));
  renderInspector(data);
}

let timer = null;
function setLoading(on) {
  $("ask-btn").disabled = on;
  const box = $("loading");
  box.hidden = !on;
  clearInterval(timer);
  if (on) {
    const start = performance.now();
    const tick = () => {
      const s = ((performance.now() - start) / 1000).toFixed(0);
      box.textContent = `Searching the documents and writing an answer… ${s}s`;
    };
    tick();
    timer = setInterval(tick, 500);
  }
}

async function ask(question) {
  question = question.trim();
  if (!question) return;
  $("error").hidden = true;
  setLoading(true);
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : "The question could not be processed.");
    }
    render(data);
  } catch (err) {
    $("error").textContent = err.message || "Something went wrong.";
    $("error").hidden = false;
  } finally {
    setLoading(false);
  }
}

// --- Startup ---------------------------------------------------------------

async function loadStatus() {
  const status = $("status");
  try {
    const h = await (await fetch("/api/health")).json();
    status.textContent = h.llm_ok
      ? `${h.model} · ${h.chunks} chunks indexed`
      : h.llm_problem;
    status.classList.toggle("problem", !h.llm_ok);
  } catch {
    status.textContent = "Server not reachable";
    status.classList.add("problem");
  }
}

async function loadDocuments() {
  try {
    const docs = await (await fetch("/api/documents")).json();
    $("documents").replaceChildren(...docs.map((d) =>
      el("li", {}, el("span", { text: d.title }), el("span", { text: `${d.chunks} chunks` }))));
  } catch { /* the list is optional */ }
}

function init() {
  const examples = $("examples");
  for (const q of EXAMPLES) {
    const b = el("button", { type: "button", text: q });
    b.addEventListener("click", () => { $("question").value = q; ask(q); });
    examples.append(b);
  }
  $("ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask($("question").value); });
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("question").value); }
  });
  loadStatus();
  loadDocuments();
}

init();
