// Scamless content script: renders verdict toasts, adds per-message check
// buttons on supported webmail/chat sites. All inference happens via the
// background service worker routing to the offscreen document - this script
// only reads text and paints results.

const SITE_SELECTORS = {
  "mail.google.com": { root: 'div[role="main"]', message: "div.ii.gt", text: "div[dir=ltr]" },
  "outlook.live.com": { root: 'div[role="main"]', message: '[aria-label="Message body"], [data-testid="MessageBody"]', text: null },
  "outlook.office.com": { root: 'div[role="main"]', message: '[aria-label="Message body"], [data-testid="MessageBody"]', text: null },
  "web.whatsapp.com": { root: "#main", message: "div.message-in", text: ".copyable-text span" },
};

const host = location.hostname;
const site = SITE_SELECTORS[host];

function showToast(verdict) {
  const existing = document.getElementById("scamless-toast");
  if (existing) existing.remove();

  const host = document.createElement("div");
  host.id = "scamless-toast";
  const shadow = host.attachShadow({ mode: "open" });

  const style = document.createElement("style");
  style.textContent = `
    .card { position: fixed; top: 18px; right: 18px; z-index: 2147483647;
      width: 320px; background: #11161d; color: #e6edf3; border: 1px solid #232d3a;
      border-radius: 12px; padding: 14px 16px; font: 14px/1.5 system-ui, sans-serif;
      box-shadow: 0 8px 30px rgba(0,0,0,.5); }
    .brand { font-weight: 800; letter-spacing: .12em; }
    .brand span { color: #4ea1ff; }
    .band { font-weight: 800; font-size: 15px; margin-top: 6px; }
    .row { display: flex; justify-content: space-between; margin-top: 6px; }
    .prob { color: #4ea1ff; }
    .blurb { color: #8b98a5; font-size: 12px; margin-top: 2px; }
    .close { position: absolute; top: 8px; right: 12px; cursor: pointer; color: #8b98a5; }
    .safe { color: #3ddc84; } .susp { color: #ffb84d; } .danger { color: #ff5c5c; }
  `;
  shadow.appendChild(style);

  const card = document.createElement("div");
  card.className = "card";
  const close = document.createElement("span");
  close.className = "close";
  close.textContent = "x";
  close.addEventListener("click", () => host.remove());

  const brand = document.createElement("div");
  brand.className = "brand";
  brand.textContent = "SCAM";
  const brandSpan = document.createElement("span");
  brandSpan.textContent = "LESS";
  brand.appendChild(brandSpan);

  card.appendChild(close);
  card.appendChild(brand);

  if (!verdict || !verdict.ok) {
    const err = document.createElement("div");
    err.className = "band danger";
    err.textContent = "Scan failed: " + ((verdict && verdict.error) || "unknown");
    card.appendChild(err);
  } else {
    const top = verdict.hits[0];
    let bandText = "SAFE";
    let bandClass = "safe";
    let score = 0;
    if (top) {
      score = Math.min(100, Math.round(top.prob * 100));
      if (score >= 70) { bandText = "DANGEROUS"; bandClass = "danger"; }
      else if (score >= 40) { bandText = "SUSPICIOUS"; bandClass = "susp"; }
    }
    const band = document.createElement("div");
    band.className = "band " + bandClass;
    band.textContent = `${bandText} - risk ${score}/100`;
    card.appendChild(band);

    const rows = (verdict.hits.length ? verdict.hits : verdict.weak).slice(0, 3);
    for (const hit of rows) {
      const row = document.createElement("div");
      row.className = "row";
      const name = document.createElement("span");
      name.textContent = (hit.weak ? "possible " : "") + hit.label.replace(/_/g, " ");
      const prob = document.createElement("span");
      prob.className = "prob";
      prob.textContent = Math.round(hit.prob * 100) + "%";
      row.append(name, prob);
      card.appendChild(row);
    }
  }

  shadow.appendChild(card);
  document.documentElement.appendChild(host);
  setTimeout(() => host.remove(), 30000);
}

function makeCheckButton(onClick) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = "Check";
  btn.setAttribute("aria-label", "Check this message for scams with Scamless");
  btn.className = "scamless-check-btn";
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    btn.disabled = true;
    btn.textContent = "...";
    onClick().finally(() => {
      btn.disabled = false;
      btn.textContent = "Check";
    });
  });
  return btn;
}

async function checkText(text) {
  const resp = await chrome.runtime.sendMessage({ type: "scan-request", text });
  showToast(resp);
}

function decorate() {
  if (!site) return;
  try {
    const containers = document.querySelectorAll(site.message);
    for (const container of containers) {
      if (container.querySelector(".scamless-check-btn")) continue;
      const textEl = site.text ? container.querySelector(site.text) : container;
      if (!textEl) continue;
      const text = (textEl.innerText || "").trim();
      if (text.length < 20) continue;
      const btn = makeCheckButton(() => checkText(text));
      container.style.position = container.style.position || "relative";
      container.appendChild(btn);
    }
  } catch {
    /* site DOM changes constantly; never break the host page */
  }
}

let debounceTimer = null;
const observer = new MutationObserver(() => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(decorate, 800);
});
observer.observe(document.documentElement, { childList: true, subtree: true });
decorate();

chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === "scan-result") {
    showToast(msg.verdict);
  }
});
