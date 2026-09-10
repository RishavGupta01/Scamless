// Scamless service worker: context menu + keyboard command, offscreen
// document lifecycle, message routing, and per-tab risk badges. The service
// worker never runs the model itself - MV3 service workers cannot do heavy
// inference, that is the offscreen document's job (per spec section 8).

const OFFSCREEN_URL = "offscreen/offscreen.html";

async function hasOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  return contexts.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["WORKERS"],
    justification: "Run the scam detection model fully on-device",
  });
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "scamless-scan-selection",
    title: "Check selected text for scams",
    contexts: ["selection"],
  });
});

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "scan-selection") return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  try {
    const [result] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString(),
    });
    const text = (result && result.result) || "";
    if (!text.trim()) return;
    const verdict = await routeScan(text);
    chrome.tabs.sendMessage(tab.id, { type: "scan-result", text, verdict }).catch(() => {
      /* page without content script: the badge still reflects the verdict */
    });
  } catch {
    /* browser-internal pages cannot be scripted - ignore silently */
  }
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "scamless-scan-selection" || !tab) return;
  const verdict = await routeScan(info.selectionText || "");
  chrome.tabs.sendMessage(tab.id, { type: "scan-result", text: info.selectionText || "", verdict });
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "scan-request") {
    (async () => {
      const verdict = await routeScan(msg.text || "");
      if (sender.tab && verdict.ok) setBadge(sender.tab.id, verdict);
      sendResponse(verdict);
    })();
    return true; // async sendResponse
  }
});

async function routeScan(text) {
  if (!text || !text.trim()) {
    return { ok: false, error: "empty text" };
  }
  try {
    await ensureOffscreen();
    const resp = await chrome.runtime.sendMessage({
      target: "offscreen",
      type: "scan",
      text,
    });
    return resp || { ok: false, error: "no response from offscreen document" };
  } catch (err) {
    return { ok: false, error: String((err && err.message) || err) };
  }
}

function setBadge(tabId, verdict) {
  const score = typeof verdict.score === "number"
    ? verdict.score
    : verdict.hits[0] ? Math.round(verdict.hits[0].prob * 100) : 0;
  const text = score >= 40 ? String(score) : "";
  const color = score >= 70 ? "#ff5c5c" : score >= 40 ? "#ffb84d" : "#3ddc84";
  chrome.action.setBadgeText({ tabId, text });
  chrome.action.setBadgeBackgroundColor({ tabId, color });
}
