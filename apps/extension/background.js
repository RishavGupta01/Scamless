// Scamless service worker: context menu, offscreen document lifecycle,
// and message routing between content scripts / popup and the offscreen
// inference document. The service worker never runs the model itself -
// MV3 service workers cannot do heavy inference, that is the offscreen
// document's job (per spec section 8).

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

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "scamless-scan-selection" || !tab) return;
  const verdict = await routeScan(info.selectionText || "");
  chrome.tabs.sendMessage(tab.id, { type: "scan-result", text: info.selectionText || "", verdict });
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "scan-request") {
    (async () => {
      const verdict = await routeScan(msg.text || "");
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
