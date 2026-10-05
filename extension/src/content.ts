const askButtonId = "personal-hr-ask";
const panelId = "personal-hr-panel";
const panelOpenKey = "personal-hr-panel-open";
const panelWideKey = "personal-hr-panel-wide";
const profileLoadedPrefix = "personal-hr-profile-loaded:";
const narrowPanelWidth = 640;
const authSyncDelayMs = 150;

const loginPathPattern = /\/(login|signin|sign-in|auth|forgot-password|reset-password)(\/|$)/i;
const loginTextPattern = /forgot password\?|sign in with google|enter the password|enter ideas2it email/i;

let requestSerial = 0;
let siteRegistered = false;
let memoryEnabled = true;
let profileApiPath = "";
let authSyncTimer = 0;
let authPollTimer = 0;
let profileSyncStarted = false;
let profilePhase: "loading" | "ready" = "loading";
let browserClientId = "";
let framedEmail = "";
let resolvedEmail = "";

const emailPattern = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i;
const emailParamNames = ["email", "useremail", "employeeemail", "mail", "user"];

function searchPairs(url: URL): Array<[string, string]> {
  const query = url.search.startsWith("?") ? url.search.slice(1) : url.search;
  if (query === "") {
    return [];
  }
  return query.split("&").map((part) => {
    const splitAt = part.indexOf("=");
    if (splitAt < 0) {
      return [decodeURIComponent(part), ""];
    }
    return [decodeURIComponent(part.slice(0, splitAt)), decodeURIComponent(part.slice(splitAt + 1))];
  });
}

function pageEmployeeEmail(): string {
  const url = new URL(window.location.href);
  const pairs = searchPairs(url);
  for (const name of emailParamNames) {
    for (const [key, value] of pairs) {
      if (key.toLowerCase() !== name) {
        continue;
      }
      const match = value.match(emailPattern);
      if (match !== null) {
        return match[0].toLowerCase();
      }
    }
  }
  for (const [, value] of pairs) {
    const match = value.match(emailPattern);
    if (match !== null) {
      return match[0].toLowerCase();
    }
  }
  const fromHash = decodeURIComponent(url.hash).match(emailPattern);
  if (fromHash !== null) {
    return fromHash[0].toLowerCase();
  }
  const fromPath = decodeURIComponent(url.pathname).match(emailPattern);
  if (fromPath !== null) {
    return fromPath[0].toLowerCase();
  }
  return "";
}

function emailStorageKey(): string {
  return `personal-hr-employee-email:${window.location.hostname}`;
}

function activeEmployeeEmail(): string {
  const fromPage = pageEmployeeEmail();
  return fromPage !== "" ? fromPage : resolvedEmail;
}

function chatUrl(clientId: string): string {
  const query = new URLSearchParams({
    siteHost: window.location.hostname,
    clientId,
    embed: "1",
  });
  const email = activeEmployeeEmail();
  if (email !== "") {
    query.set("employeeEmail", email);
  }
  return `http://127.0.0.1:3000/chat?${query}`;
}

type HostRegistration = {
  registered?: boolean;
  memoryEnabled?: boolean;
  profileApiPath?: string;
};

function profileRequestUrl(path: string): string | null {
  const cleaned = path.trim();
  if (cleaned === "") {
    return null;
  }
  if (cleaned.startsWith("/")) {
    return new URL(cleaned, window.location.origin).toString();
  }
  try {
    const url = new URL(cleaned);
    if (url.hostname !== window.location.hostname) {
      return null;
    }
    return url.toString();
  } catch {
    return null;
  }
}

function profileLoadedKey(): string {
  return `${profileLoadedPrefix}${window.location.hostname}:${activeEmployeeEmail()}`;
}

async function setProfileLoaded(loaded: boolean): Promise<void> {
  await chrome.storage.local.set({ [profileLoadedKey()]: loaded });
}

async function rememberEmployeeEmail(email: string): Promise<void> {
  resolvedEmail = email;
  await chrome.storage.local.set({ [emailStorageKey()]: email });
  syncChatEmployee();
}

async function syncEmployeeProfile(): Promise<void> {
  if (profileSyncStarted || !siteRegistered || !memoryEnabled || isLoginScreen()) {
    return;
  }
  const requestUrl = profileRequestUrl(profileApiPath);
  if (requestUrl === null) {
    profilePhase = "ready";
    announceProfilePhase();
    return;
  }
  profileSyncStarted = true;
  try {
    profilePhase = "loading";
    announceProfilePhase();
    const fetched = (await chrome.runtime.sendMessage({
      type: "fetchPortalProfile",
      url: requestUrl,
    })) as { payload?: unknown } | undefined;
    const payload = fetched?.payload;
    if (payload === null || payload === undefined) {
      profilePhase = "ready";
      announceProfilePhase();
      return;
    }
    const result = (await chrome.runtime.sendMessage({
      type: "saveIdentity",
      siteHost: window.location.hostname,
      employeeEmail: activeEmployeeEmail(),
      profile: payload,
    })) as { saved?: boolean; email?: string } | undefined;
    const email = (result?.email ?? "").trim().toLowerCase();
    profilePhase = "ready";
    if (email !== "") {
      await rememberEmployeeEmail(email);
      if (result?.saved === true) {
        await setProfileLoaded(true);
        notifyChat("memory-updated");
      }
      announceProfilePhase();
      return;
    }
    announceProfilePhase();
  } catch {
    profilePhase = "ready";
    announceProfilePhase();
  }
}

function notifyChat(type: "memory-updated" | "profile-loading" | "profile-ready"): void {
  const frame = document.getElementById(panelId)?.shadowRoot?.querySelector("iframe");
  if (frame instanceof HTMLIFrameElement) {
    postToChat(frame, type);
  }
}

function announceProfilePhase(): void {
  notifyChat(profilePhase === "loading" ? "profile-loading" : "profile-ready");
}

function isVisible(element: Element): boolean {
  if (!(element instanceof HTMLElement) || element.getClientRects().length === 0) {
    return false;
  }
  const style = window.getComputedStyle(element);
  return style.display !== "none" && style.visibility !== "hidden";
}

function isLoginScreen(): boolean {
  if (loginPathPattern.test(window.location.pathname) || loginPathPattern.test(window.location.hash)) {
    return true;
  }
  if (Array.from(document.querySelectorAll('input[type="password"]')).some(isVisible)) {
    return true;
  }
  return loginTextPattern.test(document.body?.innerText ?? "");
}

function isPortalAuthenticated(): boolean {
  const body = document.body;
  if (body === null || body.innerText.trim() === "") {
    return false;
  }
  return !isLoginScreen();
}

function removeAskButton(): void {
  document.getElementById(askButtonId)?.remove();
  document.getElementById(panelId)?.remove();
  sessionStorage.removeItem(panelOpenKey);
}

function placeFixed(element: HTMLElement, bottom: string): void {
  element.style.setProperty("position", "fixed", "important");
  element.style.setProperty("right", "24px", "important");
  element.style.setProperty("bottom", bottom, "important");
  element.style.setProperty("z-index", "2147483647", "important");
}

function panelIsNarrow(): boolean {
  return window.innerWidth < narrowPanelWidth;
}

function applyPanelSize(panel: HTMLElement): void {
  const wide = window.localStorage.getItem(panelWideKey) === "1";
  panel.style.setProperty("position", "fixed", "important");
  panel.style.setProperty("z-index", "2147483647", "important");
  if (panelIsNarrow()) {
    panel.style.setProperty("left", "0", "important");
    panel.style.setProperty("right", "0", "important");
    panel.style.setProperty("bottom", "0", "important");
    panel.style.setProperty("width", "100vw", "important");
    panel.style.setProperty("height", "min(640px, 100vh)", "important");
    return;
  }
  panel.style.setProperty("left", "auto", "important");
  panel.style.setProperty("right", "24px", "important");
  panel.style.setProperty("bottom", "88px", "important");
  panel.style.setProperty("width", wide ? "min(720px, calc(100vw - 48px))" : "min(400px, calc(100vw - 48px))", "important");
  panel.style.setProperty("height", "min(640px, calc(100vh - 120px))", "important");
}

function iconSvg(name: "expand" | "collapse" | "close" | "memory" | "refresh"): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.8");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const paths: Record<typeof name, string[]> = {
    expand: ["M15 3h6v6", "M9 21H3v-6", "M21 3l-7 7", "M3 21l7-7"],
    collapse: ["M9 3H3v6", "M15 21h6v-6", "M3 3l7 7", "M21 21l-7-7"],
    close: ["M6 6l12 12", "M18 6L6 18"],
    memory: ["M9 3v2", "M15 3v2", "M9 19v2", "M15 19v2", "M3 9h2", "M3 15h2", "M19 9h2", "M19 15h2", "M9 7h6a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2Z"],
    refresh: ["M21 12a9 9 0 1 1-2.64-6.36", "M21 3v6h-6"],
  };
  for (const d of paths[name]) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}

function postToChat(
  chatFrame: HTMLIFrameElement,
  type: "open-memory" | "reset-conversation" | "clear-employee" | "memory-updated" | "profile-loading" | "profile-ready",
): void {
  chatFrame.contentWindow?.postMessage({ source: "personal-hr-extension", type }, "*");
}

function closePanel(): void {
  sessionStorage.removeItem(panelOpenKey);
  document.getElementById(panelId)?.remove();
}

async function openPanel(): Promise<void> {
  if (!siteRegistered || !isPortalAuthenticated()) {
    removeAskButton();
    return;
  }
  sessionStorage.setItem(panelOpenKey, "1");
  const existingPanel = document.getElementById(panelId);
  if (existingPanel !== null) {
    existingPanel.hidden = false;
    applyPanelSize(existingPanel);
    return;
  }
  try {
    const response = (await chrome.runtime.sendMessage({ type: "clientId" })) as { clientId?: string } | undefined;
    browserClientId = response?.clientId ?? "";
  } catch {
    return;
  }
  if (browserClientId === "") {
    return;
  }
  const panel = document.createElement("section");
  panel.id = panelId;
  applyPanelSize(panel);
  const shadow = panel.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = `
    :host { display: flex; }
    :host([hidden]) { display: none !important; }
    section {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 100%;
      overflow: hidden;
      border-radius: 16px;
      background: #ffffff;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.28);
      font: 600 14px/1.2 "Segoe UI", sans-serif;
      color: #0f172a;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid #e2e8f0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .brand img {
      width: 22px;
      height: 22px;
      border-radius: 5px;
      object-fit: contain;
    }
    .actions { display: flex; align-items: center; gap: 6px; }
    button {
      display: grid;
      place-items: center;
      width: 32px;
      height: 32px;
      border: 0;
      border-radius: 10px;
      background: #e0e7ff;
      color: #1e1b4b;
      padding: 0;
      cursor: pointer;
    }
    button:hover { background: #c7d2fe; }
    button.record {
      width: auto;
      padding: 0 10px;
      font: 600 12px/1 "Segoe UI", sans-serif;
      white-space: nowrap;
    }
    button:focus-visible {
      outline: 3px solid #312e81;
      outline-offset: 2px;
    }
    button svg {
      width: 16px;
      height: 16px;
      display: block;
    }
    iframe {
      flex: 1;
      width: 100%;
      border: 0;
      background: #ffffff;
    }
  `;
  const frame = document.createElement("section");
  const header = document.createElement("header");
  const brand = document.createElement("span");
  brand.className = "brand";
  const mark = document.createElement("img");
  mark.src = chrome.runtime.getURL("icons/ideas2it-mark.png");
  mark.alt = "";
  const title = document.createElement("span");
  title.textContent = "Personal HR";
  brand.append(mark, title);
  const actions = document.createElement("div");
  actions.className = "actions";
  const memoryButton = document.createElement("button");
  memoryButton.type = "button";
  memoryButton.setAttribute("aria-label", "What I remember");
  memoryButton.title = "What I remember";
  memoryButton.append(iconSvg("memory"));
  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.setAttribute("aria-label", "Reset conversation");
  resetButton.title = "Reset conversation";
  resetButton.append(iconSvg("refresh"));
  const widthButton = document.createElement("button");
  widthButton.type = "button";
  const recordButton = document.createElement("button");
  recordButton.type = "button";
  recordButton.className = "record";
  recordButton.textContent = "Clear record";
  recordButton.setAttribute("aria-label", "Clear employee record");
  recordButton.title = "Clear employee record";
  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.setAttribute("aria-label", "Close");
  closeButton.title = "Close";
  closeButton.append(iconSvg("close"));
  closeButton.addEventListener("click", closePanel);
  function syncWidthButton(): void {
    const wide = window.localStorage.getItem(panelWideKey) === "1";
    widthButton.replaceChildren(iconSvg(wide ? "collapse" : "expand"));
    widthButton.setAttribute("aria-label", wide ? "Use standard width" : "Expand panel");
    widthButton.title = wide ? "Standard width" : "Expand";
    widthButton.setAttribute("aria-pressed", wide ? "true" : "false");
  }
  widthButton.addEventListener("click", () => {
    const wide = window.localStorage.getItem(panelWideKey) === "1";
    window.localStorage.setItem(panelWideKey, wide ? "0" : "1");
    applyPanelSize(panel);
    syncWidthButton();
  });
  syncWidthButton();
  actions.append(memoryButton, resetButton, widthButton, recordButton, closeButton);
  header.append(brand, actions);
  frame.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      closePanel();
    }
  });
  const chatFrame = document.createElement("iframe");
  framedEmail = activeEmployeeEmail();
  chatFrame.src = chatUrl(browserClientId);
  chatFrame.title = "Ask Your HR";
  chatFrame.addEventListener("load", () => announceProfilePhase());
  recordButton.addEventListener("click", () => postToChat(chatFrame, "clear-employee"));
  memoryButton.addEventListener("click", () => postToChat(chatFrame, "open-memory"));
  resetButton.addEventListener("click", () => postToChat(chatFrame, "reset-conversation"));
  frame.append(header, chatFrame);
  shadow.append(style, frame);
  document.documentElement.append(panel);
}

function showAskButton(): void {
  if (document.getElementById(askButtonId) !== null) {
    return;
  }
  const host = document.createElement("div");
  host.id = askButtonId;
  placeFixed(host, "24px");
  const shadow = host.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = `
    button {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 0;
      border-radius: 999px;
      background: #4f46e5;
      color: #ffffff;
      font: 600 14px/1.2 "Segoe UI", sans-serif;
      padding: 12px 16px;
      cursor: pointer;
      box-shadow: 0 10px 24px rgba(79, 70, 229, 0.35);
    }
    button img {
      width: 20px;
      height: 20px;
      border-radius: 5px;
      object-fit: contain;
      background: #ffffff;
    }
    button:focus-visible {
      outline: 3px solid #1e1b4b;
      outline-offset: 3px;
    }
  `;
  const button = document.createElement("button");
  button.type = "button";
  const buttonMark = document.createElement("img");
  buttonMark.src = chrome.runtime.getURL("icons/ideas2it-mark.png");
  buttonMark.alt = "";
  const buttonLabel = document.createElement("span");
  buttonLabel.textContent = "Ask Your HR";
  button.append(buttonMark, buttonLabel);
  button.setAttribute("aria-label", "Ask Your HR");
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    void openPanel();
  });
  shadow.append(style, button);
  document.documentElement.append(host);
}

function syncChatEmployee(): void {
  const frame = document.getElementById(panelId)?.shadowRoot?.querySelector("iframe");
  if (!(frame instanceof HTMLIFrameElement) || browserClientId === "") {
    return;
  }
  const email = activeEmployeeEmail();
  if (email === framedEmail) {
    return;
  }
  framedEmail = email;
  frame.src = chatUrl(browserClientId);
}

function watchForSignIn(): void {
  if (!siteRegistered || authPollTimer !== 0) {
    return;
  }
  authPollTimer = window.setInterval(syncWidgetVisibility, 400);
}

function stopWatchingForSignIn(): void {
  if (authPollTimer === 0) {
    return;
  }
  window.clearInterval(authPollTimer);
  authPollTimer = 0;
}

function syncWidgetVisibility(): void {
  if (!siteRegistered || !isPortalAuthenticated()) {
    removeAskButton();
    if (siteRegistered) {
      watchForSignIn();
    } else {
      stopWatchingForSignIn();
    }
    return;
  }

  stopWatchingForSignIn();
  const wasVisible = document.getElementById(askButtonId) !== null;
  showAskButton();
  void syncEmployeeProfile();
  syncChatEmployee();
  if (!wasVisible) {
    if (sessionStorage.getItem(panelOpenKey) === "1") {
      void openPanel();
    }
  }
}

function scheduleAuthSync(): void {
  if (authSyncTimer !== 0) {
    return;
  }
  authSyncTimer = window.setTimeout(() => {
    authSyncTimer = 0;
    syncWidgetVisibility();
  }, authSyncDelayMs);
}

async function refreshRegistration(): Promise<void> {
  const serial = requestSerial + 1;
  requestSerial = serial;
  try {
    const response = (await chrome.runtime.sendMessage({
      type: "lookupSite",
      host: window.location.hostname,
    })) as HostRegistration | undefined;
    if (serial !== requestSerial) {
      return;
    }
    siteRegistered = response?.registered === true;
    memoryEnabled = response?.memoryEnabled !== false;
    profileApiPath = typeof response?.profileApiPath === "string" ? response.profileApiPath : "";
    void syncEmployeeProfile();
    syncWidgetVisibility();
  } catch {
    if (serial === requestSerial) {
      siteRegistered = false;
      syncWidgetVisibility();
    }
  }
}

chrome.runtime.onMessage.addListener((message: unknown) => {
  if (typeof message === "object" && message !== null && (message as { type?: unknown }).type === "navigation") {
    profileSyncStarted = false;
    void refreshRegistration();
  }
});

void chrome.storage.local.get(emailStorageKey()).then((stored) => {
  const saved = stored[emailStorageKey()];
  if (typeof saved === "string" && saved.trim() !== "") {
    resolvedEmail = saved.trim().toLowerCase();
    syncChatEmployee();
  }
});

void refreshRegistration();

const historyPushState = history.pushState.bind(history);
const historyReplaceState = history.replaceState.bind(history);
history.pushState = (...args) => {
  const result = historyPushState(...args);
  scheduleAuthSync();
  return result;
};
history.replaceState = (...args) => {
  const result = historyReplaceState(...args);
  scheduleAuthSync();
  return result;
};
window.addEventListener("popstate", scheduleAuthSync);
window.addEventListener("hashchange", scheduleAuthSync);

window.addEventListener("resize", () => {
  const panel = document.getElementById(panelId);
  if (panel !== null) {
    applyPanelSize(panel);
  }
});

window.addEventListener("message", (event) => {
  const data = event.data as { source?: unknown; type?: unknown } | null;
  if (data?.source === "personal-hr-chat" && data.type === "memory-cleared") {
    void setProfileLoaded(false);
    return;
  }
  if (data?.source !== "personal-hr" || data.type !== "close-panel") {
    return;
  }
  const frame = document.getElementById(panelId)?.shadowRoot?.querySelector("iframe");
  if (frame instanceof HTMLIFrameElement && event.source === frame.contentWindow) {
    closePanel();
  }
});

const pageObserver = new MutationObserver(() => {
  scheduleAuthSync();
});
pageObserver.observe(document.documentElement, {
  childList: true,
  subtree: true,
  characterData: true,
});
