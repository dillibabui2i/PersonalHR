const apiBaseUrl = "http://127.0.0.1:8000";
const extensionKey = "__EXTENSION_KEY__";

type SiteLookup = {
  registered: boolean;
  portalEnabled: boolean;
  memoryEnabled: boolean;
  profileApiPath: string;
};

type LookupRequest = {
  type: "lookupSite";
  host: string;
};

type ClientRequest = {
  type: "clientId";
};

type SnapshotRequest = {
  type: "saveSnapshot";
  siteHost: string;
  pageUrl: string;
  visibleText: string;
};

type IdentityRequest = {
  type: "saveIdentity";
  siteHost: string;
  employeeEmail?: string;
  profile: unknown;
};

type ProfileFetchRequest = {
  type: "fetchPortalProfile";
  url: string;
};

function isLookupRequest(value: unknown): value is LookupRequest {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as { type?: unknown; host?: unknown };
  return candidate.type === "lookupSite" && typeof candidate.host === "string";
}

function isSnapshotRequest(value: unknown): value is SnapshotRequest {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as { type?: unknown; siteHost?: unknown; pageUrl?: unknown; visibleText?: unknown };
  return (
    candidate.type === "saveSnapshot" &&
    typeof candidate.siteHost === "string" &&
    typeof candidate.pageUrl === "string" &&
    typeof candidate.visibleText === "string"
  );
}

function isIdentityRequest(value: unknown): value is IdentityRequest {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as { type?: unknown; siteHost?: unknown; profile?: unknown };
  return candidate.type === "saveIdentity" && typeof candidate.siteHost === "string" && "profile" in candidate;
}

function isProfileFetchRequest(value: unknown): value is ProfileFetchRequest {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as { type?: unknown; url?: unknown };
  return candidate.type === "fetchPortalProfile" && typeof candidate.url === "string";
}

async function fetchPortalProfileInPage(tabId: number, url: string): Promise<unknown | null> {
  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId },
      world: "MAIN",
      args: [url],
      func: async (profileUrl: string) => {
        const tokenHints = /token|auth|jwt|bearer|access|session|id_token/i;
        const tokens: string[] = [];

        function addToken(value: string): void {
          const cleaned = value.trim().replace(/^Bearer\s+/i, "");
          if (cleaned.length < 16 || tokens.includes(cleaned)) {
            return;
          }
          tokens.push(cleaned);
        }

        function collectFromValue(value: string, key = ""): void {
          if (value.length < 16) {
            return;
          }
          if (tokenHints.test(key) || value.split(".").length === 3) {
            addToken(value);
          }
          if (!(value.startsWith("{") || value.startsWith("["))) {
            return;
          }
          try {
            const parsed: unknown = JSON.parse(value);
            collectFromUnknown(parsed);
          } catch {
            return;
          }
        }

        function collectFromUnknown(value: unknown, depth = 0): void {
          if (depth > 4 || value === null || value === undefined) {
            return;
          }
          if (typeof value === "string") {
            collectFromValue(value);
            return;
          }
          if (Array.isArray(value)) {
            for (const item of value.slice(0, 30)) {
              collectFromUnknown(item, depth + 1);
            }
            return;
          }
          if (typeof value !== "object") {
            return;
          }
          for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
            if (typeof nested === "string") {
              collectFromValue(nested, key);
            } else {
              collectFromUnknown(nested, depth + 1);
            }
          }
        }

        for (const store of [window.localStorage, window.sessionStorage]) {
          for (let index = 0; index < store.length; index += 1) {
            const key = store.key(index) ?? "";
            collectFromValue(store.getItem(key) ?? "", key);
          }
        }

        async function attempt(headers: Record<string, string>): Promise<unknown | null> {
          const controller = new AbortController();
          const timer = window.setTimeout(() => controller.abort(), 4000);
          try {
            const response = await fetch(profileUrl, {
              credentials: "include",
              signal: controller.signal,
              headers: {
                Accept: "application/json",
                ...headers,
              },
            });
            if (!response.ok) {
              return null;
            }
            return (await response.json()) as unknown;
          } catch {
            return null;
          } finally {
            window.clearTimeout(timer);
          }
        }

        const attempts = [
          attempt({}),
          ...tokens.slice(0, 3).map((token) => attempt({ Authorization: `Bearer ${token}` })),
        ];
        const results = await Promise.all(attempts);
        return results.find((result) => result !== null) ?? null;
      },
    });
    return injection?.result ?? null;
  } catch {
    return null;
  }
}

function isProfileStatusRequest(value: unknown): value is { type: "profileStatus"; siteHost: string } {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as { type?: unknown; siteHost?: unknown };
  return candidate.type === "profileStatus" && typeof candidate.siteHost === "string";
}

function isClientRequest(value: unknown): value is ClientRequest {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  return (value as { type?: unknown }).type === "clientId";
}

async function savedClientId(): Promise<string> {
  const stored = await chrome.storage.local.get("clientId");
  if (typeof stored.clientId === "string" && stored.clientId !== "") {
    return stored.clientId;
  }
  const created = crypto.randomUUID();
  await chrome.storage.local.set({ clientId: created });
  return created;
}

async function saveSnapshot(_message: SnapshotRequest): Promise<void> {
  return;
}

async function saveIdentity(message: IdentityRequest): Promise<{ saved: boolean; email: string }> {
  try {
    const site = await lookupSite(message.siteHost);
    if (!site.registered || !site.memoryEnabled) {
      return { saved: false, email: "" };
    }
    const clientId = await savedClientId();
    const response = await fetch(`${apiBaseUrl}/extension/identity`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-extension-key": extensionKey,
      },
      body: JSON.stringify({
        clientId,
        siteHost: message.siteHost,
        employeeEmail: message.employeeEmail ?? "",
        profile: message.profile,
      }),
    });
    if (!response.ok) {
      return { saved: false, email: "" };
    }
    const payload = (await response.json()) as { saved?: unknown; email?: unknown };
    return {
      saved: payload.saved === true,
      email: typeof payload.email === "string" ? payload.email : "",
    };
  } catch {
    return { saved: false, email: "" };
  }
}

async function profileStatus(siteHost: string): Promise<{ loaded: boolean }> {
  try {
    const clientId = await savedClientId();
    const query = new URLSearchParams({ clientId, siteHost });
    const response = await fetch(`${apiBaseUrl}/extension/identity?${query}`, {
      headers: { "x-extension-key": extensionKey },
    });
    if (!response.ok) {
      return { loaded: false };
    }
    const payload = (await response.json()) as { loaded?: unknown };
    return { loaded: payload.loaded === true };
  } catch {
    return { loaded: false };
  }
}

async function lookupSite(host: string): Promise<SiteLookup> {
  try {
    const response = await fetch(
      `${apiBaseUrl}/extension/sites?host=${encodeURIComponent(host)}`,
      { headers: { "x-extension-key": extensionKey } },
    );
    if (!response.ok) {
      return {
        registered: false,
        portalEnabled: false,
        memoryEnabled: false,
        profileApiPath: "",
      };
    }
    const payload = (await response.json()) as {
      registered?: unknown;
      portalEnabled?: unknown;
      memoryEnabled?: unknown;
      profileApiPath?: unknown;
    };
    return {
      registered: payload.registered === true,
      portalEnabled: payload.portalEnabled !== false,
      memoryEnabled: payload.memoryEnabled !== false,
      profileApiPath: typeof payload.profileApiPath === "string" ? payload.profileApiPath : "",
    };
  } catch {
    return {
      registered: false,
      portalEnabled: false,
      memoryEnabled: false,
      profileApiPath: "",
    };
  }
}

function notifyTab(tabId: number): void {
  chrome.tabs.sendMessage(tabId, { type: "navigation" }).catch(() => undefined);
}

chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id) {
    return false;
  }
  if (isLookupRequest(message)) {
    void lookupSite(message.host).then(sendResponse);
    return true;
  }
  if (isClientRequest(message)) {
    void savedClientId().then((clientId) => sendResponse({ clientId }));
    return true;
  }
  if (isSnapshotRequest(message)) {
    void saveSnapshot(message).then(() => sendResponse({ stored: true }));
    return true;
  }
  if (isIdentityRequest(message)) {
    void saveIdentity(message).then(sendResponse);
    return true;
  }
  if (isProfileStatusRequest(message)) {
    void profileStatus(message.siteHost).then(sendResponse);
    return true;
  }
  if (isProfileFetchRequest(message)) {
    const tabId = sender.tab?.id;
    if (tabId === undefined) {
      sendResponse({ payload: null });
      return false;
    }
    void fetchPortalProfileInPage(tabId, message.url).then((payload) => sendResponse({ payload }));
    return true;
  }
  return false;
});

chrome.webNavigation.onHistoryStateUpdated.addListener((details) => {
  if (details.frameId === 0) {
    notifyTab(details.tabId);
  }
});

chrome.webNavigation.onReferenceFragmentUpdated.addListener((details) => {
  if (details.frameId === 0) {
    notifyTab(details.tabId);
  }
});
