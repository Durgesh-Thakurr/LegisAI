const API_BASE = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const REQUEST_TIMEOUT_MS = 60000;
const SESSION_KEY = "legisai-session";

export const MAX_MESSAGE_LENGTH = 1000;

const STATUS_MESSAGES = {
  422: "Your message could not be processed. Keep it under 1000 characters.",
  429: "Too many requests. Please wait a minute and try again.",
  503: "The service is busy right now. Please try again shortly.",
};

export class ApiError extends Error {}

function createId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `s-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function getSessionId() {
  try {
    let id = sessionStorage.getItem(SESSION_KEY);
    if (!id) {
      id = createId();
      sessionStorage.setItem(SESSION_KEY, id);
    }
    return id;
  } catch {
    return createId();
  }
}

export function safeUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : null;
  } catch {
    return null;
  }
}

export async function sendChat(sessionId, message) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new ApiError(STATUS_MESSAGES[response.status] ?? "Something went wrong. Please try again.");
    }
    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(
      error.name === "AbortError"
        ? "The request took too long. Please try again."
        : "Could not reach the server. Check your connection and try again.",
    );
  } finally {
    clearTimeout(timer);
  }
} 