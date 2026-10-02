import { useEffect, useRef, useState } from "react";

import { ApiError, MAX_MESSAGE_LENGTH, getSessionId, sendChat } from "./api";
import Message from "./components/Message";

const COUNTER_THRESHOLD = MAX_MESSAGE_LENGTH - 100;

export default function App() {
  const [sessionId] = useState(getSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const appendMessage = (message) => setMessages((previous) => [...previous, message]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    appendMessage({ role: "user", text });
    setInput("");
    setLoading(true);

    try {
      const data = await sendChat(sessionId, text);
      appendMessage({
        role: "bot",
        text: data.answer,
        sources: data.sources ?? [],
        confidence: data.confidence,
      });
    } catch (error) {
      appendMessage({
        role: "bot",
        text: error instanceof ApiError ? error.message : "Something went wrong. Please try again.",
        isError: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      <header className="bg-slate-900 text-white px-6 py-4">
        <h1 className="text-xl font-semibold">LegisAI</h1>
        <p className="text-sm text-slate-300">
          Ask about Indian cyber law. Answers are grounded in official sources.
        </p>
      </header>

      <main className="flex-1 overflow-y-auto px-4 py-6 space-y-4 max-w-2xl w-full mx-auto" aria-live="polite">
        {messages.length === 0 && (
          <p className="text-slate-400 text-center mt-10">
            Try asking: &quot;What should I do if I got scammed via OTP fraud?&quot;
          </p>
        )}
        {messages.map((message, index) => (
          <Message key={index} message={message} />
        ))}
        {loading && (
          <div className="rounded-lg px-4 py-3 bg-white border border-slate-200 text-slate-500 max-w-[85%]">
            Thinking...
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <form onSubmit={handleSubmit} className="border-t border-slate-200 bg-white px-4 py-3">
        <div className="max-w-2xl mx-auto flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Type your question..."
            maxLength={MAX_MESSAGE_LENGTH}
            disabled={loading}
            aria-label="Your question"
            className="flex-1 border border-slate-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-blue-600 text-white px-5 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            Send
          </button>
        </div>
        {input.length > COUNTER_THRESHOLD && (
          <p className="max-w-2xl mx-auto text-xs text-slate-400 mt-1 text-right">
            {input.length}/{MAX_MESSAGE_LENGTH}
          </p>
        )}
      </form>

      <footer className="text-center text-xs text-slate-400 py-2">
        This is informational guidance, not a substitute for professional legal advice.
      </footer>
    </div>
  );
} 