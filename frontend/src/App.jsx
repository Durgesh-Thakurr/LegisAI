import { useEffect, useRef, useState } from "react";

import { ApiError, MAX_MESSAGE_LENGTH, getSessionId, sendChat } from "./api";
import Message from "./components/Message";

const COUNTER_THRESHOLD = MAX_MESSAGE_LENGTH - 100;
const STARTER_QUESTIONS = [
  "I got scammed through OTP fraud. What should I do?",
  "What is a digital arrest scam?",
  "Section 66C kya hai?",
  "BNS mein cheating ki saza kya hai?",
  "How do I report a cybercrime online?",
];

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

  const sendMessage = async (rawText) => {
    const text = rawText.trim();
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

  const handleSubmit = (event) => {
    event.preventDefault();
    sendMessage(input);
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
          <div className="mt-10 text-center">
            <p className="text-slate-400">Try asking:</p>
            <div className="mt-3 flex flex-wrap justify-center gap-2">
              {STARTER_QUESTIONS.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => sendMessage(question)}
                  disabled={loading}
                  className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
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