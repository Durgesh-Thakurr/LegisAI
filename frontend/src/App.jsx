import { useState } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = "http://127.0.0.1:8000/chat";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: "web-session-1", message: input }),
      });
      const data = await res.json();

      const botMessage = {
        role: "bot",
        text: data.answer,
        sources: data.sources || [],
        confidence: data.confidence,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Something went wrong. Please try again.", sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      <header className="bg-slate-900 text-white px-6 py-4">
        <h1 className="text-xl font-semibold">LegisAI</h1>
        <p className="text-sm text-slate-300">
          Ask about Indian cyber law -- answers grounded in official sources
        </p>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4 max-w-2xl w-full mx-auto">
        {messages.length === 0 && (
          <p className="text-slate-400 text-center mt-10">
            Try asking: "What should I do if I got scammed via OTP fraud?"
          </p>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`rounded-lg px-4 py-3 max-w-[85%] ${
              msg.role === "user"
                ? "bg-blue-600 text-white ml-auto"
                : "bg-white border border-slate-200 text-slate-800"
            }`}
          >
            {msg.role === "bot" ? (
              <div className="prose prose-sm max-w-none prose-p:my-2 prose-headings:my-2 prose-table:my-2">
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
            ) : (
              <p className="whitespace-pre-wrap">{msg.text}</p>
            )}
            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-3 pt-2 border-t border-slate-200 text-sm">
                <strong className="text-slate-500">Sources:</strong>
                <ul className="mt-1 space-y-1">
                  {msg.sources.map((s, j) => (
                    <li key={j}>
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-blue-600 hover:underline"
                      >
                        {s.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="rounded-lg px-4 py-3 bg-white border border-slate-200 text-slate-500 max-w-[85%]">
            Thinking...
          </div>
        )}
      </div>

      <div className="border-t border-slate-200 bg-white px-4 py-3">
        <div className="max-w-2xl mx-auto flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your question..."
            disabled={loading}
            className="flex-1 border border-slate-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="bg-blue-600 text-white px-5 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>

      <footer className="text-center text-xs text-slate-400 py-2">
        This is informational guidance, not a substitute for professional legal advice.
      </footer>
    </div>
  );
}

export default App; 