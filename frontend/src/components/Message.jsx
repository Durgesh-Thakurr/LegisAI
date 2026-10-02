import ReactMarkdown from "react-markdown";

import { safeUrl } from "../api";

const markdownComponents = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer noopener">
      {children}
    </a>
  ),
};

function Sources({ sources }) {
  return (
    <div className="mt-3 pt-2 border-t border-slate-200 text-sm">
      <strong className="text-slate-500">Sources:</strong>
      <ul className="mt-1 space-y-1">
        {sources.map((source) => {
          const href = safeUrl(source.url);
          return (
            <li key={source.url}>
              {href ? (
                <a
                  href={href}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="text-blue-600 hover:underline"
                >
                  {source.title}
                </a>
              ) : (
                <span className="text-slate-600">{source.title}</span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function getBubbleClasses({ role, isError }) {
  if (role === "user") return "bg-blue-600 text-white ml-auto";
  if (isError) return "bg-red-50 border border-red-200 text-red-800";
  return "bg-white border border-slate-200 text-slate-800";
}

export default function Message({ message }) {
  const { role, text, isError, sources = [], confidence } = message;
  const isMarkdown = role === "bot" && !isError;

  return (
    <div className={`rounded-lg px-4 py-3 max-w-[85%] ${getBubbleClasses(message)}`}>
      {isMarkdown ? (
        <div className="prose prose-sm max-w-none prose-p:my-2 prose-headings:my-2 prose-table:my-2">
          <ReactMarkdown components={markdownComponents}>{text}</ReactMarkdown>
        </div>
      ) : (
        <p className="whitespace-pre-wrap break-words">{text}</p>
      )}

      {isMarkdown && confidence === "low" && sources.length > 0 && (
        <p className="mt-3 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1">
          Low confidence match. Please verify with the official sources below.
        </p>
      )}

      {sources.length > 0 && <Sources sources={sources} />}
    </div>
  );
} 