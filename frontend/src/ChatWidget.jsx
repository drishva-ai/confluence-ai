import { useState, useRef, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function ChatWidget() {
  const [messages, setMessages]   = useState([
    {
      role: "assistant",
      content: "Hi! I can answer questions about your Confluence documentation. What would you like to know?",
      sources: [],
    }
  ]);
  const [input, setInput]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [isOpen, setIsOpen]       = useState(false);
  const [streaming, setStreaming] = useState("");
  const messagesEndRef            = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  const sendMessage = async () => {
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setLoading(true);
    setStreaming("");

    // Add user message
    setMessages(prev => [...prev, { role: "user", content: question }]);

    try {
      // Use streaming endpoint for typing effect
      const response = await fetch(
        `${API_URL}/api/chat/stream?question=${encodeURIComponent(question)}`,
        { method: "GET" }
      );

      const reader  = response.body.getReader();
      const decoder = new TextDecoder();
      let   fullText = "";
      let   sources  = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") break;
            if (data.startsWith("Error:")) {
              fullText = data;
              break;
            }
            fullText += data;
            setStreaming(fullText);
          }
        }
      }

      // Fetch sources separately (from non-streaming endpoint)
      try {
        const chatRes = await fetch(`${API_URL}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, top_k: 3 }),
        });
        const chatData = await chatRes.json();
        sources = chatData.sources || [];
      } catch {}

      setStreaming("");
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: fullText, sources }
      ]);

    } catch (err) {
      setStreaming("");
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, I encountered an error. Please check if the backend is running.",
          sources: [],
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Chat bubble button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: "fixed", bottom: "24px", right: "24px",
          width: "56px", height: "56px", borderRadius: "50%",
          background: "#2563EB", border: "none", cursor: "pointer",
          display: "flex", alignItems: "center", justifyContent: "center",
          boxShadow: "0 4px 12px rgba(37,99,235,0.4)",
          zIndex: 9999, transition: "transform 0.2s",
        }}
        onMouseEnter={e => e.target.style.transform = "scale(1.1)"}
        onMouseLeave={e => e.target.style.transform = "scale(1)"}
      >
        {isOpen ? (
          <svg width="20" height="20" viewBox="0 0 20 20" fill="white">
            <path d="M4 4l12 12M16 4L4 16" stroke="white" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"
              stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
      </button>

      {/* Chat window */}
      {isOpen && (
        <div style={{
          position: "fixed", bottom: "92px", right: "24px",
          width: "380px", height: "560px",
          background: "#fff", borderRadius: "16px",
          boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
          display: "flex", flexDirection: "column",
          zIndex: 9999, overflow: "hidden",
          border: "1px solid #E2E8F0",
        }}>

          {/* Header */}
          <div style={{
            background: "#2563EB", padding: "16px 20px",
            display: "flex", alignItems: "center", gap: "12px",
          }}>
            <div style={{
              width: "36px", height: "36px", borderRadius: "50%",
              background: "rgba(255,255,255,0.2)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
                  stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <div>
              <p style={{ margin: 0, color: "white", fontWeight: 600, fontSize: "15px" }}>
                Docs Assistant
              </p>
              <p style={{ margin: 0, color: "rgba(255,255,255,0.8)", fontSize: "12px" }}>
                Powered by your Confluence
              </p>
            </div>
          </div>

          {/* Messages */}
          <div style={{
            flex: 1, overflowY: "auto", padding: "16px",
            display: "flex", flexDirection: "column", gap: "12px",
          }}>
            {messages.map((msg, i) => (
              <div key={i} style={{
                display: "flex",
                justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
              }}>
                <div style={{
                  maxWidth: "85%",
                  background: msg.role === "user" ? "#2563EB" : "#F1F5F9",
                  color: msg.role === "user" ? "white" : "#0F172A",
                  borderRadius: msg.role === "user"
                    ? "16px 16px 4px 16px"
                    : "16px 16px 16px 4px",
                  padding: "10px 14px",
                  fontSize: "14px",
                  lineHeight: "1.6",
                }}>
                  <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{msg.content}</p>

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div style={{
                      marginTop: "10px", paddingTop: "8px",
                      borderTop: "1px solid rgba(0,0,0,0.1)",
                    }}>
                      <p style={{ margin: "0 0 4px", fontSize: "11px",
                        color: "#64748B", fontWeight: 600 }}>
                        Sources:
                      </p>
                      {msg.sources.map((src, si) => (
                        <a key={si} href={src.url} target="_blank" rel="noreferrer"
                          style={{
                            display: "block", fontSize: "11px",
                            color: "#2563EB", textDecoration: "none",
                            marginBottom: "2px",
                            overflow: "hidden", textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}>
                          📄 {src.title}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Streaming message */}
            {streaming && (
              <div style={{ display: "flex", justifyContent: "flex-start" }}>
                <div style={{
                  maxWidth: "85%", background: "#F1F5F9", color: "#0F172A",
                  borderRadius: "16px 16px 16px 4px", padding: "10px 14px",
                  fontSize: "14px", lineHeight: "1.6",
                }}>
                  <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{streaming}</p>
                  <span style={{
                    display: "inline-block", width: "6px", height: "14px",
                    background: "#2563EB", marginLeft: "2px",
                    animation: "blink 1s infinite",
                  }}/>
                </div>
              </div>
            )}

            {/* Loading dots */}
            {loading && !streaming && (
              <div style={{ display: "flex", justifyContent: "flex-start" }}>
                <div style={{
                  background: "#F1F5F9", borderRadius: "16px 16px 16px 4px",
                  padding: "12px 16px", display: "flex", gap: "4px",
                }}>
                  {[0, 1, 2].map(i => (
                    <div key={i} style={{
                      width: "6px", height: "6px", borderRadius: "50%",
                      background: "#94A3B8",
                      animation: `bounce 1.2s infinite ${i * 0.2}s`,
                    }}/>
                  ))}
                </div>
              </div>
            )}
            <div ref={messagesEndRef}/>
          </div>

          {/* Input */}
          <div style={{
            padding: "12px 16px", borderTop: "1px solid #E2E8F0",
            display: "flex", gap: "8px", alignItems: "flex-end",
          }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your docs..."
              rows={1}
              style={{
                flex: 1, border: "1px solid #E2E8F0", borderRadius: "12px",
                padding: "10px 14px", fontSize: "14px", resize: "none",
                outline: "none", fontFamily: "inherit", lineHeight: "1.5",
                maxHeight: "100px", overflowY: "auto",
              }}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              style={{
                width: "40px", height: "40px", borderRadius: "10px",
                background: loading || !input.trim() ? "#E2E8F0" : "#2563EB",
                border: "none", cursor: loading || !input.trim() ? "default" : "pointer",
                display: "flex", alignItems: "center", justifyContent: "center",
                transition: "background 0.2s", flexShrink: 0,
              }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"
                  stroke={loading || !input.trim() ? "#94A3B8" : "white"}
                  strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
        </div>
      )}

      <style>{`
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }
        * { box-sizing: border-box; }
      `}</style>
    </>
  );
}
