import React, { useState } from "react";
import axios, { AxiosError } from "axios";

type Step =
  | "size"
  | "task"
  | "anchor"
  | "candidate"
  | "candidates"
  | "confirm"
  | "response";

type ChatMessage = {
  type: "user" | "bot";
  text: string;
};

export default function EntityMatchApp() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { type: "bot", text: "Welcome! Let's start — what model size would you like? (small, medium, large)" },
  ]);

  const [step, setStep] = useState<Step>("size");

  const [size, setSize] = useState<"small" | "medium" | "large">("small");
  const [task, setTask] = useState<"matching" | "comparing" | "selecting">("matching");
  const [anchor, setAnchor] = useState<string>("");
  const [candidate, setCandidate] = useState<string>("");
  const [candidates, setCandidates] = useState<string[]>([]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const addMessage = (msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  };

  const handleSubmit = async () => {
    const payload: Record<string, any> = {
      size,
      task,
      anchor,
      ...(task === "matching" && { candidate }),
      ...(task !== "matching" && { candidates }),
    };

    try {
      const res = await axios.post("http://localhost:8000/entity-match", payload);
      addMessage({
        type: "bot",
        text: "Here’s your result:\n" + JSON.stringify(res.data, null, 2),
      });
    } catch (err: unknown) {
      const axiosErr = err as AxiosError<{ error?: string }>;
      addMessage({
        type: "bot",
        text: "Error: " + (axiosErr.response?.data?.error || "Something went wrong."),
      });
    }
  };

  const handleNext = () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    addMessage({ type: "user", text: trimmed });

    switch (step) {
      case "size":
        if (["small", "medium", "large"].includes(trimmed)) {
          setSize(trimmed as any);
          addMessage({ type: "bot", text: "Got it. What task are you doing? (matching, comparing, selecting)" });
          setStep("task");
        } else {
          addMessage({ type: "bot", text: "Please enter one of: small, medium, or large." });
        }
        break;

      case "task":
        if (["matching", "comparing", "selecting"].includes(trimmed)) {
          setTask(trimmed as any);
          addMessage({ type: "bot", text: "Perfect. Now enter the anchor product description." });
          setStep("anchor");
        } else {
          addMessage({ type: "bot", text: "Please enter: matching, comparing, or selecting." });
        }
        break;

      case "anchor":
        setAnchor(trimmed);
        if (task === "matching") {
          addMessage({ type: "bot", text: "Now enter the candidate product." });
          setStep("candidate");
        } else {
          addMessage({ type: "bot", text: "Add a candidate. Type 'submit' when you're done adding." });
          setCandidates([]);
          setStep("candidates");
        }
        break;

      case "candidate":
        setCandidate(trimmed);
        setStep("confirm");
        addMessage({ type: "bot", text: "All set. Type 'submit' to run the match." });
        break;

      case "candidates":
        if (trimmed.toLowerCase() === "submit") {
          setStep("confirm");
          addMessage({ type: "bot", text: "Awesome. Type 'submit' again to run the match." });
        } else {
          setCandidates((prev) => [...prev, trimmed]);
          addMessage({ type: "bot", text: "Candidate added. Add more or type 'submit' to continue." });
        }
        break;

      case "confirm":
        if (trimmed.toLowerCase() === "submit") {
          setStep("response");
          setLoading(true);
          handleSubmit().then(() => setLoading(false));
        } else {
          addMessage({ type: "bot", text: "Type 'submit' when you're ready." });
        }
        break;

      case "response":
        addMessage({ type: "bot", text: "Conversation ended. Refresh to restart." });
        break;
    }

    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleNext();
    }
  };

  return (
    <div className="flex flex-col h-screen w-full bg-[#f7f7f8] text-sm">
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`rounded-xl px-4 py-3 max-w-[75%] whitespace-pre-wrap ${
                  msg.type === "user"
                    ? "bg-blue-600 text-white rounded-br-none"
                    : "bg-gray-200 text-gray-800 rounded-bl-none"
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-200 text-gray-600 px-4 py-3 rounded-xl rounded-bl-none">
                Loading...
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="border-t bg-white p-4">
        <div className="max-w-3xl mx-auto">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            placeholder="Type your response and hit Enter..."
            className="w-full border border-gray-300 rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          />
        </div>
      </div>
    </div>
  );
}
