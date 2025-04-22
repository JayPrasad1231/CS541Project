"use client";
import Image from "next/image";
import { motion } from "framer-motion";
import React, { useEffect, useState, useRef } from 'react';
import axios, { AxiosError } from "axios";
import redPfp from "../imgs/red.png";
import glumbus_subtubbo from "../imgs/gs.png";
import { v4 as uuidv4 } from "uuid";

type Step =
  | "size"
  | "task"
  | "background"
  | "anchor"
  | "candidate"
  | "candidates"
  | "confirm"
  | "response";

type ChatMessage = {
  id: string;
  type: "user" | "bot";
  text: string;
};

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: uuidv4(),
      type: "bot",
      text: "Welcome! Let's start — what model size would you like? (small, medium, large)",
    },
  ]);

  const [step, setStep] = useState<Step>("size");

  const [size, setSize] = useState<"small" | "medium" | "large">("small");
  const [task, setTask] = useState<"matching" | "comparing" | "selecting">("matching");
  const [anchor, setAnchor] = useState<string>("");
  const [candidate, setCandidate] = useState<string>("");
  const [candidates, setCandidates] = useState<string[]>([]);
  const [background, setBackground] = useState<string | null>(null);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const animatedMessagesRef = useRef<Set<string>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const addMessage = (msg: Omit<ChatMessage, "id">) => {
    const withId = { ...msg, id: uuidv4() };
    setMessages((prev) => [...prev, withId]);
  };

  type Payload = {
    size: "small" | "medium" | "large";
    task: "matching" | "comparing" | "selecting";
    anchor: string;
    background?: string | null;
    candidate?: string;
    candidates?: string[];
  };

  const TypewriterText = ({ id, text }: { id: string; text: string }) => {
    const [displayed, setDisplayed] = useState(
      animatedMessagesRef.current.has(id) ? text : ""
    );

    useEffect(() => {
      if (animatedMessagesRef.current.has(id)) return;

      let i = 0;
      const interval = setInterval(() => {
        i++;
        setDisplayed(text.slice(0, i));
        if (i >= text.length) {
          animatedMessagesRef.current.add(id);
          clearInterval(interval);
        }
      }, 25);

      return () => clearInterval(interval);
    }, [id, text]);

    return <>{displayed}</>;
  };

  const handleSubmit = async () => {
    const payload: Payload = {
      size,
      task,
      anchor,
      background,
      ...(task === "matching" && { candidate }),
      ...(task !== "matching" && { candidates }),
    };

    try {
      const res = await axios.post("http://localhost:8000/entity-match", payload);
      addMessage({
        type: "bot",
        text: "Here's your result:\n" + JSON.stringify(res.data, null, 2),
      });
    } catch (err: unknown) {
      const axiosErr = err as AxiosError<{ error?: string }>;
      addMessage({
        type: "bot",
        text: "Error: " + (axiosErr.response?.data?.error || "Something went wrong."),
      });
    }
  };

  const resetConversation = () => {
    setMessages([
      {
        id: uuidv4(),
        type: "bot",
        text: "Welcome! Let's start — what model size would you like? (small, medium, large)",
      },
    ]);
    setStep("size");
    setSize("small");
    setTask("matching");
    setAnchor("");
    setCandidate("");
    setCandidates([]);
    setBackground(null);
    setInput("");
    animatedMessagesRef.current.clear();
  };

  const handleNext = () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    if (trimmed.toLowerCase() === "restart") {
      resetConversation();
      return;
    }

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
          addMessage({ type: "bot", text: "Optional: Enter any background context or type 'none'." });
          setStep("background");
        } else {
          addMessage({ type: "bot", text: "Please enter: matching, comparing, or selecting." });
        }
        break;

      case "background":
        setBackground(trimmed.toLowerCase() === "none" ? null : trimmed);
        addMessage({
          type: "bot",
          text: "Perfect. Now enter the anchor product description.",
        });
        setStep("anchor");
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
        addMessage({ type: "bot", text: "Conversation ended. Type 'restart' to begin again." });
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

  const bounceVariants = {
    initial: { scale: 0.8, opacity: 0 },
    animate: {
      scale: 1,
      opacity: 1,
      transition: { type: "spring", stiffness: 260, damping: 20 },
    },
  };

  return (
    <div className="flex flex-col h-screen w-full">
      <div className="flex items-center justify-center bg-gradient-to-r from-amber-500 to-amber-700 shadow-md">
        <div className="flex items-center py-4 px-6">
          <svg className="w-10 h-10 mr-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <h1 className="text-3xl font-semibold text-white tracking-tight">Entity Matching Bot</h1>
        </div>
      </div>

      <div className="flex-1 flex items-center justify-center p-4 md:p-8">
        <div className="bg-white rounded-xl shadow-lg w-full max-w-4xl flex flex-col" style={{ height: "85vh" }}>
          <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-4 py-6">
            <div className="flex flex-col items-center justify-center mb-8">
              <div className="border-4 border-slate-950 rounded-full p-1 shadow-lg">
                <Image src={glumbus_subtubbo} alt="Mr. Bot" width={120} height={120} className="rounded-full" />
              </div>
              <h2 className="text-2xl font-bold mt-4 text-amber-600">Conversation with Mr. Bot</h2>
            </div>

            <div className="space-y-6">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex items-end gap-3 ${msg.type === "user" ? "justify-end" : "justify-start"}`}
                >
                  {msg.type === "bot" && (
                    <div className="flex-shrink-0 border-2 border-black rounded-full self-end">
                      <Image src={glumbus_subtubbo} alt="bot avatar" width={60} height={60} className="rounded-full" />
                    </div>
                  )}
                  <motion.div
                    initial="initial"
                    animate="animate"
                    variants={bounceVariants}
                    className={`rounded-3xl px-4 py-3 max-w-[75%] whitespace-pre-wrap text-xl border-2 border-black ${
                      msg.type === "user"
                        ? "bg-amber-600 text-white rounded-br-none ml-auto"
                        : "bg-orange-100 text-gray-800 rounded-bl-none"
                    }`}
                  >
                    {msg.type === "bot" ? (
                      <TypewriterText id={msg.id} text={msg.text} />
                    ) : (
                      msg.text
                    )}
                  </motion.div>
                  {msg.type === "user" && (
                    <div className="flex-shrink-0 border-2 border-black rounded-full self-end">
                      <Image src={redPfp} alt="user avatar" width={60} height={60} className="rounded-full" />
                    </div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="flex justify-start items-end">
                  <motion.div
                    initial="initial"
                    animate="animate"
                    variants={bounceVariants}
                    className="bg-gray-200 text-gray-600 px-4 py-3 rounded-xl rounded-bl-none border-2 border-black"
                  >
                    Loading...
                  </motion.div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className="border-t bg-gray-50 p-4 rounded-b-xl">
            <div className="flex items-end gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={3}
                placeholder="Type your response and hit Enter..."
                className="flex-1 border-2 border-black text-xl text-amber-600 rounded-md p-3 focus:outline-none focus:border-transparent focus:ring-2 focus:ring-amber-600 resize-none"
              />
              <div className="border-2 border-black rounded-full">
                <Image src={redPfp} alt="user" width={80} height={80} className="rounded-full" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
