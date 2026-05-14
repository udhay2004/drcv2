import React, { useEffect, useRef, useState, useCallback } from "react";
import "./App.css";

const API = process.env.REACT_APP_BACKEND_URL || "";

/* ═════════════════════════════════════════════════════════════
   Dr. CV — Voice Agent
═════════════════════════════════════════════════════════════ */

// ── tunables ─────────────────────────────────────────────
const VAD_THRESHOLD = 0.012;
const SILENCE_MS_BASE = 1100;
const SILENCE_MS_MIN = 700;
const SILENCE_MS_MAX = 2400;
const MIN_SPEECH_MS = 350;
const POST_TTS_GAP = 220;

const LEAD = (() => {
  const q = new URLSearchParams(window.location.search);
  return {
    name: q.get("name") || "",
    email: q.get("email") || "",
    company: q.get("company") || "",
    country: q.get("country") || "",
    stage: q.get("stage") || "",
  };
})();

const SYSTEM_PROMPT = (extracted) => `
You are Dr. CV — a Global Expansion Strategist at Comply Globally on a LIVE voice call.

CLIENT: ${LEAD.name || "Guest"} | ${LEAD.company || "—"} | Target: ${LEAD.country || "—"}
Extracted so far: ${JSON.stringify(extracted)}

VOICE CALL RULES:
- Keep responses short and natural.
- One question at a time.
- Match user's energy.
- No markdown or lists.
`;

function splitSentence(buf) {
  const m = buf.match(/^(.+?[.!?…])(\s+|$)/s);
  if (!m) return { sentence: null, rest: buf };
  return {
    sentence: m[1].trim(),
    rest: buf.slice(m[0].length),
  };
}

let _ctx = null;

function audioCtx() {
  if (!_ctx || _ctx.state === "closed") {
    _ctx = new (window.AudioContext || window.webkitAudioContext)();
  }

  if (_ctx.state === "suspended") {
    _ctx.resume();
  }

  return _ctx;
}

export default function App() {
  const [phase, setPhase] = useState("idle");
  const [transcript, setTranscript] = useState("");
  const [aiText, setAiText] = useState("");
  const [muted, setMuted] = useState(false);
  const [extracted, setExtracted] = useState({});
  const [splash, setSplash] = useState(true);
  const [history, setHistory] = useState([]);
  const [vadLevel, setVadLevel] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");

  const phaseRef = useRef("idle");
  const recRef = useRef(null);
  const streamRef = useRef(null);
  const analyserRef = useRef(null);
  const vadRafRef = useRef(null);
  const speechStartRef = useRef(null);
  const lastVoiceRef = useRef(null);
  const sentChunksRef = useRef([]);
  const silenceMsRef = useRef(SILENCE_MS_BASE);
  const audioElRef = useRef(null);
  const ttsQueueRef = useRef([]);
  const ttsPlayingRef = useRef(false);
  const historyRef = useRef([]);
  const extractedRef = useRef({});
  const mutedRef = useRef(false);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    historyRef.current = history;
  }, [history]);

  useEffect(() => {
    extractedRef.current = extracted;
  }, [extracted]);

  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);

  const stopMic = useCallback(() => {
    cancelAnimationFrame(vadRafRef.current);

    try {
      if (recRef.current && recRef.current.state !== "inactive") {
        recRef.current.stop();
      }
    } catch {}

    recRef.current = null;

    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    } catch {}

    streamRef.current = null;
    analyserRef.current = null;
    speechStartRef.current = null;
    lastVoiceRef.current = null;
    sentChunksRef.current = [];

    setVadLevel(0);
  }, []);

  const startMic = useCallback(async () => {
    if (phaseRef.current === "closed") return;
    if (streamRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });

      streamRef.current = stream;

      const ctx = audioCtx();
      const src = ctx.createMediaStreamSource(stream);

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.5;

      src.connect(analyser);
      analyserRef.current = analyser;

      const data = new Float32Array(analyser.fftSize);

      const mime = MediaRecorder.isTypeSupported(
        "audio/webm;codecs=opus"
      )
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const rec = new MediaRecorder(stream, {
        mimeType: mime,
        audioBitsPerSecond: 32000,
      });

      recRef.current = rec;
      sentChunksRef.current = [];

      rec.ondataavailable = (e) => {
        if (e.data.size > 0) {
          sentChunksRef.current.push(e.data);
        }
      };

      rec.onstop = () => {
        const chunks = sentChunksRef.current;
        sentChunksRef.current = [];

        if (!chunks.length) return;

        const blob = new Blob(chunks, { type: mime });

        if (speechStartRef.current) {
          commitTranscription(blob);
        }
      };

      rec.start(250);

      setPhase("listening");

      const tick = () => {
        if (!analyserRef.current) return;

        analyserRef.current.getFloatTimeDomainData(data);

        let sum = 0;

        for (let i = 0; i < data.length; i++) {
          sum += data[i] * data[i];
        }

        const rms = Math.sqrt(sum / data.length);

        setVadLevel(rms);

        const now = Date.now();

        if (rms > VAD_THRESHOLD) {
          if (!speechStartRef.current) {
            speechStartRef.current = now;
          }

          lastVoiceRef.current = now;

          const spoke = now - speechStartRef.current;

          if (spoke < 1500) {
            silenceMsRef.current = SILENCE_MS_MAX;
          } else if (spoke > 5000) {
            silenceMsRef.current = SILENCE_MS_MIN;
          } else {
            silenceMsRef.current = SILENCE_MS_BASE;
          }
        } else if (
          speechStartRef.current &&
          lastVoiceRef.current
        ) {
          const silentFor = now - lastVoiceRef.current;
          const spokeFor =
            lastVoiceRef.current - speechStartRef.current;

          if (
            silentFor >= silenceMsRef.current &&
            spokeFor >= MIN_SPEECH_MS
          ) {
            stopMic();
            return;
          }
        }

        vadRafRef.current = requestAnimationFrame(tick);
      };

      vadRafRef.current = requestAnimationFrame(tick);
    } catch (err) {
      console.error("mic err:", err);
      setErrorMsg("Microphone access denied.");
      setPhase("ready");
    }
  }, [stopMic]);

  const commitTranscription = useCallback(
    async (blob) => {
      setPhase("thinking");

      try {
        const fd = new FormData();

        fd.append("file", blob, "audio.webm");
        fd.append("language", "en");

        const r = await fetch(`${API}/api/transcribe`, {
          method: "POST",
          body: fd,
        });

        if (!r.ok) {
          throw new Error(`STT ${r.status}`);
        }

        const { text } = await r.json();

        const clean = (text || "").trim();

        if (clean.length < 2) {
          setPhase("ready");

          setTimeout(() => {
            if (
              !mutedRef.current &&
              phaseRef.current === "ready"
            ) {
              startMic();
            }
          }, 200);

          return;
        }

        setTranscript(clean);

        const newHist = [
          ...historyRef.current,
          {
            role: "user",
            content: clean,
          },
        ];

        setHistory(newHist);
        historyRef.current = newHist;

        extractFromTranscript(clean);

        streamChat(newHist);
      } catch (e) {
        console.error("commit", e);
        setErrorMsg("Couldn't transcribe.");
        setPhase("ready");
      }
    },
    [startMic]
  );

  const extractFromTranscript = async (text) => {
    try {
      const r = await fetch(`${API}/api/extract`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          transcript: text,
          existing: extractedRef.current,
        }),
      });

      if (!r.ok) return;

      const { extracted: ex } = await r.json();

      if (ex && Object.keys(ex).length > 0) {
        const merged = {
          ...extractedRef.current,
          ...ex,
        };

        setExtracted(merged);
        extractedRef.current = merged;
      }
    } catch {}
  };

  const streamChat = useCallback(async (hist) => {
    setAiText("");

    let buf = "";
    let full = "";

    setPhase("speaking");

    try {
      const r = await fetch(`${API}/api/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          system: SYSTEM_PROMPT(extractedRef.current),
          messages: hist.slice(-12),
        }),
      });

      if (!r.ok || !r.body) {
        throw new Error(`Chat ${r.status}`);
      }

      const reader = r.body.getReader();
      const dec = new TextDecoder();

      let raw = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        raw += dec.decode(value, { stream: true });

        const lines = raw.split("\n");

        raw = lines.pop();

        for (const ln of lines) {
          if (!ln.startsWith("data:")) continue;

          const p = ln.slice(5).trim();

          if (!p || p === "[DONE]") continue;

          try {
            const d = JSON.parse(p);

            if (d.type === "token" && d.text) {
              buf += d.text;
              full += d.text;

              setAiText(full);

              while (true) {
                const { sentence, rest } =
                  splitSentence(buf);

                if (!sentence) break;

                buf = rest;

                enqueueTTS(sentence);
              }
            } else if (d.type === "error") {
              throw new Error(d.message);
            }
          } catch {}
        }
      }

      if (buf.trim()) {
        enqueueTTS(buf.trim());
      }

      const ai = full.trim();

      const newHist = [
        ...hist,
        {
          role: "assistant",
          content: ai,
        },
      ];

      setHistory(newHist);
      historyRef.current = newHist;
    } catch (e) {
      console.error("chat", e);
      setErrorMsg("Connection issue.");
      setPhase("ready");
    }
  }, []);

  const enqueueTTS = useCallback(
    async (sentence) => {
      if (mutedRef.current) return;

      ttsQueueRef.current.push(sentence);

      if (ttsPlayingRef.current) return;

      ttsPlayingRef.current = true;

      stopMic();

      while (ttsQueueRef.current.length > 0) {
        const next = ttsQueueRef.current.shift();

        try {
          const r = await fetch(`${API}/api/tts`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              text: next,
              voice: "shimmer",
              speed: 1.05,
            }),
          });

          if (!r.ok) continue;

          const blob = await r.blob();

          const url = URL.createObjectURL(blob);

          await playAudio(url);

          URL.revokeObjectURL(url);
        } catch (e) {
          console.warn("tts", e);
        }
      }

      ttsPlayingRef.current = false;

      if (
        phaseRef.current !== "closed" &&
        !mutedRef.current
      ) {
        setTimeout(() => {
          setPhase("ready");
          startMic();
        }, POST_TTS_GAP);
      } else {
        setPhase("ready");
      }
    },
    [stopMic, startMic]
  );

  const playAudio = (url) =>
    new Promise((resolve) => {
      const a = audioElRef.current;

      if (!a) return resolve();

      a.src = url;

      a.onended = resolve;
      a.onerror = resolve;

      a.play().catch(() => resolve());
    });

  const begin = useCallback(async () => {
    setSplash(false);
    setErrorMsg("");
    setPhase("thinking");

    try {
      audioCtx();
    } catch {}

    const greet =
      LEAD.name && LEAD.country
        ? `Good to have you here, ${
            LEAD.name.split(" ")[0]
          }. What's the biggest challenge on your mind right now?`
        : "Welcome. What's the main challenge you're trying to solve?";

    const initHist = [
      {
        role: "assistant",
        content: greet,
      },
    ];

    setHistory(initHist);
    historyRef.current = initHist;

    setAiText(greet);

    setPhase("speaking");

    enqueueTTS(greet);
  }, [enqueueTTS]);

  useEffect(() => {
    return () => stopMic();
  }, [stopMic]);

  const onMicTap = () => {
    setErrorMsg("");

    if (
      phaseRef.current === "speaking" ||
      phaseRef.current === "thinking"
    ) {
      ttsQueueRef.current = [];

      if (audioElRef.current) {
        try {
          audioElRef.current.pause();
        } catch {}
      }

      ttsPlayingRef.current = false;

      setPhase("ready");

      setTimeout(() => startMic(), 100);

      return;
    }

    if (phaseRef.current === "listening") {
      stopMic();
      commitNow();
      return;
    }

    if (
      phaseRef.current === "ready" ||
      phaseRef.current === "idle"
    ) {
      startMic();
    }
  };

  const commitNow = () => {
    if (
      recRef.current &&
      recRef.current.state === "recording"
    ) {
      try {
        recRef.current.stop();
      } catch {}
    }
  };

  const onMute = () => {
    const next = !muted;

    setMuted(next);

    if (next) {
      ttsQueueRef.current = [];

      if (audioElRef.current) {
        try {
          audioElRef.current.pause();
        } catch {}
      }

      ttsPlayingRef.current = false;

      stopMic();

      setPhase("ready");
    }
  };

  const onReset = () => {
    stopMic();

    ttsQueueRef.current = [];
    ttsPlayingRef.current = false;

    if (audioElRef.current) {
      try {
        audioElRef.current.pause();
      } catch {}
    }

    setHistory([]);
    historyRef.current = [];

    setExtracted({});
    extractedRef.current = {};

    setAiText("");
    setTranscript("");

    setSplash(true);

    setPhase("idle");
  };

  const stateLabel =
    phase === "listening"
      ? "Listening"
      : phase === "thinking"
      ? "Thinking"
      : phase === "speaking"
      ? "Speaking"
      : phase === "ready"
      ? "Ready"
      : phase === "closed"
      ? "Done"
      : "—";

  return (
    <div className="dr-app" data-phase={phase}>
      <div className="grain" />

      <header className="topbar">
        <div className="brand">
          <div className="brand-name">
            Comply Globally
          </div>
        </div>

        <div className={`status-pill p-${phase}`}>
          {stateLabel}
        </div>
      </header>

      <main className="stage">
        <div className="orb-wrap">
          <div className="orb-core">
            <div className="orb-glyph">CV</div>

            <div
              className="orb-pulse"
              style={{
                transform: `scale(${
                  1 + Math.min(vadLevel * 12, 0.35)
                })`,
                opacity:
                  phase === "listening" ? 0.6 : 0,
              }}
            />
          </div>
        </div>

        {(transcript || aiText) && (
          <div className="tx-row">
            {transcript && (
              <div className="tx tx-you">
                <span className="tx-who">You</span>
                <span className="tx-text">
                  {transcript}
                </span>
              </div>
            )}

            {aiText && (
              <div className="tx tx-ai">
                <span className="tx-who">Dr. CV</span>
                <span className="tx-text">
                  {aiText}
                </span>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="ctrl">
        <button
          className="ibtn"
          onClick={onReset}
        >
          Reset
        </button>

        <button
          className={`mic-btn mic-${phase}`}
          onClick={onMicTap}
        >
          🎤
        </button>

        <button
          className={`ibtn ${muted ? "on" : ""}`}
          onClick={onMute}
        >
          {muted ? "Unmute" : "Mute"}
        </button>
      </footer>

      {errorMsg && (
        <div className="toast">
          {errorMsg}
        </div>
      )}

      <audio ref={audioElRef} hidden />

      {splash && (
        <div
          className="splash"
          onClick={begin}
        >
          <button className="sp-play">
            Start
          </button>
        </div>
      )}
    </div>
  );
}

function Bubble({ label, value, accent, testId }) {
  return (
    <div
      className={`bubble bubble-${accent}`}
      data-testid={testId}
    >
      <div className="bubble-label">
        {label}
      </div>

      <div className="bubble-value">
        {value}
      </div>
    </div>
  );
}
