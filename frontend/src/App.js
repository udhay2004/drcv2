"import React, { useEffect, useRef, useState, useCallback } from \"react\";
import \"./App.css\";

const API = process.env.REACT_APP_BACKEND_URL || \"\";

/* ═════════════════════════════════════════════════════════════
   Dr. CV — Voice Agent (fast, accent-tolerant, no self-hearing)
   ─────────────────────────────────────────────────────────────
   Pipeline:
     mic → MediaRecorder (webm/opus) → VAD-driven endpoint detect
       → /api/transcribe (Whisper)
       → /api/chat/stream (GPT-4o SSE, sentence-chunked)
         ↳ per sentence → /api/tts (mp3) → HTMLAudio play
     During TTS playback: mic tracks are STOPPED (.stop()) — no
     possible self-hearing. Stream is rebuilt fresh after TTS.
═════════════════════════════════════════════════════════════ */

// ── tunables ─────────────────────────────────────────────
const VAD_THRESHOLD = 0.012;   // RMS energy threshold for \"voice\"
const SILENCE_MS_BASE = 1100;  // base silence before commit
const SILENCE_MS_MIN = 700;    // long utterance commits faster
const SILENCE_MS_MAX = 2400;   // short utterance gets more grace
const MIN_SPEECH_MS = 350;     // ignore micro-bursts
const POST_TTS_GAP = 220;      // delay after TTS before mic reopens

const LEAD = (() => {
  const q = new URLSearchParams(window.location.search);
  return {
    name: q.get(\"name\") || \"\",
    email: q.get(\"email\") || \"\",
    company: q.get(\"company\") || \"\",
    country: q.get(\"country\") || \"\",
    stage: q.get(\"stage\") || \"\",
  };
})();

const SYSTEM_PROMPT = (extracted) => `You are Dr. CV — a Global Expansion Strategist at Comply Globally on a LIVE voice call.

CLIENT: ${LEAD.name || \"Guest\"} | ${LEAD.company || \"—\"} | Target: ${LEAD.country || \"—\"}
Extracted so far: ${JSON.stringify(extracted)}

VOICE CALL RULES (NON-NEGOTIABLE):
- 1–2 short sentences MAX per turn. One sentence is often best.
- ONE question per turn, at the end.
- Vary openers — never start two turns the same way.
- Natural fillers (\"Right.\", \"Got it.\", \"Hmm.\") max once per 4 turns.
- Zero markdown, no lists, no form numbers.
- Match user's energy. If they're brief, you're briefer.

EXAMPLES:
✓ \"Got it. What's your timeline?\"
✓ \"Wyoming was your instinct — what's driving that?\"
✓ \"That's the smarter move. Who else is involved?\"
✗ \"Great question! There are several things to consider...\"

EXPERTISE (drop one fact at a time, only if asked):
- Wyoming LLC: $62/yr, private, zero state income tax
- Delaware C-Corp: only for raising VC
- Banking: Mercury / Relay, no US SSN needed
- Tax: no US income tax if no US presence
- Wise: under 1.5% for transfers home

GOAL: discover their biggest challenge, timeline, decision-makers, success criteria. One thread at a time. After 8+ turns, wrap warmly.`;

// ── sentence splitter for streaming TTS ──────────────────
function splitSentence(buf) {
  const m = buf.match(/^(.+?[.!?…])(\s+|$)/s);
  if (!m) return { sentence: null, rest: buf };
  return { sentence: m[1].trim(), rest: buf.slice(m[0].length) };
}

// ── audio context (single) ───────────────────────────────
let _ctx = null;
function audioCtx() {
  if (!_ctx || _ctx.state === \"closed\") {
    _ctx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (_ctx.state === \"suspended\") _ctx.resume();
  return _ctx;
}

// ═════════════════════════════════════════════════════════
// Main Component
// ═════════════════════════════════════════════════════════
export default function App() {
  const [phase, setPhase] = useState(\"idle\"); // idle, ready, listening, thinking, speaking, closed
  const [transcript, setTranscript] = useState(\"\");
  const [aiText, setAiText] = useState(\"\");
  const [muted, setMuted] = useState(false);
  const [extracted, setExtracted] = useState({});
  const [splash, setSplash] = useState(true);
  const [history, setHistory] = useState([]);
  const [vadLevel, setVadLevel] = useState(0);
  const [errorMsg, setErrorMsg] = useState(\"\");

  // refs (mutable, not reactive)
  const phaseRef = useRef(\"idle\");
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

  useEffect(() => { phaseRef.current = phase; }, [phase]);
  useEffect(() => { historyRef.current = history; }, [history]);
  useEffect(() => { extractedRef.current = extracted; }, [extracted]);
  useEffect(() => { mutedRef.current = muted; }, [muted]);

  // ── HARD STOP MIC ─────────────────────────────────────
  // releases stream entirely — mic LED off, browser cannot hear
  const stopMic = useCallback(() => {
    cancelAnimationFrame(vadRafRef.current);
    try { if (recRef.current && recRef.current.state !== \"inactive\") recRef.current.stop(); } catch {}
    recRef.current = null;
    try { if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop()); } catch {}
    streamRef.current = null;
    analyserRef.current = null;
    speechStartRef.current = null;
    lastVoiceRef.current = null;
    sentChunksRef.current = [];
    setVadLevel(0);
  }, []);

  // ── START MIC + VAD ───────────────────────────────────
  const startMic = useCallback(async () => {
    if (phaseRef.current === \"closed\") return;
    if (streamRef.current) return; // already active

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

      // VAD setup
      const ctx = audioCtx();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.5;
      src.connect(analyser);
      analyserRef.current = analyser;
      const data = new Float32Array(analyser.fftSize);

      // MediaRecorder
      const mime = MediaRecorder.isTypeSupported(\"audio/webm;codecs=opus\")
        ? \"audio/webm;codecs=opus\"
        : \"audio/webm\";
      const rec = new MediaRecorder(stream, { mimeType: mime, audioBitsPerSecond: 32000 });
      recRef.current = rec;
      sentChunksRef.current = [];

      rec.ondataavailable = (e) => { if (e.data.size > 0) sentChunksRef.current.push(e.data); };
      rec.onstop = () => {
        const chunks = sentChunksRef.current;
        sentChunksRef.current = [];
        if (!chunks.length) return;
        const blob = new Blob(chunks, { type: mime });
        // commit only if we actually heard speech
        if (speechStartRef.current) {
          commitTranscription(blob);
        }
      };

      rec.start(250); // 250ms chunks
      setPhase(\"listening\");

      // VAD loop
      const tick = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getFloatTimeDomainData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
        const rms = Math.sqrt(sum / data.length);
        setVadLevel(rms);

        const now = Date.now();
        if (rms > VAD_THRESHOLD) {
          if (!speechStartRef.current) speechStartRef.current = now;
          lastVoiceRef.current = now;
          // adapt silence window based on speech length
          const spoke = now - speechStartRef.current;
          if (spoke < 1500) silenceMsRef.current = SILENCE_MS_MAX;
          else if (spoke > 5000) silenceMsRef.current = SILENCE_MS_MIN;
          else silenceMsRef.current = SILENCE_MS_BASE;
        } else if (speechStartRef.current && lastVoiceRef.current) {
          const silentFor = now - lastVoiceRef.current;
          const spokeFor = lastVoiceRef.current - speechStartRef.current;
          if (silentFor >= silenceMsRef.current && spokeFor >= MIN_SPEECH_MS) {
            // COMMIT — user finished speaking
            stopMic();
            return; // ondataavailable+onstop fires commit
          }
        }
        vadRafRef.current = requestAnimationFrame(tick);
      };
      vadRafRef.current = requestAnimationFrame(tick);
    } catch (err) {
      console.error(\"mic err:\", err);
      setErrorMsg(\"Microphone access denied.\");
      setPhase(\"ready\");
    }
  }, [stopMic]);

  // ── COMMIT: send audio → Whisper → chat ───────────────
  const commitTranscription = useCallback(async (blob) => {
    setPhase(\"thinking\");
    try {
      const fd = new FormData();
      fd.append(\"file\", blob, \"audio.webm\");
      fd.append(\"language\", \"en\");
      const r = await fetch(`${API}/api/transcribe`, { method: \"POST\", body: fd });
      if (!r.ok) throw new Error(`STT ${r.status}`);
      const { text } = await r.json();
      const clean = (text || \"\").trim();
      if (clean.length < 2) {
        // false trigger
        setPhase(\"ready\");
        setTimeout(() => { if (!mutedRef.current && phaseRef.current === \"ready\") startMic(); }, 200);
        return;
      }
      setTranscript(clean);
      const newHist = [...historyRef.current, { role: \"user\", content: clean }];
      setHistory(newHist);
      historyRef.current = newHist;
      extractFromTranscript(clean); // async, fire-and-forget
      streamChat(newHist);
    } catch (e) {
      console.error(\"commit\", e);
      setErrorMsg(\"Couldn't transcribe — retry?\");
      setPhase(\"ready\");
    }
  }, []);

  // ── extract live data → bubbles ───────────────────────
  const extractFromTranscript = async (text) => {
    try {
      const r = await fetch(`${API}/api/extract`, {
        method: \"POST\",
        headers: { \"Content-Type\": \"application/json\" },
        body: JSON.stringify({ transcript: text, existing: extractedRef.current }),
      });
      if (!r.ok) return;
      const { extracted: ex } = await r.json();
      if (ex && Object.keys(ex).length > 0) {
        const merged = { ...extractedRef.current, ...ex };
        setExtracted(merged);
        extractedRef.current = merged;
      }
    } catch {}
  };

  // ── stream chat → enqueue per-sentence TTS ────────────
  const streamChat = useCallback(async (hist) => {
    setAiText(\"\");
    let buf = \"\";
    let full = \"\";
    setPhase(\"speaking\"); // about to speak
    try {
      const r = await fetch(`${API}/api/chat/stream`, {
        method: \"POST\",
        headers: { \"Content-Type\": \"application/json\" },
        body: JSON.stringify({
          system: SYSTEM_PROMPT(extractedRef.current),
          messages: hist.slice(-12),
        }),
      });
      if (!r.ok || !r.body) throw new Error(`Chat ${r.status}`);
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let raw = \"\";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        raw += dec.decode(value, { stream: true });
        const lines = raw.split(\"
\");
        raw = lines.pop();
        for (const ln of lines) {
          if (!ln.startsWith(\"data:\")) continue;
          const p = ln.slice(5).trim();
          if (!p || p === \"[DONE]\") continue;
          try {
            const d = JSON.parse(p);
            if (d.type === \"token\" && d.text) {
              buf += d.text;
              full += d.text;
              setAiText(full);
              // emit complete sentences
              while (true) {
                const { sentence, rest } = splitSentence(buf);
                if (!sentence) break;
                buf = rest;
                enqueueTTS(sentence);
              }
            } else if (d.type === \"error\") {
              throw new Error(d.message);
            }
          } catch {}
        }
      }
      if (buf.trim()) enqueueTTS(buf.trim());
      const ai = full.trim();
      const newHist = [...hist, { role: \"assistant\", content: ai }];
      setHistory(newHist);
      historyRef.current = newHist;
    } catch (e) {
      console.error(\"chat\", e);
      setErrorMsg(\"Connection issue — tap mic to retry.\");
      setPhase(\"ready\");
    }
  }, []);

  // ── TTS queue: serial playback w/ mic OFF guarantee ───
  const enqueueTTS = useCallback(async (sentence) => {
    if (mutedRef.current) return;
    ttsQueueRef.current.push(sentence);
    if (ttsPlayingRef.current) return;
    ttsPlayingRef.current = true;
    // hard-stop mic before any audio plays
    stopMic();

    while (ttsQueueRef.current.length > 0) {
      const next = ttsQueueRef.current.shift();
      try {
        const r = await fetch(`${API}/api/tts`, {
          method: \"POST\",
          headers: { \"Content-Type\": \"application/json\" },
          body: JSON.stringify({ text: next, voice: \"shimmer\", speed: 1.05 }),
        });
        if (!r.ok) continue;
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        await playAudio(url);
        URL.revokeObjectURL(url);
      } catch (e) { console.warn(\"tts\", e); }
    }
    ttsPlayingRef.current = false;
    // resume listening
    if (phaseRef.current !== \"closed\" && !mutedRef.current) {
      setTimeout(() => {
        setPhase(\"ready\");
        startMic();
      }, POST_TTS_GAP);
    } else {
      setPhase(\"ready\");
    }
  }, [stopMic, startMic]);

  const playAudio = (url) =>
    new Promise((resolve) => {
      const a = audioElRef.current;
      if (!a) return resolve();
      a.src = url;
      a.onended = resolve;
      a.onerror = resolve;
      a.play().catch(() => resolve());
    });

  // ── INIT ──────────────────────────────────────────────
  const begin = useCallback(async () => {
    setSplash(false);
    setErrorMsg(\"\");
    setPhase(\"thinking\");
    // unlock audio context (mobile autoplay policies)
    try { audioCtx(); } catch {}
    // initial greeting
    const greet = LEAD.name && LEAD.country
      ? `Good to have you here, ${LEAD.name.split(\" \")[0]}. ${LEAD.company || \"Your venture\"} expanding into ${LEAD.country} — exactly what we work on. What's the biggest challenge on your mind right now?`
      : LEAD.name
      ? `Great to connect, ${LEAD.name.split(\" \")[0]}. Which market are you focused on, and what's standing in your way?`
      : `Welcome. Which market are you looking at, and what's the main challenge you're trying to solve?`;
    const initHist = [{ role: \"assistant\", content: greet }];
    setHistory(initHist);
    historyRef.current = initHist;
    setAiText(greet);
    setPhase(\"speaking\");
    enqueueTTS(greet);
  }, [enqueueTTS]);

  // unmount
  useEffect(() => () => stopMic(), [stopMic]);

  // ── handlers ──────────────────────────────────────────
  const onMicTap = () => {
    setErrorMsg(\"\");
    if (phaseRef.current === \"speaking\" || phaseRef.current === \"thinking\") {
      // barge in
      ttsQueueRef.current = [];
      if (audioElRef.current) { try { audioElRef.current.pause(); } catch {} }
      ttsPlayingRef.current = false;
      setPhase(\"ready\");
      setTimeout(() => startMic(), 100);
      return;
    }
    if (phaseRef.current === \"listening\") { stopMic(); commitNow(); return; }
    if (phaseRef.current === \"ready\" || phaseRef.current === \"idle\") startMic();
  };

  const commitNow = () => {
    // force commit current buffer
    if (recRef.current && recRef.current.state === \"recording\") {
      try { recRef.current.stop(); } catch {}
    }
  };

  const onMute = () => {
    const next = !muted;
    setMuted(next);
    if (next) {
      ttsQueueRef.current = [];
      if (audioElRef.current) try { audioElRef.current.pause(); } catch {}
      ttsPlayingRef.current = false;
      stopMic();
      setPhase(\"ready\");
    }
  };

  const onReset = () => {
    stopMic();
    ttsQueueRef.current = [];
    ttsPlayingRef.current = false;
    if (audioElRef.current) try { audioElRef.current.pause(); } catch {}
    setHistory([]); historyRef.current = [];
    setExtracted({}); extractedRef.current = {};
    setAiText(\"\"); setTranscript(\"\");
    setSplash(true);
    setPhase(\"idle\");
  };

  // ── derived UI state ──────────────────────────────────
  const stateLabel =
    phase === \"listening\" ? \"Listening\"
    : phase === \"thinking\" ? \"Thinking\"
    : phase === \"speaking\" ? \"Speaking\"
    : phase === \"ready\" ? \"Ready\"
    : phase === \"closed\" ? \"Done\"
    : \"—\";

  return (
    <div className=\"dr-app\" data-phase={phase}>
      {/* ambient bg */}
      <div className=\"amb amb-1\" />
      <div className=\"amb amb-2\" />
      <div className=\"amb amb-3\" />
      <div className=\"grain\" />

      {/* topbar */}
      <header className=\"topbar\">
        <div className=\"brand\">
          <div className=\"brand-mark\"><span /></div>
          <div>
            <div className=\"brand-name\">Comply Globally</div>
            <div className=\"brand-sub\">Global Expansion Advisory</div>
          </div>
        </div>
        <div className={`status-pill p-${phase}`} data-testid=\"status-pill\">
          <span className=\"dot\" />
          <span>{stateLabel}</span>
        </div>
        <div className=\"lead-tag\">
          {LEAD.name && <span>{LEAD.name.split(\" \")[0]}</span>}
          {LEAD.country && <span className=\"sep\">·</span>}
          {LEAD.country && <span>{LEAD.country}</span>}
        </div>
      </header>

      {/* main stage */}
      <main className=\"stage\">
        {/* orb */}
        <div className=\"orb-wrap\" data-testid=\"voice-orb\">
          <div className=\"ring r1\" />
          <div className=\"ring r2\" />
          <div className=\"ring r3\" />
          <div className={`orb-halo h-${phase}`} />
          <div className=\"orb-core\">
            <div className=\"orb-glyph\">CV</div>
            <div className=\"orb-line\" />
            <div className=\"orb-sub\">Dr. CV · Strategist</div>
            <div className=\"orb-pulse\" style={{
              transform: `scale(${1 + Math.min(vadLevel * 12, 0.35)})`,
              opacity: phase === \"listening\" ? 0.6 : 0,
            }} />
          </div>
        </div>

        {/* data bubbles */}
        <aside className=\"bubbles\" data-testid=\"data-bubbles\">
          <div className=\"bubbles-title\">Session Memory</div>
          {Object.keys(extracted).length === 0 ? (
            <div className=\"bubble bubble-empty\">
              <span className=\"bubble-label\">Listening for context…</span>
              <span className=\"bubble-hint\">Goal · Timeline · Blockers · Decision-makers</span>
            </div>
          ) : (
            <>
              {extracted.goal && <Bubble label=\"Goal\" value={extracted.goal} accent=\"jade\" testId=\"bubble-goal\" />}
              {extracted.timeline && <Bubble label=\"Timeline\" value={extracted.timeline} accent=\"gold\" testId=\"bubble-timeline\" />}
              {extracted.budget && <Bubble label=\"Budget\" value={extracted.budget} accent=\"aurora\" testId=\"bubble-budget\" />}
              {extracted.blocker && <Bubble label=\"Blocker\" value={extracted.blocker} accent=\"flame\" testId=\"bubble-blocker\" />}
              {extracted.decision_maker && <Bubble label=\"Decides\" value={extracted.decision_maker} accent=\"ice\" testId=\"bubble-decision\" />}
              {extracted.success && <Bubble label=\"Success\" value={extracted.success} accent=\"mint\" testId=\"bubble-success\" />}
              {extracted.industry && <Bubble label=\"Industry\" value={extracted.industry} accent=\"jade\" testId=\"bubble-industry\" />}
            </>
          )}
        </aside>
      </main>

      {/* transcript bubble */}
      {(transcript || aiText) && (
        <div className=\"tx-row\">
          {transcript && (
            <div className=\"tx tx-you\" data-testid=\"user-transcript\">
              <span className=\"tx-who\">You</span>
              <span className=\"tx-text\">{transcript}</span>
            </div>
          )}
          {aiText && (
            <div className=\"tx tx-ai\" data-testid=\"ai-transcript\">
              <span className=\"tx-who\">Dr. CV</span>
              <span className=\"tx-text\">{aiText}</span>
            </div>
          )}
        </div>
      )}

      {/* controls */}
      <footer className=\"ctrl\">
        <button className=\"ibtn\" onClick={onReset} title=\"Restart\" data-testid=\"reset-btn\">
          <svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" strokeWidth=\"1.8\">
            <path d=\"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8\"/><path d=\"M3 3v5h5\"/>
          </svg>
        </button>

        <button
          className={`mic-btn mic-${phase}`}
          onClick={onMicTap}
          disabled={phase === \"closed\"}
          data-testid=\"mic-btn\"
        >
          <svg width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" strokeWidth=\"1.7\">
            <rect x=\"9\" y=\"2\" width=\"6\" height=\"12\" rx=\"3\"/>
            <path d=\"M5 10a7 7 0 0 0 14 0M12 19v3M8 22h8\"/>
          </svg>
        </button>

        <button className={`ibtn ${muted ? \"on\" : \"\"}`} onClick={onMute} title=\"Mute\" data-testid=\"mute-btn\">
          <svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" strokeWidth=\"1.8\">
            {muted
              ? <><path d=\"M11 5 6 9H2v6h4l5 4V5z\"/><line x1=\"23\" y1=\"9\" x2=\"17\" y2=\"15\"/><line x1=\"17\" y1=\"9\" x2=\"23\" y2=\"15\"/></>
              : <><path d=\"M11 5 6 9H2v6h4l5 4V5z\"/><path d=\"M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07\"/></>}
          </svg>
        </button>
      </footer>

      <div className=\"hint\" data-testid=\"hint-text\">
        {phase === \"listening\" ? \"Take your time. I'm listening.\" :
         phase === \"thinking\" ? \"Thinking…\" :
         phase === \"speaking\" ? \"Tap mic to interrupt\" :
         phase === \"ready\" ? \"Tap mic to speak\" :
         phase === \"idle\" ? \"Ready\" : \"\"}
      </div>

      {errorMsg && <div className=\"toast\" data-testid=\"error-toast\">{errorMsg}</div>}

      <audio ref={audioElRef} hidden />

      {/* splash */}
      {splash && (
        <div className=\"splash\" onClick={begin} data-testid=\"splash\">
          <div className=\"sp-logo\">Comply Globally</div>
          <div className=\"sp-rule\" />
          <button className=\"sp-play\" data-testid=\"start-btn\">
            <svg width=\"26\" height=\"26\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" strokeWidth=\"1.4\">
              <polygon points=\"5 3 19 12 5 21 5 3\"/>
            </svg>
          </button>
          <div className=\"sp-hint\">Tap to begin your session with Dr. CV</div>
        </div>
      )}
    </div>
  );
}

function Bubble({ label, value, accent, testId }) {
  return (
    <div className={`bubble bubble-${accent}`} data-testid={testId}>
      <div className=\"bubble-label\">{label}</div>
      <div className=\"bubble-value\">{value}</div>
    </div>
  );
}
"
