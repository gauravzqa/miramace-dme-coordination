# Voice smoke test (Mac, local LiveKit)

The PCP call runs as two in-process LiveKit `AgentSession`s (caller "mira-caller" ↔ persona "pcp-frontdesk")
against a local `livekit-server --dev`. If anything fails, the call is re-run over text automatically
(`transport="text_fallback"`, flag `voice_failed:<reason>`), so the demo never blocks on voice.

## 1. One-time setup

```bash
brew install livekit livekit-cli          # livekit-server + lk
cd ~/MiraMace/app
uv sync --extra voice                     # livekit-agents[openai]==1.8.10, livekit-plugins-openai==1.8.10
cp .env.example .env                      # then fill in OPENAI_API_KEY and MIRA_TEXT_MODEL
```

## 2. Run

Terminal A:
```bash
livekit-server --dev                      # ws://localhost:7880, key devkey / secret secret
```

Terminal B:
```bash
cd ~/MiraMace/app
export OPENAI_API_KEY=sk-...              # or via .env
export MIRA_LLM_MODE=live MIRA_TEXT_MODEL=<your text model>
export MIRA_VOICE=1 MIRA_VOICE_LEGS=pcp MIRA_DEMO_ROOM=mira-pcp-call
export LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
# optional: MIRA_STT_MODEL=gpt-4o-mini-transcribe MIRA_TTS_MODEL=gpt-4o-mini-tts
#           MIRA_TTS_VOICE_CALLER=sage MIRA_TTS_VOICE_PERSONA=coral MIRA_TURN_MODEL=<faster model>

# once B7 wires the CLI:
uv run mira call pcp --voice --listen
# works today, without the CLI:
uv run python -m miramace.voice.transport --listen
```

`--listen` prints (and opens) a link like
`https://meet.livekit.io/custom?liveKitUrl=ws://localhost:7880&token=<hidden subscribe-only token>`.
Open it in **Chrome**, turn mic/camera off, click Join and allow audio. The script waits up to 60 s for the
`listener` participant before the persona picks up. Alternative token via the CLI (paste into the same page):

```bash
lk token create --url ws://localhost:7880 --api-key devkey --api-secret secret \
  --join --room mira-pcp-call --identity listener --valid-for 1h
```

Expected: Maria answers ("Sunrise Family Medicine, this is Maria"), the caller discloses it is an automated AI
assistant, verifies the practice, asks the open items one at a time, reads back, and hangs up. A rich table
prints the caller-side transcript (STT of Maria as `text`, Maria's true line as `heard_text`), claims and flags,
with `transport=voice`.

## 3. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Agents never hear each other, persona greeting is not followed by the caller | Agents ignore AGENT-kind participants by default. We pass `room_io.RoomOptions(participant_identity=<other>, participant_kinds=[STANDARD, AGENT])`. Check both identities are in the room: `lk room participants list mira-pcp-call --url ws://localhost:7880 --api-key devkey --api-secret secret`. A stale room from a crashed run: `lk room delete mira-pcp-call ...` and retry. |
| Errors mentioning the turn detector / inference gateway / LiveKit Cloud | The default turn detector is `inference.TurnDetector()` (cloud). We set `turn_handling={"turn_detection": "vad", "interruption": {"enabled": False}, "preemptive_generation": {"enabled": False}}` and a local `inference.VAD(model="silero")`. If `livekit-local-inference` failed to install, `uv sync --extra voice` again. |
| Long silence, then "Hello, are you still there?" | 20 s silence watchdog: two nudges, then the caller says goodbye (`end_reason` no_answer/silence → text fallback if <2 persona utterances). Usually STT got nothing — check `OPENAI_API_KEY` and that the persona's TTS is audible in the listener tab. |
| `voice_failed:connect` | `livekit-server --dev` not running or wrong `LIVEKIT_URL`. |
| `voice_failed:livekit_missing` | Voice extra not installed: `uv sync --extra voice`. |
| `voice_failed:timeout` | Call exceeded 240 s; slow text model — set `MIRA_TURN_MODEL` to a faster model. |
| `voice_failed:caller_brain:…` / `persona_brain:…` | The LLM brain raised (bad key/model). Text fallback will usually fail the same way; fix the model id. |
| Listener page shows nothing / no audio | Use Chrome, click on the page to allow autoplay. The listener is hidden, so the agents don't see it. |
