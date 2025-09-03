# Speaker Identification Improvements Roadmap

This document tracks planned improvements to make multi-user speaker identification robust in noisy, overlapping, far‑field environments (e.g., a 5‑person car).

## Priorities (Impact → Effort)

1) Noise suppression + VAD tuning (WebRTC NS/RNNoise) – High impact / Low–Med effort
2) Temporal aggregation + hysteresis for IDs – High impact / Low effort
3) Overlap‑aware diarization (pyannote/NeMo/Riva) – High impact / Med–High effort
4) Multi‑embedding enrollment + augmentation – Med–High impact / Med effort
5) Score normalization/calibration (Z/T‑norm & env thresholds) – Med impact / Med effort
6) In‑memory ring buffer for segment extraction – Med impact (latency) / Low effort
7) Verification mode for known candidate sets – Med impact / Med effort
8) Session‑level mapping UX/API + manual override – Med impact / Low–Med effort
9) Metrics & evaluations (accuracy/latency) – Med impact / Low effort
10) Docs for thresholds and enrollment best practices – Med impact / Low effort

## Work Items

- Noise & VAD
  - Integrate WebRTC noise suppression or RNNoise front‑end before VAD
  - Tune VAD aggressiveness per environment; prevent over‑clipping
  - Optional AEC if TTS present; pre‑emphasis and AGC safeguards

- Temporal smoothing
  - Aggregate embeddings over sliding windows
  - Majority vote / exponential smoothing on identity; dampen rapid switches
  - Require sustained evidence to switch identities

- Overlap‑aware diarization
  - Add diarization with overlap support (pyannote.audio or NVIDIA NeMo/Riva)
  - Maintain track‑wise segments and per‑track histories

- Enrollment robustness
  - Store multiple embeddings per user (clean + noisy + varied distance)
  - Data augmentation: noise, reverb, speed/tempo, gain
  - Periodically refresh embeddings with in‑situ captures

- Scoring & calibration
  - Apply Z‑norm/T‑norm; environment‑specific thresholds
  - Prefer verification (is this one of these users?) for in‑car participants

- Low‑latency pipeline
  - Replace disk reads with in‑memory ring buffer; zero‑copy slices
  - Batch ASR/diarization windows aligned with buffer

- Session/UX
  - Expose `speaker_map` and confidence over WS/API
  - Manual override: “I’m Alice” updates mapping w/ short re‑enroll
  - Fallback to generic addressing when confidence low

- Metrics & evals
  - Track precision/recall per user, false accept/reject, switch rate
  - Latency budgets for extraction/matching/ASR

- Documentation
  - Threshold defaults per environment (quiet room vs. car)
  - Enrollment checklists and sample scripts

## Milestone 1 (fast wins)
- Noise suppression + VAD tuning
- Temporal aggregation + hysteresis
- Ring buffer segment extraction

## Milestone 2
- Multi‑embedding enrollment + augmentation
- Verification mode + score normalization

## Milestone 3
- Overlap‑aware diarization integration
- Session UX and manual overrides
- Metrics dashboards and evaluation harness
