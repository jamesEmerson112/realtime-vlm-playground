# Pipeline Optimizations from ProTAS

Source: Shen & Elhamifar, "Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos" (CVPR 2024)

Date: 2026-04-15

This document synthesizes two optimizations from the ProTAS paper that are applicable to our API-based VLM pipeline: **Task Graph Constraints** and **Progress Prediction**. Both address the same core failure mode we observed in our baseline runs: the VLM makes predictions that are locally plausible but procedurally impossible.

---

## The Problem: Online Action Segmentation

The paper addresses a fundamental mismatch: most action segmentation models are trained offline (seeing entire videos) but must run online (seeing only past and current frames). When an offline-trained model is applied to streaming video, performance collapses. On the EgoProceL dataset, accuracy drops from 69.5% to 13.2% and F1@0.5 drops from 48.8% to 5.4% (Table 2, ASFormer backbone).

The two dominant failure modes in the online setting are:

1. **Oversegmentation** -- the model flickers between action predictions, especially early in a video when few frames are available. It might predict "spread peanut butter" for 3 frames, switch to "apply jam" for 2 frames, then back to "spread peanut butter." Each flicker creates a spurious segment boundary.

2. **Procedure-inconsistent predictions** -- the model predicts actions in an impossible order. For example, predicting "put tea bag into trash can" before "steep tea bag for 3 minutes," or predicting "put bowl in microwave" before "measure water" and "pour water to bowl."

These are exactly the problems we saw in our baseline pipeline: step tracking got stuck at step 3 because the VLM kept oscillating between "step 3 complete" and "error: door not closed" (oversegmentation), and when it did advance, it sometimes skipped prerequisite steps entirely (procedure inconsistency).

The paper's first contribution (CAS -- causal architecture modification) addresses the train-test mismatch by making models causal during training. This requires model retraining and isn't applicable to our API-based VLM approach. The other two contributions -- progress prediction and task graph constraints -- are conceptually applicable and are the focus of this document.

### Why We Don't Train Custom Models

ProTAS achieves its results by training specialized neural network modules (causal backbones, GRU progress heads, learnable graph matrices) on labeled procedural video datasets. All three upgrades -- CAS, APP, and task graph -- are trained end-to-end with supervision, meaning they require frame-level ground-truth annotations of which action is happening at every frame, plus manually defined or transcript-derived task graphs as training signal.

This is a fundamentally different approach from ours. Our pipeline uses a general-purpose VLM (Gemini 2.5 Flash) through an API, with zero task-specific training. We send a single frame plus a text prompt and get back a natural language response. The VLM has never seen our procedures, our videos, or our ground truth labels.

Competitors who train custom models on procedure-specific data will likely achieve higher scores because they are fitting to the task distribution. ProTAS itself closes the offline-to-online gap substantially: CAS+APP+TG recovers from 5.4% F1@0.5 (naive online) to 36.8% (Table 2, ASFormer on EgoProceL), compared to the 48.8% offline ceiling. That recovery comes from supervised training on the same domain.

We deliberately chose the zero-shot API approach because:

1. **No training data required** -- our pipeline works on any procedure JSON + video with no labeled training set. ProTAS needs hundreds of annotated videos per task.
2. **Generalization** -- a trained model is specific to the procedures and environments it was trained on (e.g., EgoPER's 5 cooking tasks). Our pipeline handles arbitrary procedures without retraining.
3. **Practical deployment** -- training and maintaining custom vision models per procedure is operationally expensive. API-based VLMs improve automatically as providers update their models.
4. **The assignment constraint** -- this is a take-home assignment evaluating pipeline design and prompt engineering against a general-purpose VLM API, not custom model training.

The tradeoff is clear: custom-trained models will outscore zero-shot VLM pipelines on benchmarks where training data matches test data. Our bet is that the gap narrows as VLMs improve, and the operational simplicity of zero-shot is worth the score difference for real-world deployment.

Where we **can** borrow from ProTAS without training: task graph constraints (using procedure JSON as a free manual graph) and progress estimation (approximated via VLM prompting). These are the conceptual insights that transfer to our architecture.

---

## Optimization 1: Task Graph Constraints

### What a Task Graph Is

A task graph is a directed acyclic graph (DAG) where nodes represent actions (steps) and edges represent ordering dependencies. It encodes all feasible ways to execute a procedure. For example, in "making a peanut butter and jelly tortilla," the task graph captures that you must put the tortilla on the cutting board before spreading peanut butter, you must spread both peanut butter and jelly before rolling the tortilla, but the order of spreading peanut butter vs. jelly is flexible (Figure 1c in the paper).

The graph encodes two types of constraints:

- **Prerequisites (predecessors)** -- certain actions must complete before an action can start. You must take the tortilla out of its bag and put it on the board before applying jam on it. Formally, every node k has a set of predecessor nodes Ap(k), and all predecessors must be completed before k can begin.

- **Irreversibility (successors)** -- certain actions cannot be performed after other actions have completed. You cannot apply peanut butter again after you have rolled the tortilla. Formally, every node k has a set of successor nodes As(k), and none of k's successors should have started if k is still being predicted.

### How the Paper Applies Task Graph Constraints

The paper defines two penalty scores for each action k at time t.

The **predecessor penalty** (Eq. 7) measures how far the predecessors of k are from completion:

    alpha_p(t,k) = sum over k' in Ap(k) of (1 - c(t,k'))

This score is zero when all predecessors of k have been completed before frame t. If any predecessor is incomplete, the penalty is positive -- the more incomplete predecessors, the higher the penalty.

The **successor penalty** (Eq. 8) measures whether any successor of k has already started:

    alpha_s(t,k) = sum over k' in As(k) of c(t,k')

This score is zero when no successor of k has started before time t. If any successor has progressed, the penalty is positive.

At inference time, these penalties are subtracted from the model's raw logits before the softmax (Eq. 12):

    y(t,k) = softmax( o(t,k) - eta * (alpha_p(t,k) + alpha_s(t,k)) )

The hyperparameter eta controls how aggressively the task graph overrides the model's raw predictions. A higher eta means the task graph dominates; a lower eta means the model's visual evidence takes precedence.

The effect: if the model tries to predict an action whose prerequisites haven't been completed, the penalty pushes that action's probability down. If the model tries to predict an action whose successors have already started (meaning the procedure has moved past this point), the penalty again pushes it down. The result is smoother, procedure-consistent predictions.

### The Paper's Concrete Examples

In the "making tea" qualitative result (Figure 7), the CAS+APP model predicts "put tea bag into trash can" before "steep tea bag for 3 minutes." The task graph knows that steeping must precede disposal -- it penalizes the trash-can prediction while steeping hasn't completed, and the corrected output matches the ground truth.

In the "making oatmeal" qualitative result, CAS+APP fails to correctly detect "measure water" and "pour water to bowl" before "put bowl in microwave." The task graph enforces that measuring and pouring are prerequisites for microwaving, correcting the prediction sequence.

### How the Paper Builds Task Graphs

Two methods (Section 4.4):

**Transcript-based** -- count the frequency of action i occurring before action j across all training video transcripts. This produces soft predecessor/successor indicator matrices Mp and Ms (K x K, where K is the number of actions). If action i always appears before action j across all training videos, entry (i,j) in Mp is close to 1. This approach is automatic and requires no external knowledge -- just observed action orderings.

**Learnable** -- initialize from the transcript-based graph, then set the indicator matrices as learnable parameters during training, with a regularization term (MSE loss) to prevent degenerate solutions where all edges disappear.

Table 3 shows the results: the transcript-based graph is already close to the manually-built graph (only a slight gap on EgoPER), confirming that observed action orderings capture most of the procedural structure. The learnable graph can further refine this, though with diminishing returns.

### Results

Task graph constraints consistently improve Edit similarity and F1 scores across all three datasets (Tables 1-2). Edit similarity specifically penalizes oversegmentation, so improvement there means smoother, more procedure-consistent predictions. The gains are modest in absolute terms (1-3% F1@0.5) but meaningful because they fix qualitative failures -- wrong-ordering errors that no amount of visual accuracy can resolve.

### Our Angle

Our procedure JSONs already define step ordering explicitly. Each procedure has a `steps` array where steps are numbered sequentially. This is a free, manually-built task graph -- arguably better than a transcript-based one because it captures the intended procedure, not just observed behavior. No training or frequency counting needed. We can implement the same predecessor/successor penalty logic directly in our pipeline's step-tracking state machine.

---

## Optimization 2: Progress Prediction

### What Progress Means

Instead of treating action detection as binary ("step 3: yes or no"), progress estimation assigns a continuous value from 0.0 to 1.0 representing how far through an action the performer has progressed. The paper defines the target progress for action k at frame t as (Eq. 4):

    p*(t,k) = (t - t_s) / (t_e - t_s)

where t_s is the start frame and t_e is the end frame of the action segment. This is a simple linear ramp: 0 at the start of the action, 1 at completion.

### Why It Helps

The key insight is what the paper calls the "30% rule": if an action was at 30% progress at the previous time instant, the performer will continue the same action with high probability. Conversely, if progress is near 100%, a new action is likely to start.

This continuous signal directly addresses oversegmentation. When the model produces a binary "step 3: yes" followed by "step 3: no" followed by "step 3: yes" (three segment boundaries in rapid succession), progress provides a smoothing signal. If progress was rising steadily (10%, 20%, 30%), the interruption is likely spurious -- the action is still underway. The model learns to trust the progress trajectory rather than making frame-by-frame binary decisions.

In our baseline pipeline, this manifests as the VLM oscillating between "step complete" and "error detected" on consecutive frames. A progress signal would indicate the step is still in progress, dampening the oscillation.

### Which Actions Benefit Most

The paper's action-wise analysis (Figure 5) reveals that progress prediction helps most for actions with visible appearance change over their duration:

**High benefit (up to +18% F1@0.5):**
- Cut carrot -- the carrot gets progressively shorter, pieces accumulate
- Peel cucumber -- skin gradually disappears
- Spread peanut butter on tortilla -- coverage area visibly expands
- Apply pizza sauce on batter -- sauce coverage increases
- Fold paper filter -- shape changes predictably

**Low or no benefit:**
- Pour oil in pan -- brief action with no gradual visual change
- Mix all contents -- stirring looks similar at 10% and 90%
- Add salt -- momentary action, no meaningful duration to track

The pattern: progress estimation works when the visual scene contains cues about how far along the action is. Actions where "halfway done" looks different from "just started" benefit the most. Actions that are either instantaneous or visually static throughout their duration don't benefit.

### Our Angle

We can approximate progress estimation without retraining by asking the VLM to estimate how far through the current step the technician has progressed. Rather than a GRU predicting progress from learned features, we use a prompt like "estimate what percentage of this step appears complete based on the visible scene." If the VLM reports progress climbing steadily (20%, 40%, 60%), we maintain the current step prediction. If progress plateaus near 100%, we watch for a transition to the next step. Same principle as the paper -- continuous signal reduces binary classification flickering -- implemented through prompt engineering rather than model architecture.

This will likely work better for some steps than others, matching the paper's findings. Steps involving visible physical change (tightening a bolt, closing a panel, routing a wire) will show progress cues. Steps that are momentary (pressing a button, flipping a switch) won't benefit and shouldn't be tracked with progress.

---

## How They Connect

Task graph constraints and progress prediction are not independent optimizations -- progress feeds directly into the task graph.

The task graph needs to know which actions have been completed to compute its penalties. The paper defines the completion state of action k at time t as (Eq. 6):

    c(t,k) = max over t' = 1,...,t-1 of p(t',k)

That is, the completion state is the maximum progress ever observed for action k up to the current time. If the progress for "steep tea bag" peaked at 0.95 at some earlier time, then c(t, steep) = 0.95 -- the system considers steeping nearly complete.

This completion state is what the predecessor and successor penalties use. When the predecessor penalty checks whether predecessors of "put tea bag in trash" are completed, it checks whether the progress of "steep tea bag" ever reached near 1.0. When the successor penalty checks whether successors of an action have started, it checks whether any successor's progress has risen above 0.

So the data flow is: progress estimation produces p(t,k) for each action at each frame, which feeds into completion state c(t,k) via cumulative maximum, which feeds into the predecessor/successor penalties, which modify the model's logits before softmax. Progress is the input signal; the task graph is the constraint engine that uses that signal.

In our pipeline, this means implementing progress tracking first, then layering task graph constraints on top. The task graph can function without fine-grained progress (using binary step completion as a degenerate case where progress jumps from 0 to 1), but it works better with continuous progress because it can reason about partial completion -- "step 6 is 80% done, so don't start step 7 yet."

---

## Optimization 3: Temporal Binding Window for Audio-Visual Fusion

Source: Kazakos et al., "EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition" (ICCV 2019)

### The Core Insight

Audio and video of the same event are not temporally aligned. The sound of cracking an egg happens at a different frame than the visual of the egg breaking. An instructor's verbal correction comes 2-5 seconds after the technician's visual mistake. The paper calls this natural asynchrony and proposes fusing modalities within a Temporal Binding Window (TBW) -- a range of temporal offsets -- rather than requiring strict synchrony.

The paper trains a 32.64M parameter Temporal Binding Network (TBN) end-to-end on EPIC-Kitchens (39,596 labeled action segments), using BN-Inception backbones per modality. This trained architecture is not applicable to our API-based pipeline.

However, the temporal binding window concept itself is directly implementable as pipeline logic.

### What the Paper Shows

- Audio alone achieves comparable top-1 verb accuracy to RGB on EPIC-Kitchens (43.56% vs 45.68% on seen kitchens)
- Fusing all three modalities (RGB + Flow + Audio) with temporal binding consistently outperforms any single modality and late fusion
- Audio is especially informative for verbs with distinctive sounds: "wash", "spray", "turn-on", "close"
- Audio helps identify objects by their sound signature: "switch", "extractor fan", "foil"
- The model remains robust even when 14-46% of action segments contain irrelevant background audio

Key result: allowing temporal offsets during fusion (TBW) outperforms requiring synchronized inputs. The width of the binding window is set relative to action length, not a fixed duration.

### Our Angle: Sliding Audio Buffer

Instead of sending only the most recent audio transcript with each VLM call, send a sliding window of the last K audio chunks. When the VLM analyzes a frame at timestamp t, inject audio transcripts from a window around t (e.g., t-10s to t+5s), not just the single chunk closest to t.

This directly maps the temporal binding window concept to our prompt construction:

- **Current approach:** Each VLM call gets at most the single most recent audio transcript. If the instructor's correction came 3 seconds before the current frame, and the audio chunk boundary doesn't align, the transcript is already gone.
- **Improved approach:** Maintain a sliding buffer of the last K transcripts (e.g., K=3 at 5s chunks = 15s window). Inject all of them into the VLM prompt with their timestamps. The VLM sees: "Audio [35-40s]: 'no, not that one' / Audio [40-45s]: 'use the other wrench' / Audio [45-50s]: [silence]". This gives the model temporal context to correlate verbal corrections with the visual scene.

The window should be asymmetric -- more past audio than future, since we're processing online and future audio hasn't arrived yet. A reasonable starting point: 2-3 past chunks + the current chunk = 10-15s of audio context per VLM call.

This costs nothing extra in API calls (audio transcription already happens in `on_audio()`). The only change is accumulating transcripts in a buffer and including multiple of them in the VLM prompt instead of just one.

---

## Optimization 4: Multimodal Consensus Voting

Source: Chen et al., "SoundingActions: Learning How Actions Sound from Narrated Egocentric Videos" (CVPR 2024)

### The Core Insight

Not all modalities agree all the time. A visual action might be silent (lifting a plate). A sound might come from off-screen (TV playing). A narration might describe something that isn't currently audible. The paper proposes that the intersection of all modalities agreeing (what they call "Region I" in their Venn diagram) is the reliable signal, and everything else is noise.

The MC3 (Multimodal Contrastive-Consensus Coding) embedding trains modality encoders on 250K Ego4D clips using 8 A40 GPUs. The trained architecture is not applicable to our API-based pipeline.

However, the consensus mechanism -- requiring agreement across modalities before trusting a prediction -- is directly implementable as pipeline logic.

### What the Paper Shows

- The consensus score (Eq. 2) is the minimum pairwise similarity across all modality pairs. It's only high when ALL modalities agree. If any single modality disagrees, the score drops.
- Language (narrations) serves as an anchor modality that disambiguates audio-visual associations. Without language, audio-visual contrastive learning picks up incidental correlations (e.g., traffic noise correlating with street visuals) rather than action-specific correspondences.
- The two-stage training (align first, then refine with consensus) substantially outperforms joint training without consensus (AUC 0.598 vs 0.563 for audio-visual discovery, Table 2).
- Actions vary widely in how "sounding" they are: wash (90%), close (82%), cut (77%) are highly sounding; lift (19%), hold (9%) are mostly silent (Table 1). This maps directly to our audio being informative for some steps but not others.

### Our Angle: Three-Signal Voting

Our pipeline already has three "modalities" available at each decision point:

1. **Visual** -- the VLM's analysis of the current frame ("step 5 appears complete")
2. **Audio** -- the transcript from the audio model ("instructor says: good, move on")
3. **Procedure text** -- the step descriptions from the procedure JSON ("step 5: close the circuit breaker panel")

The consensus pattern: cross-reference these signals before emitting events, and adjust confidence based on agreement.

**High consensus (all agree) -- emit with high confidence:**
- VLM says "step 5 complete" + audio has instructor confirmation + procedure text says step 5 should follow step 4 (which is already done) → strong step_completion event

**Partial consensus (two agree) -- emit with medium confidence:**
- VLM says "step 5 complete" + procedure ordering checks out, but audio is silence → reasonable step_completion, no contradiction
- VLM says "error" + audio has instructor correction, but visual is ambiguous → likely error, audio reinforces

**No consensus (disagreement) -- downgrade or suppress:**
- VLM says "step 5 complete" + audio has instructor saying "no, that's wrong" → suppress step_completion, emit error instead
- VLM says "error" + audio is silence + procedure text says this action is expected → likely false positive, suppress

**The procedure text as anchor:** Following the paper's finding that language anchors disambiguate noisy audio-visual correlations, our procedure JSON serves the same role. The procedure text defines what SHOULD happen at each point. When the VLM or audio signal is ambiguous, the procedure text provides the expected baseline. This is particularly valuable because the procedure text is always available and noise-free, unlike audio (pitch-shifted, sometimes silent) or visual (VLM sometimes hallucinates).

This costs zero additional API calls. The signals are already being generated -- visual from `call_vlm()`, audio from `call_audio_llm()`, procedure from the loaded JSON. The consensus logic is purely Python-side: compare signals, compute agreement, adjust confidence and event emission accordingly.

---

## Optimization 5: Hand-Object Aware Prompting

Source: Zhang et al., "Helping Hands: An Object-Aware Ego-Centric Video Recognition Model" (ICCV 2023)

### The Core Insight

In egocentric video, hands and the objects they interact with are the primary carriers of action semantics. The paper trains an object-aware decoder that predicts hand positions, object positions, and semantic labels during training -- improving downstream performance across all benchmarks by ~1% even as a frozen representation. The trained decoder is not applicable to our API-based pipeline.

However, the insight directly informs how we write VLM prompts.

### The Problem with Generic Prompts

A prompt like "describe what the technician is doing" gives the VLM too much freedom. It might describe the background, the technician's posture, or the overall scene -- none of which are the action-carrying signals. In egocentric video, what matters is: what are the hands touching, and what state is the object in?

### Our Angle: Targeted Prompt Structure

Instead of asking the VLM broadly, ask specifically about hands and objects:

- "What are the technician's hands interacting with right now?"
- "What is the state of the object being manipulated?"
- "Has the object's state changed compared to the expected state for this step?"

This focuses the VLM's attention on the same features that the Helping Hands paper found most discriminative. The procedure JSON provides the expected object states per step (e.g., "circuit breaker panel should be closed"), so the VLM can compare current state against expected state.

This is a prompt-level change -- no additional API calls, no architecture changes. Just rewriting `prompts/vlm_prompt.txt` to ask about hands and objects rather than general scene description.

---

## Optimization 6: Instructor Speech as Expert Commentary

Source: Grauman et al., "Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives" (CVPR 2024)

### The Core Insight

Ego-Exo4D recruited 52 domain experts (coaches, teachers) to watch recorded videos of skilled activities and provide commentary -- critiquing technique, calling out strengths and weaknesses, explaining how specific behaviors affect performance. This "expert commentary" focuses on HOW the activity is executed rather than WHAT it entails, capturing subtle differences in skilled execution.

This is exactly what our instructor audio is: an expert watching a technician perform a procedure and providing real-time verbal corrections.

### The Mapping

| Ego-Exo4D | Our Pipeline |
|---|---|
| Expert commentary (coaches narrating critique) | Instructor audio (verbal corrections) |
| Proficiency ratings (skill scores) | Error detection (binary: correct/incorrect) |
| Keystep temporal segmentation | Step completion tracking |
| Procedural dependency graphs | Procedure JSON step ordering |

### Our Angle: Instructor Speech = High-Confidence Error Signal

The Ego-Exo4D framing reinforces what our audio benchmark already showed empirically: instructor speech correlates strongly with errors. In error-heavy videos (z065: 21 errors), there is significantly more instructor speech than in clean-execution videos (R073: 0 errors).

This means:

- **Instructor speech detected** → high probability an error or correction is happening. Treat as a strong error signal, not just supplementary context.
- **Silence / no speech** → technician is likely on track (or idle).
- **Instructor tone/content matters** → "good" or "okay" is confirmation (reinforces step completion). "No", "stop", "wrong" is correction (triggers error detection). "Wait" or silence after a question is ambiguity (hold current state).

This upgrades instructor audio from "nice to have context" to "primary error detection modality" in our consensus voting (Optimization 4). When the audio transcript contains correction language, it should override visual signals that suggest normal progress.

---

## Optimization 7: Mother Agent (Observer + Judge Orchestration)

### The Problem

Everything above — task graph, progress, temporal audio, consensus voting, hand-object prompting, instructor speech — asks the VLM to do more work per frame. But the VLM has two fundamental limitations that no amount of prompt engineering can overcome:

1. **No memory between calls.** Each VLM call is stateless. It sees one frame and a few lines of recent audio. It cannot reason about the trajectory of an action ("the student has been reaching toward the breaker for the last 8 seconds and is now pulling it out — step 1 is completing").
2. **One shot at the decision.** The VLM must produce status/step_id/confidence from a single frame. There is no opportunity to revise, consolidate across frames, or weigh evidence from before and after.

The result, visible in our baseline R066 run: the VLM oscillates between "step_complete" and "error" on adjacent frames, hallucinates errors when it hasn't seen earlier steps, and flags blurry frames as "idle" because it has no way to say "I'm not sure — let me see the next few frames first."

### The Fix: Split Observer and Judge

Rather than asking one agent to observe AND decide, we split the responsibilities across two agents:

1. **Observer (VLM)**: Describes what's visible. No decisions. Outputs `{hands, objects, action, visual_cues, confidence}` per frame. Stateless, but that's fine because it's not making decisions.
2. **Judge (Mother Agent)**: A stronger reasoning LLM (e.g., `gpt-5.4` with `reasoning_effort="low"`). Reads observer outputs, audio transcripts, procedure JSON, and the state of previously-detected events. Makes the actual event decisions. Has full temporal context.

```
Frame ──▶ Observer (VLM) ──▶ observation buffer
                                      │
                                      ▼
                   ┌───────────────────────────────┐
                   │ Audio transcripts + procedure │
                   │ + observation buffer          │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                          Judge (Mother)
                                  │
                                  ▼
                         events.json (final)
```

### Two Implementation Variants

**V1: Batch-at-End (simpler, accuracy-first)**
- Stream runs as normal. Observer buffers observations, audio is pre-computed (as today).
- After `harness.run()` finishes, Mother fires ONCE with the full observation log + audio log + procedure.
- Mother returns the complete events list.
- Detection latency is effectively video_duration for all events → `latency_score ≈ 0`.
- Accepts the 20% latency weight sacrifice for expected 3-5× accuracy gain.

**V2: Real-Time (harder, latency-aware)**
- Mother fires periodically during the stream (e.g., every 10-30s of video).
- Each fire reasons over the observations and audio accumulated in that window.
- Events are emitted shortly after they happen → `latency_score` stays on the table.
- More complex: buffer management, state synchronization across fires, deciding when to fire.

### The Philosophical Line

V1 is a Type 2 offline optimization (true lookahead reasoning — the mother uses observations from the entire video to decide events that happened early). It crosses a line the pipeline didn't cross before — pre-computed audio (already in use) is Type 1 (temporal shifting, replaceable by streaming Whisper), but V1 mother is not replaceable in production. A live technician can't wait for the video to end.

**We're taking V1 anyway as the starting point**, for three reasons:
1. The scoring formula weights F1 accuracy (0.80) 4× over latency (0.20). Maximizing F1 even at `latency_score=0` is likely a net gain.
2. The project's evaluation phase is itself batch (evaluator and dashboard read `events.json` as a complete file). V1 is consistent with how scoring actually happens.
3. V1 establishes a baseline for what full-context reasoning can achieve. V2 optimizations (periodic batch, streaming) can then be measured against that ceiling.

### Our Angle

With `gpt-5.4 + reasoning_effort="low"`, the mother has the reasoning strength to:

- **Enforce task graph** (Optimization 1) automatically — it reads the procedure and won't emit step 7 if steps 3-6 haven't been seen.
- **Track progress across observations** (Optimization 2) — it sees "hands reaching toward breaker" over 3 frames and concludes step 1 is completing at frame 3, not frame 1.
- **Run consensus voting** (Optimization 4) — visual + audio + procedure are all in its context window.
- **Use instructor speech as error signal** (Optimization 6) — negative language in audio → emit error, even if visual is ambiguous.

The mother becomes the vehicle for actually executing Optimizations 1, 2, 4, and 6, rather than asking the VLM to do them all in one shot per frame.

### Cost

- One OpenAI API call per pipeline run (post-stream). Roughly 8-10K tokens in, 1-2K tokens out for a 3-minute video.
- Observer VLM cost unchanged — same number of VLM calls, just describing instead of deciding.
- Net: small cost increase, large accuracy upside.

### V1 Status: SHIPPED (2026-04-15)

First V1 run benchmarked on R066-15July-Circuit-Breaker-part2 (176s, 11 steps, 6 errors, 5 idle periods):

| Run             | step_f1 | error_f1 | combined | × baseline |
| --------------- | ------- | -------- | -------- | ---------- |
| Baseline        | 0.167   | 0.000    | 0.067    | 1.00×      |
| V1 @ 10x        | 0.444   | 0.250    | 0.278    | **4.15×**  |
| V1 @ 1x         | 0.476   | 0.000    | 0.190    | **2.84×**  |

Headline findings:
- Mother V1 dramatically improves step detection (2.66–2.85× F1) without any model retraining.
- Latency-score saturates at 0 as expected (Option G1 stamps honest `video_end - timestamp_sec`).
- Known failure modes: step-number confusion in dense early regions, zero `idle_detected`, counterintuitive 10x>1x error_f1.

Full run report + event-by-event GT comparison + failure analysis + V2 next steps:
**[`docs/mother_agent/README.md`](mother_agent/README.md)**

---

## Key Takeaways for Our Pipeline

- **Task graph is the highest-priority optimization.** Our procedure JSONs already provide a manual task graph for free. Implementing predecessor/successor checks in the step-tracking logic requires no VLM changes, no additional API calls, and no model retraining. It directly prevents the kind of wrong-ordering errors we saw in baseline runs.

- **Progress prediction approximates cheaply via prompting.** We can't train a GRU module, but we can ask the VLM "how far through this step does the scene appear?" and track the response over time. The value comes from the continuous signal itself, not the specific architecture used to produce it.

- **Progress and task graph are coupled, not independent.** Progress provides the completion state signal that the task graph consumes. Implementing both is more effective than either alone (Tables 1-2 show consistent improvement from CAS+APP to CAS+APP+TG).

- **Not all steps benefit equally from progress tracking.** Steps with visible physical transformation benefit most. Steps that are instantaneous or visually ambiguous throughout their duration should rely on the task graph for ordering rather than progress for timing.

- **Transcript-based task graphs are nearly as good as manual ones.** Table 3 shows only a slight gap between automatically derived and manually built graphs. This suggests that even imperfect procedure JSONs (e.g., with optional steps or unclear ordering) will still provide value as task graph constraints.

- **Sliding audio buffer is free to implement.** Send the last K audio transcripts (not just the most recent) with each VLM call. This applies the temporal binding window concept without any model training -- just prompt engineering. Costs zero additional API calls since audio transcription already runs in `on_audio()`.

- **Multimodal consensus voting gates event emission.** Cross-reference visual, audio, and procedure text signals before emitting events. When all three agree, high confidence. When audio contradicts visual (e.g., instructor says "no" but VLM says "step complete"), suppress the step and emit an error instead. Procedure text anchors the consensus as the noise-free baseline. Zero additional API cost -- just compare existing signals in Python.

- **Ask about hands and objects, not general scenes.** Rewrite VLM prompts to target the action-carrying features: what are the hands interacting with, what state is the object in, has it changed? Generic "describe the scene" prompts waste VLM attention on non-discriminative features.

- **Treat instructor speech as the primary error signal.** Instructor audio is expert commentary -- speech correlates strongly with errors, silence correlates with correct execution. Instructor corrections should override visual signals in the consensus voting.

- **Split observer and judge responsibilities.** A single VLM doing both observation and decision-making is the root cause of our baseline's oscillation, hallucinations, and idle spam. Splitting into an observer VLM (describes frames) and a judge LLM (reasons over observations + audio + procedure) gives us temporal context and full reasoning that single-frame VLM calls cannot provide. V1 fires the judge once at end of stream (accuracy-first); V2 will fire it periodically for latency-sensitive deployments.
