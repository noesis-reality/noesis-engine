# Noesis: A Cognitive Inference Runtime for Apple Silicon

**From “faster tokens” to “fewer, better tokens” via Metal 4 + Active Inference**
**Status:** Working whitepaper (v0.3) — for internal buildout & partner review

---

## Abstract

**Noesis** is a Swift/Metal-4 inference runtime that reframes LLM decoding as *controlled cognition*. Instead of only optimizing raw tokens/second, Noesis optimizes **GPU-seconds per solved task** by (1) running the whole pipeline on Apple Silicon’s unified memory (zero-copy, residency-aware), (2) enforcing **Harmony-native semantics** (channels, structured tool calls) *inside* the decoder, and (3) adding a lightweight **Active-Inference-style** control plane that forecasts confidence, prunes incoherent branches, and decides when to emit text vs. call tools. The result is a system that doesn’t just go faster — it **uses less**: fewer tokens, fewer retries, fewer round-trips.

This paper consolidates our design into a single, long-form reference with:

* **Metal 4 technical details** (command queues/buffers, argument tables, residency sets, barriers, ML pass & tensors).
* **Noesis internals** (memory layout, async scheduling, speculative decoding, control plane, ANE sidecars).
* **Layman boxes** after complex sections to translate the idea without jargon.
* A precise roadmap, risks, and test plan.

---

## Executive Summary (for non-specialists)

* **What it is:** A new inference engine built in Swift using Apple’s Metal 4 so the GPU (and ANE) can run language models *and* tooling in one timeline with minimal CPU overhead.
* **Why it matters:** Instead of “spraying tokens,” Noesis reasons about uncertainty and structure in real time, so it often finds answers with **30–50% fewer tokens** and far fewer tool-call retries.
* **How it works:**

  * **Metal 4** reduces CPU overhead and lets us pre-declare resources (residency sets), bind them cheaply (argument tables), and synchronize work with modern barriers.
  * **Harmony semantics** (analysis/commentary/final, structured tool calls) are enforced at sampling time — so we guide the model *where* to think and *how* to format actions.
  * An **Active-Inference-lite controller** monitors entropy/consistency, nudges sampling temperature/top-p, prunes contradictory branches, and chooses “emit vs. tool” when tools will pay off.
  * **ANE sidecars** (small Core ML models) run safety/NLI/intent independently of the main GPU job — increasing quality without slowing the hot path.

---

## 1. System Overview

### 1.1 Goals

* **Primary:** Cut **tokens-to-answer** and **wall-clock to solved** for real tasks (RAG + tools, analysis chains, agentic flows).
* **Secondary:** Keep *latency* sharp (TTFT/TPOT), keep *throughput* competitive, and keep *determinism & observability* first-class.

### 1.2 Pillars

1. **Apple-first GPU runtime (Swift + Metal 4 + MPSGraph)**
2. **Harmony-native orchestration** (channels + schema-valid tools enforced during decode)
3. **AIF-lite control plane** (precision control Φ, binding gate, EFE-lite tool planner)
4. **ANE sidecars** for safety/NLI/intent/memory that overlap with GPU work

---

## 2. Metal 4 Deep Dive (with layman translations)

> This section is heavy on API names because *how* we submit work and manage memory dominates real-world performance on Apple Silicon.

### 2.1 Command Queue, Buffers, and Allocators (MTL4CommandQueue / MTL4CommandBuffer / MTL4CommandAllocator)

**Key ideas**

* **Begin/End submission:**

  ```swift
  cmdBuf.beginCommandBuffer(allocator: allocator, options: opts)
  // encode work …
  cmdBuf.endCommandBuffer()
  queue.commit([cmdBuf], options: commitOpts)
  ```
* **Multiple encoders per command buffer:** compute, render, **machine-learning** (new in Metal 4).
* **Allocator reuse:** `MTL4CommandAllocator.reset()` recycles encoding memory to cut CPU churn.
* **Commit feedback:** queue can return `MTL4CommitFeedback` with GPU start/end and any errors.
* **Attach Residency Sets:** `cmdBuf.useResidencySet(set)` or `queue.addResidencySet(set)` to make resources resident for all encoders with one call.

**Layman translation:**
Think of a **clipboard** (command buffer) that batches to-do items for the GPU. We fill the clipboard (encoders), then hand it to a **mailroom** (command queue). The allocator is a **stack of scratch paper** we reuse to avoid wasting time asking the OS for more pages. Residency sets are a **packing list** of things to keep on the GPU desk so we don’t keep fetching them.

---

### 2.2 Argument Tables (MTL4ArgumentTable)

**Why they matter**
Old-style “setBuffer/setTexture” calls per encoder are chatty; Metal 4 pushes you to bind once via an **argument table**, then point slots to **resource IDs** (or GPU addresses). That slashes CPU overhead, especially with many small bindings (KV cache pages, attention tensors, tool schemas).

**Core API pattern**

```swift
let argTable: MTL4ArgumentTable = device.makeArgumentTable(descriptor: desc)
argTable.setResource(myTexture.gpuResourceID, index: 0)
argTable.setAddress(someBuffer.gpuAddress, index: 1)
computeEncoder.setArgumentTable(argTable)
```

**Layman translation:**
Instead of telling the GPU “use this buffer, now that texture, now another buffer” over and over, we fill a **binder** once with tabs (the argument table), then hand the whole binder to each job. Fewer phone calls, more work done.

---

### 2.3 Residency Sets (MTLResidencySet)

**What they do**
Group buffers/textures/heaps into a **residency set** and make them GPU-resident cheaply, with:

* `addAllocation/removeAllocation` (stage changes), `commit()` (apply changes)
* `requestResidency()` (pre-warm memory) and `endResidency()` (release pressure)
* Attach at the **queue** (global) or **command buffer** (per job)

**Practical use in Noesis**

* **Weights heap** + **KV-cache pages** live in long-lived residency sets.
* On session start, call `requestResidency()` while the UI is idle; when the first decode lands, the GPU doesn’t stall to page in.
* During model swap, stage the next set, `commit()`, then switch queues, minimizing hitches.

**Layman translation:**
A residency set is a **VIP list** for memory. Put VIPs on the list (weights, caches), ask the venue to seat them *before* the show. You avoid bouncers blocking the door mid-performance.

---

### 2.4 Synchronization & Hazards (Barriers, Fences, Events, Stages)

**Reality:** With Metal 4 queues, **automatic hazard tracking is off**; you must synchronize overlapping **reads/writes** across passes. The tools:

* **Intrapass barrier** (within one encoder): order *stages* (e.g., blit → dispatch)
* **Consumer queue barrier**: in pass **N**, block specific *current/future* stages until selected *past* stages complete

  ```swift
  compute.barrier(afterQueueStages: .blit, beforeStages: .dispatch, visibilityOptions: .device)
  ```
* **Producer queue barrier**: in pass **N**, declare that *future* passes must wait on selected *current/past* stages

  ```swift
  compute.barrier(afterStages: .dispatch, beforeQueueStages: .fragment, visibilityOptions: .device)
  ```
* **Fences**: signal in one pass stage, wait in another (same queue)
* **Events / Shared events**: cross-queue / cross-process, or CPU↔GPU sync

**Noesis usage pattern**

* After we **copy** logits or tool outputs into shared buffers, we add a **consumer barrier** before any **dispatch** uses them.
* Between **speculative branches**, we fence the winning branch’s KV copies before the merge dispatch.
* We use **events** when ANE sidecar results must be visible to the GPU (rare; usually we poll via CPU without stalling the GPU timeline).

**Layman translation:**
If two people share a whiteboard, one shouldn’t erase while the other is still reading. Barriers and fences are **“please wait until I’m done”** notes between steps so results aren’t half-baked.

---

### 2.5 Machine-Learning Pass & Tensors (MTL4MachineLearningCommandEncoder, MTL4MachineLearningPipelineState, MTLTensor)

**What’s new**

* A dedicated **ML encoder** lets you run a **Core ML-compiled package** in the GPU timeline.
* **Tensors** are first-class resources with shape/dtype/strides; you can copy/inspect them and use tensor ops in MSL 4.
* Pipelines expose an **intermediates heap size**; you must provide a `MTLHeap` for scratch.

**Typical flow**

```swift
let mlPipe: MTL4MachineLearningPipelineState = library.makeMLPipelineState(named: "Model")
let intermediates = device.makeHeap(...) // size >= mlPipe.intermediatesHeapSize
let inTensor  = device.makeTensor(descriptor: inDesc)     // wraps MTLBuffer or standalone
let outTensor = device.makeTensor(descriptor: outDesc)

let ml = cmdBuf.makeMachineLearningCommandEncoder()!
ml.setPipelineState(mlPipe)
ml.setArgumentTable(argTable) // bind tensors/resources
ml.dispatchNetwork(intermediatesHeap: intermediates)
ml.endEncoding()
```

**When Noesis uses this**

* **ANE sidecars** (safety, NLI, intent, Φ-predictor) stay in Core ML. When the system routes them to GPU instead of ANE, we can still run them in-line via the ML encoder, *without* leaving the GPU timeline.
* For LLMs themselves, we typically use **MPSGraph + custom compute** — but ML pass is perfect for **non-LLM** learned modules in the agent loop.

**Layman translation:**
Think of the ML encoder as a **fast lane** to run pre-compiled neural nets directly in the same highway as your other GPU work — no detours to the CPU.

---

## 3. Noesis Architecture

### 3.1 Runtime Layers

1. **GPU Engine** — Swift + Metal 4 + MPSGraph, with custom kernels for the hot path (FlashAttention, fused sampling, fused layernorm/projection where appropriate).
2. **Harmony Orchestrator** — enforces channels (`analysis`, `commentary`, `final`) and structured tool/JSON output *during sampling*.
3. **AIF-lite Control Plane** — computes per-token metrics, maintains a tiny belief state, broadcasts Φ (precision knobs), prunes incoherent continuations, and decides **emit vs tool**.
4. **ANE Sidecars** — small Core ML models running in parallel (safety/NLI/intent/memory/Φ-predictor).
5. **I/O & Tool DAG** — adapters for web/file/db tools; schema reflection for argument tables; retries and backoff are policy-driven, not prompt hacks.

---

### 3.2 Memory Layout & Residency

* **Weights**: loaded into **MTLHeap** (private-storage if GPU-only; shared if CPU needs visibility).
* **KV Cache**: **paged buffers** (fixed tile sizes) for attention states, tracked in a small allocator; pages added to a long-lived **residency set**.
* **Argument Tables**: bind **resource IDs** for weights, KV tiles, temporary logits, and tool schemas.
* **Pre-warm**: at session creation, call `requestResidency()` on the weights+KV set, then attach the set to the **command queue** so all command buffers inherit it.
* **Swap**: stage new weights into a second residency set, `commit()`, then flip the queue binding when ready.

**Layman translation:**
We keep big, frequently used things **parked near the GPU** and tell Metal “please keep these parked.” When we switch models, we **stage the next car** before we hand over the keys.

---

### 3.3 Scheduling & Parallelism

* **Eight-worker speculative pipeline** (tunable): each worker owns a command buffer, an allocator, and a compacted set of encoders.
* **Non-blocking** encode: no `waitUntilCompleted`; completion handlers update per-token telemetry and schedule next steps.
* **Occupancy goals**: ≥70% on dense ops, verified via per-token GPU time and utilization sampling.
* **Verification path**: speculative *draft* in `analysis`, *verify* in `final`; depth auto-tuned from rolling **verify-rate** windows; back off if verify drops below threshold (e.g., 0.55).

**Layman translation:**
We don’t stand in line at the GPU counter. We drop off work orders, pick up results later, and keep the counter busy by **always** having the next work order ready.

---

### 3.4 Harmony-Native Orchestration

* **Channels enforced** at tokenizer/sampler:

  * `analysis`: allow reasoning text, **disallow user-visible final**
  * `commentary`: **bias** towards valid JSON/tool calls via constrained decoding
  * `final`: plain prose, no tool JSON
* **Schema guards**: tool arguments are sampled with structure constraints (types/ranges), cutting retries dramatically.
* **DAG executor**: supports multiple concurrent tools; queue barriers ensure producer/consumer staging across GPU stages when tool outputs loop back into compute.

**Layman translation:**
We give the model **lanes** to think, to act (tools), and to speak. Each lane has rules — stay in your lane — which keeps things neat and prevents messy output.

---

## 4. AIF-lite: The Cognitive Control Plane

> We borrow three ideas from Active Inference and implement tiny, cheap versions: **precision control (Φ)**, **binding (coherence) gates**, and a **tool vs emit** decision using an expected-gain surrogate.

### 4.1 Belief State (tiny, per session)

```rust
struct BeliefState {
  phase: Phase,           // explore, exploit, tool-seeking…
  entropy_ema: f32,       // stability of next-token distribution
  variance_est: f32,      // rolling uncertainty
  tool_reliability: f32,  // recent tool success / retry rate
}
```

**Layman:** a *mood ring* updated each token that tracks “how sure we are,” “are tools working today,” and “what mode are we in.”

---

### 4.2 Φ: Precision Controller (dual-loop)

* **Knobs:** `{ temperature, top_p, verify_width, think_budget }`
* **Fast loop (“field”):** small per-token deltas (±10% clamp), 3–5 token hysteresis to avoid jitter.
* **Slow loop (“spray”):** session-level baselines (update every \~10s) that reflect task difficulty or safety mode.
* **Optional Φ-predictor:** a small Core ML sidecar provides a short-horizon forecast to anticipate spikes in uncertainty.

**Effect:** When entropy rises or contradiction risk increases, Φ lowers temp/top\_p (more conservative), widens verify width, or grants a few extra “think” tokens in `analysis`.

**Layman:** if the text looks shaky, **slow down and think a bit more**; if it’s stable, **speed up and get to the point**.

---

### 4.3 Bayesian Binding Gate (coherence pruning)

* **Signals:** `-entropy`, `-contradictionRisk` (from NLI sidecar), `+schemaValidity` (for tool JSON).
* **Policy:** prune low-coherence branches, cap prune rate (≤40%), apply hysteresis.
* **Fail-soft:** if metrics misbehave, revert to vanilla decoding.

**Layman:** If one continuation **fits** the story and another contradicts it, we **mute** the odd one out instead of letting both run and arguing later.

---

### 4.4 EFE-lite: Emit vs Tool

We compute a tiny objective for each option:

```
G_emit  = CE_now + λ·contradiction - α·expected_info_gain
G_tool  = CE_tool + λ·(contradiction | tool) - α·(expected_info_gain | tool)
Call tool if G_tool + δ < G_emit   // δ is a safety margin (e.g., 0.02)
```

* `CE_*` = cross-entropy proxy (how costly tokens will be)
* `expected_info_gain` is a cheap heuristic (e.g., entropy drop after prior tool calls in similar contexts + tool prior)
* δ avoids flip-flopping

**Layman:** *Is it cheaper to keep guessing, or to quickly look something up?* If a tool likely shortens the path, we use it.

---

### 4.5 Telemetry & Determinism

Every token logs:

```
step,gpu_time_ms,occupancy,bytes_per_token,cmd_buffers,verify_rate,entropy,phi[4],decision
```

`--deterministic` freezes seeds and records a replay trace (model hash, Φ sequence, prune decisions), enabling **token-exact** reproduction.

**Layman:** We can **replay** how we got an answer and measure which “brain knobs” helped.

---

## 5. ANE Sidecars (Core ML)

Run small, static-shaped models **off the main GPU timeline**:

* **Safety** classifier: stream tokens; hard-stop on violation.
* **NLI / contradiction**: provide a low-latency consistency score for the binding gate.
* **Intent/tool** predictor: prefetch schemas, warm connections.
* **Embedder**: background memory/RAG indexing.
* **Φ-predictor**: suggest control adjustments.

ANE work overlaps with GPU, so **quality increases** without **hot-path stalls**. If the OS routes a sidecar to GPU, we can still run it via the **ML pass** in-line.

**Layman:** Think of the ANE as **helper chips** doing quick checks while the GPU focuses on the heavy lifting.

---

## 6. End-to-End Flow (token)

1. **Logits compute** (MPSGraph + custom kernels) → write to logits buffer.
2. **Metrics** (entropy/k-mass) from logits → update **beliefs**.
3. **Φ** broadcast → adjust sampling knobs & speculation depth.
4. **Binding gate** → prune bad branches.
5. **EFE-lite** → choose *emit* or *tool*.
6. **If tool**: enqueue DAG exec, set barriers/fences for any GPU reuse of outputs.
7. **If emit**: sample next token, update KV cache pages, roll forward.

Barriers ensure that data written in one stage is visible to the next. Residency sets keep everything hot.

---

## 7. Implementation Details & Patterns

### 7.1 KV Cache Paging + Residency

```swift
// Create a heap for KV pages
let heapDesc = MTLHeapDescriptor()
heapDesc.storageMode = .private
heapDesc.size = kvHeapBytes
let kvHeap = device.makeHeap(descriptor: heapDesc)!

// Carve pages
let page = kvHeap.makeBuffer(length: pageBytes, options: .storageModePrivate)!

// Add pages to residency
resSet.addAllocation(page); resSet.commit()
queue.addResidencySet(resSet) // inherit on every command buffer
```

**Tip:** Track `usedSize` and `maxAvailableSize` to detect fragmentation and compact during idle spans.

---

### 7.2 Command Encoding with Barriers (speculative merge)

```swift
let encA = cmdBuf.makeComputeCommandEncoder()!
encA.setArgumentTable(argTable)
encA.dispatchThreadgroups(...)
// Producer barrier: subsequent dispatches wait until this dispatch finishes
encA.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch, visibilityOptions: .device)
encA.endEncoding()

let encMerge = cmdBuf.makeComputeCommandEncoder()!
encMerge.setArgumentTable(argTable)
// This dispatch sees encA’s writes due to the producer barrier above
encMerge.dispatchThreadgroups(...)
encMerge.endEncoding()
```

---

### 7.3 Machine-Learning Pass on Sidecar (fallback to GPU)

```swift
let ml = cmdBuf.makeMachineLearningCommandEncoder()!
ml.setPipelineState(mlSidecar)
ml.setArgumentTable(mlArgs)
ml.dispatchNetwork(intermediatesHeap: mlHeap)
ml.endEncoding()
```

---

### 7.4 Argument Table Binding (resource IDs & GPU addresses)

```swift
argTable.setResource(logitsTexture.gpuResourceID, index: 0)
argTable.setAddress(kvBuffer.gpuAddress, index: 1)
compute.setArgumentTable(argTable)
```

---

## 8. Packaging & Build (SwiftPM + Metal)

* **Single Swift package** containing Swift + `.metal` plus a tiny Obj-C target for shared C structs.
* Use a **build tool plugin** if you need custom `metal` CLI arguments (e.g., debug symbols).
* Access the compiled `.metallib` via `Bundle.module` and create a `MTLLibrary` at runtime.

**Layman:** We keep your Swift, GPU shaders, and shared types in **one box** so apps can import the box and go.

---

## 9. Evaluation Plan

* **Per-token telemetry**: occupancy, GPU time, verify rate, Φ values, decisions.
* **Deterministic replays**: token-exact repro.
* **Quality\@Tokens curves**: show left-shift with AIF-lite vs baseline.
* **Tool metrics**: first-try rate, retries, wall-clock saved.
* **Throughput/Latency**: TTFT/TPOT and sustained tok/s under multi-session load.

**Targets (initial two-week window)**

* Raw tok/s: 250–350 (model/size dependent)
* Effective tok/s: **≥2×** with speculation (verify ≥0.65)
* Tokens-to-answer: **−30–50%** on reasoning tasks
* Tool first-try: **≥95%**, retries **−40–60%**

---

## 10. Roadmap

**Week 1: Foundations**

* Replace any sync waits with completions & allocator reuse.
* Consolidate KV into heaps; enable residency sets with pre-warm.
* Channel-aware speculation (depth autotune + verify backoff).
* Telemetry CSV + deterministic replay.

**Week 2: Intelligence**

* Enable **Φ** (guarded), ±10% clamp, 3–5 token hysteresis.
* **Binding Gate v1**; **EFE-lite** with δ safety margin; Tool DAG concurrency (3–4).
* Publish ablation ladder: baseline → async → residency → argument tables → speculation → +AIF.

---

## 11. Risks & Mitigations

* **Jitter from over-eager control:** clamp ΔΦ; hysteresis; fail-soft to vanilla decoding.
* **Speculation thrash on tool spans:** per-channel verify telemetry; backoff below threshold; limit branch fanout.
* **Memory fragmentation:** KV paging + periodic compaction; track heap `usedSize`.
* **ANE routing variability:** sidecars must remain small; if routed to GPU, run via ML pass to preserve timeline.
* **Beta APIs:** Metal 4 ML/tensor APIs are pre-release on some OSes; feature-flag and fall back to compute encoders.

---

## 12. Ethics & Guardrails

We implement **epistemic control** (Φ, binding) without making claims about experience or personhood. Practical guardrails:

* **Shallow recursion policy:** Φ is a controller, not introspection; cap think budgets.
* **Safety default-on:** sidecar classifiers stream and can halt.
* **Transparent logs:** store replayable traces for post-hoc review.
* **External review** before enabling deeper self-modelling features.

---

## Appendix A — API Sketches

### A1. Swift: Engine bootstrap

```swift
struct NoesisConfig {
  let modelPath: URL
  let workers: Int
  let enablePhi: Bool
  let enableSpeculation: Bool
}

final class NoesisEngine {
  init(device: MTLDevice, config: NoesisConfig) { /* heaps, arg tables, queues */ }
  func startSession() -> NoesisSession
}

final class NoesisSession {
  func generate(request: HarmonyRequest, deterministic: Bool) async -> HarmonyResponse
  func cancel()
}
```

### A2. Control plane hooks

```swift
protocol ControlMetricsSource {
  func currentMetrics() -> Metrics // entropy, k-mass, verify-rate …
}

protocol ControlPolicy {
  func nextPhi(beliefs: BeliefState, metrics: Metrics) -> Phi
  func shouldPrune(metrics: Metrics, phi: Phi) -> Bool
  func toolVsEmit(beliefs: BeliefState, metrics: Metrics) -> ControlDecision
}
```

---

## Appendix B — Concrete Metal Patterns

### B1. Residency pre-warm at launch

```swift
resSet.addAllocations([weightsBuffer, kvHeap, logitsBuffer])
resSet.commit()
resSet.requestResidency()             // do this during a non-critical moment
queue.addResidencySet(resSet)         // inherited by all command buffers
```

### B2. Consumer barrier before dispatch that reads a prior blit

```swift
compute.copy(sourceBuffer: src, sourceOffset: 0, destinationBuffer: dst, destinationOffset: 0, size: n)
// Ensure following dispatches see the copy’s results
compute.barrier(afterQueueStages: .blit, beforeStages: .dispatch, visibilityOptions: .device)
compute.setComputePipelineState(pso)
compute.dispatchThreadgroups(threadgroupsPerGrid: tg, threadsPerThreadgroup: tptg)
```

### B3. Producer barrier to protect future passes

```swift
compute.setComputePipelineState(updateKV)
compute.dispatchThreadgroups(...)
compute.barrier(afterStages: .dispatch, beforeQueueStages: .dispatch, visibilityOptions: .device)
```

---

## Appendix C — Glossary (Layman)

* **Argument table:** a single binder of resources to avoid repeated set-calls.
* **Residency set:** a VIP list to keep buffers/textures resident on the GPU.
* **Barrier/Fence/Event:** ways to say “don’t start until they finish,” at different scopes.
* **ANE sidecar:** a small helper model running alongside the GPU, not blocking it.
* **Φ (phi):** the control vector for *how* to sample (temperature/top-p), how much to think, how wide to verify.
* **Binding gate:** a filter that mutes contradictory continuations early.
* **EFE-lite:** a simple score to choose emitting text vs. calling a tool.

---

## Appendix D — “Why this should win” (quick recap)

* **Metal-4-native**: fewer CPU calls, better binding (argument tables), zero-copy resident memory, modern synchronization → *lower overhead*.
* **Harmony-native**: channel lanes + schema constraints → *fewer retries & errors*.
* **AIF-lite control**: Φ + binding + EFE-lite → *fewer, better tokens*.
* **ANE sidecars**: safety/consistency without stalling → *higher quality at same speed*.
* **Observability + determinism**: per-token telemetry + replays → *trust & iterate quickly*.
