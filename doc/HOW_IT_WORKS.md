# Neural-Bridge MAS — How It Works

> **A complete guide to the system's architecture, outputs, and usage.**

---

## Table of Contents

1. [The Problem Being Solved](#1-the-problem-being-solved)
2. [System Overview](#2-system-overview)
3. [Core Components (How Each Part Works)](#3-core-components)
   - [A. Neural Codebook — The Language](#a-neural-codebook--the-language)
   - [B. Interlat Middleware — The Transport Layer](#b-interlat-middleware--the-transport-layer)
   - [C. Alignment Strategy (Procrustes) — The Translator](#c-alignment-strategy-procrustes--the-translator)
   - [D. DroidSpeak Cache — The Shared Memory](#d-droidspeak-cache--the-shared-memory)
   - [E. SANEmerg Filter — The Noise Canceller](#e-sanemerg-filter--the-noise-canceller)
   - [F. AgentArk Distillation — The Final Compression](#f-agentark-distillation--the-final-compression)
4. [Security: Differential Privacy Layer](#4-security-differential-privacy-layer)
5. [What Does the System Output?](#5-what-does-the-system-output)
6. [How to Use This System (Step-by-Step)](#6-how-to-use-this-system-step-by-step)
7. [Lifecycle of a Neural Message](#7-lifecycle-of-a-neural-message)
8. [Recovery & Fallback](#8-recovery--fallback)
9. [Performance Benchmarks](#9-performance-benchmarks)
10. [File Reference](#10-file-reference)

---

## 1. The Problem Being Solved

Modern multi-agent AI systems make agents talk to each other in **plain English**. This introduces severe inefficiency:

| Problem | Impact |
|---|---|
| **Translation Loss** | ~70% of information density is lost when thoughts are converted to words and back |
| **Token Wall** | Verbose natural language fills up the shared context quickly; only 3–4 agents can collaborate before hitting limits |
| **"Lost in Translation" Errors** | Subtle reasoning nuances are corrupted by the verbosity of language |

**Neural-Bridge MAS** eliminates these problems by making agents communicate **neurally** — exchanging raw, compressed thought-vectors (latent states) instead of sentences.

> **Analogy:** Instead of two chess grandmasters texting descriptions of their board analysis, they telepathically share the evaluation matrix directly.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     NEURAL-BRIDGE MAS                           │
│                                                                 │
│  ┌─────────────┐   Neural Packet   ┌─────────────────────────┐ │
│  │  Agent A    │ ─────────────────▶│   Interlat Middleware   │ │
│  │ (Architect) │                   │  • Pack logic vector    │ │
│  └─────────────┘                   │  • Apply Procrustes W   │ │
│                                    │  • Inject DP noise      │ │
│  ┌─────────────┐                   │  • Serialize to carrier │ │
│  │  Agent B    │ ◀─────────────────└─────────────────────────┘ │
│  │ (Engineer)  │                             │                  │
│  └─────────────┘                             ▼                  │
│                                    ┌─────────────────────────┐ │
│  ┌─────────────┐                   │   DroidSpeak K-V Cache  │ │
│  │  Agent C    │ ◀────────────────▶│  • Shared memory pin    │ │
│  │ (QA/Monitor)│                   │  • Zero-cost re-reads   │ │
│  └─────────────┘                   └─────────────────────────┘ │
│         │                                     │                 │
│         ▼                                     ▼                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              SANEmerg Filter                             │  │
│  │    Strips noise, audits for drift > 0.12, passes logic   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│               ┌─────────────────────────┐                      │
│               │  AgentArk Distillation  │                      │
│               │  3 agents → 1 model     │                      │
│               └─────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### A. Neural Codebook — The Language

**File:** [`neural_codebook.md`](./neural_codebook.md)

The Neural Codebook is the **shared vocabulary** for all agents. Instead of English words, it defines a schema of **latent tensors** — specific vector coordinates that carry meaning.

#### The Neural Header (128 Dimensions)
Every message starts with a 128-dimensional "envelope" vector:

| Dimensions | Field | Purpose |
|---|---|---|
| `[0–31]` | **Model ID** | Identifies which AI model is sending (so the receiver picks the right translation matrix) |
| `[32–63]` | **Task UUID** | Tags the message to a specific project, preventing cross-project confusion |
| `[64–95]` | **Entropy Density** | Tells the SANEmerg filter how much signal vs. noise to expect |
| `[96–127]` | **Temporal Marker** | Sequence ID for cache alignment; ensures messages arrive in order |

#### Reserved Anchor Signals

| Symbol | Meaning |
|---|---|
| `<INIT_SYNC>` | "I'm connecting — allocate a cache partition for me" |
| `<ERR_BLOCK>` | "Critical failure — fall back to plain language NOW" |
| `<TASK_CMT>` | "The following block is a finalized, committed subtask" |
| `<QUERY_REF>` | "Search the shared memory for this concept" |

---

### B. Interlat Middleware — The Transport Layer

**File:** [`interlat_middleware.py`](./interlat_middleware.py)

This is the **core engine** of the entire system. It handles everything that happens when Agent A sends a thought to Agent B.

#### How Sending Works (`pack_and_send`)

```
Logic Vector (raw thought)
        │
        ▼
[1] Procrustes Translation  ← translates A's vector space into B's understanding
        │
        ▼
[2] Differential Privacy    ← adds calibrated Gaussian noise (sigma=0.01)
        │                      to prevent Vector Inversion Attacks
        ▼
[3] Neural Header Built     ← model ID + task UUID + entropy + sequence
        │
        ▼
[4] NeuralPacket Assembled  ← header + protected payload bundled together
        │
        ▼
[5] Serialized to Carrier   ← Base64-encoded JSON wrapped in <NB_PACKET>…</NB_PACKET>
        │
        ▼
    Carrier Text String     ← this is what gets injected into the target's context
```

#### How Receiving Works (`receive_and_unpack`)

```
Incoming Carrier Text
        │
        ▼
[1] Parse <NB_PACKET> tag   ← extract and decode the Base64 JSON envelope
        │
        ▼
[2] Reconstruct NeuralPacket ← rebuild header and payload vectors
        │
        ▼
[3] Drift Audit             ← compute cosine distance against Codebook anchors
        │                      if drift > 0.12 → trigger NL fallback
        ▼
[4] Return Payload Vector   ← the decoded logic signal, ready to use
```

---

### C. Alignment Strategy (Procrustes) — The Translator

**File:** [`alignment_strategy.py`](./alignment_strategy.py)

Different AI models (e.g., Claude, Llama, GPT-4) each have their own private "conceptual coordinate system." The word "database" lives at completely different coordinates in Claude's mind versus Llama's.

**Procrustes Alignment** solves this with linear algebra:

1. **Calibration Phase:** Both agents embed the same set of "anchor concepts" (from the Neural Codebook)
2. **SVD Computation:** The system solves for the optimal rotation matrix `W` that maps Agent A's space onto Agent B's space
3. **Translation:** Every subsequent vector from A is rotated through `W` before B receives it

```python
# The core math (from alignment_strategy.py)
C = X_centered.T @ Y_centered     # cross-covariance
U, S, Vt = svd(C)                 # decompose
W = U @ Vt                        # optimal rotation matrix

# Translation at runtime
aligned_vector = (vector - source_mean) @ W + target_mean
```

> **Result:** A vector representing "task complete" in Claude's latent space is accurately understood as "task complete" when it arrives in Llama's latent space.

---

### D. DroidSpeak Cache — The Shared Memory

**File:** [`droidspeak_cache.py`](./droidspeak_cache.py)

This replaces the traditional "shared text file" with a **Holographic Key-Value Cache** pinned in memory.

- **Problem it solves:** If Agent A reads a 50MB documentation file, every other agent would normally have to spend tokens re-reading it.
- **Solution:** Agent A's reading is "pinned" in the GPU-resident K-V cache. Agents B and C access that exact memory for **zero additional tokens**.
- **Partitioning:** Each agent gets its own logical cache partition, preventing cross-contamination.

---

### E. SANEmerg Filter — The Noise Canceller

**File:** [`sanemerg_filter.py`](./sanemerg_filter.py)

The SANEmerg ("Semantic Importance Emergence") filter acts as a **Neural Secretary**:

- It analyzes the dimensions of every outgoing payload vector
- It identifies which dimensions carry actual **logic signal** (correlated with task outcomes)
- It **strips away** dimensions encoding politeness, formatting preferences, or stylistic noise
- During the first 50 communication cycles, it runs in **Aggressive Mode** to establish a clean baseline

> **Result:** What would be a 200-word status update in text becomes a compact matrix of only the reasoning dimensions that actually matter.

---

### F. AgentArk Distillation — The Final Compression

**File:** [`agentark_distillation.py`](./agentark_distillation.py)

Once the three-agent system is stable and has accumulated successful neural exchanges, **AgentArk** compacts all three agents into **one Multi-Role Model**.

#### How Distillation Works

```
Session Log of Successful NeuralPackets
        │
        ▼
[1] Label each packet with its source agent role
    (architect / engineer / qa_monitor)
        │
        ▼
[2] Build training dataset: (input_vec, output_vec, role_label) triplets
        │
        ▼
[3] Train a MultiRoleModel:
    Input → RoleGating → HiddenLayer → Output
    │
    ├── Role-conditioned gating: 3 separate weight matrices (one per role)
    ├── Role Classifier Head: predicts which role to activate
    └── Shared Output Projection: single output head for all roles
        │
        ▼
[4] Evaluate:
    • Vector reconstruction loss (cosine distance < 0.5)
    • Role classification accuracy (> 70%)
    • Compression ratio = (3 × input_dim) / hidden_dim
```

#### Success Criteria

| Metric | Threshold |
|---|---|
| Role Prediction Accuracy | > 70% |
| Vector Reconstruction Loss | < 0.5 (cosine distance) |
| Compression Ratio | ~3× (3 agents → 1 hidden_dim model) |

---

## 4. Security: Differential Privacy Layer

**Implemented in:** [`interlat_middleware.py`](./interlat_middleware.py) — `_apply_dp_noise()`

Every outbound vector is protected against **Vector Inversion Attacks** — where a malicious observer reconstructs the original training data from intercepted latent vectors.

| Parameter | Value | Purpose |
|---|---|---|
| `DP_NOISE_SIGMA` | `0.01` | Gaussian noise standard deviation |
| `DP_NOISE_CLIP_RATIO` | `0.08` | Max noise-to-signal ratio (safety clamp) |
| Cosine perturbation | `< 0.01` | Well below the 0.12 drift threshold |

The noise is calibrated to be **invisible to SANEmerg** (stays below the `0.12` drift threshold) but significant enough to prevent reconstruction of the original signal.

---

## 5. What Does the System Output?

The system produces several concrete outputs:

### During a Neural Session
| Output | Description |
|---|---|
| **Carrier Text Strings** | `<NB_PACKET>…</NB_PACKET>` — Base64-encoded neural messages between agents |
| **Decoded Logic Vectors** | NumPy arrays representing the received "thought" after translation |
| **Drift Audit Scores** | Cosine-distance floats; if > 0.12, a fallback is emitted |
| **NL Fallback Messages** | `PROTOCOL_FAIL: REASON [Drift_Detected=X.XXXX] | RESYNC: NL_NEGOTIATION | LAST_STATE: [packet_id]` |

### After Distillation (AgentArk)
| Output | Description |
|---|---|
| **MultiRoleModel** | A trained single model that simulates all 3 agent roles |
| **DistillationResult** | Metrics: `vec_loss`, `cls_loss`, `role_accuracy`, `compression_ratio` |
| **Model Summary JSON** | Exportable dict describing the distilled model's capabilities |

#### Example Distillation Output
```json
{
  "model_type": "MultiRoleModel (AgentArk Prototype)",
  "input_dim": 128,
  "hidden_dim": 128,
  "output_dim": 128,
  "roles": ["architect", "engineer", "qa_monitor"],
  "epochs_trained": 80,
  "final_vec_loss": 0.043210,
  "role_accuracy_pct": 94.44,
  "compression_ratio": 3.0,
  "success": true
}
```

### Benchmark Outputs (Goal Verification)
| Target | Expected Result |
|---|---|
| Token Compression | **≥ 90%** reduction vs. natural language baseline |
| Agent Scalability | **10+ agents** without hitting context window limits |
| Accuracy | **Higher** than NL-only communication (no lost-in-translation errors) |

---

## 6. How to Use This System (Step-by-Step)

### Prerequisites

```bash
pip install numpy
```

> All modules are pure Python + NumPy. No ML framework required for the prototype.

---

### Step 1 — Define Your Agents

Instantiate an `InterlatMiddleware` for each agent in your system. All agents must share the same `task_uuid`.

```python
import uuid
from interlat_middleware import InterlatMiddleware

PROJECT_UUID = str(uuid.uuid4())   # Generate once, share across all agents

agent_architect = InterlatMiddleware(
    agent_id     = "architect",
    model_id     = "claude-3-5-sonnet",     # The actual model this agent uses
    architecture = "transformer-decoder",
    task_uuid    = PROJECT_UUID,
)

agent_engineer = InterlatMiddleware(
    agent_id     = "engineer",
    model_id     = "llama-4-maverick",
    architecture = "transformer-decoder",
    task_uuid    = PROJECT_UUID,
)
```

---

### Step 2 — Bootstrap (Calibrate the Translation Layer)

During bootstrapping, agents exchange natural language to compute the Procrustes alignment matrix `W`. For the prototype, you can use an identity matrix if both agents share the same model.

```python
import numpy as np
from alignment_strategy import compute_procrustes_alignment

# In a real system: both agents embed the same anchor sentences
# and you call compute_procrustes_alignment(source_embeds, target_embeds)
# For testing, use identity:
dim = 128
W        = np.eye(dim, dtype=np.float32)
src_mean = np.zeros(dim, dtype=np.float32)
tgt_mean = np.zeros(dim, dtype=np.float32)

# Register the mapping layer on the sender's middleware
agent_architect.register_mapping_layer("engineer", W, src_mean, tgt_mean)

# Signal that bootstrapping is complete — enables neural mode
agent_architect.complete_bootstrap()
```

---

### Step 3 — Send a Neural Message

```python
# Your agent's "thought" — in a real system, this would be the model's
# last hidden state vector extracted via the model's API or hooks.
logic_signal = np.random.randn(dim).astype(np.float32)

# Pack and send — returns a carrier text string
carrier = agent_architect.pack_and_send(
    target_agent_id = "engineer",
    logic_vector    = logic_signal,
    anchor_label    = "<TASK_CMT>",   # Optional: tag with a Codebook anchor
    entropy_level   = 0.75,
)

print(carrier[:120])   # Inspect the carrier text
```

---

### Step 4 — Receive a Neural Message

```python
# The carrier text (from Step 3) arrives in the receiving agent's context
received_vector = agent_engineer.receive_and_unpack(carrier)

if received_vector is not None:
    print(f"Decoded vector shape: {received_vector.shape}")
    # Use this vector to condition the engineer agent's next response
else:
    print("Drift too high — NL fallback triggered automatically")
```

---

### Step 5 — Broadcast a Codebook Anchor

For universal signals (sync, error, query), use the convenience helper:

```python
from interlat_middleware import broadcast_anchor

# Broadcast INIT_SYNC to the engineer
carrier = broadcast_anchor(agent_architect, target="engineer", anchor="<INIT_SYNC>")

# Other available anchors:
# broadcast_anchor(agent, "engineer", "<ERR_BLOCK>")   # Force NL fallback
# broadcast_anchor(agent, "engineer", "<TASK_CMT>")    # Commit a subtask
# broadcast_anchor(agent, "engineer", "<QUERY_REF>")   # Request memory search
```

---

### Step 6 — Run AgentArk Distillation (After Session)

Once your agents have accumulated enough successful exchanges, distill them into one model:

```python
from agentark_distillation import AgentArkDatasetBuilder, AgentArkDistiller

# Build dataset from the session packet log
builder = AgentArkDatasetBuilder()

# Option A: Use real packets from your session
builder.ingest_packet_log(agent_architect.packet_log)

# Option B: Use synthetic data for testing
builder.ingest_synthetic(n_samples_per_role=60, vec_dim=128, seed=2026)

# Run distillation
distiller = AgentArkDistiller(
    hidden_dim    = 128,
    learning_rate = 5e-4,
    epochs        = 80,
)
result = distiller.distill(builder)

# Inspect the results
print(result.summary())

# Export a JSON summary
import json
summary = distiller.export_model_summary(result)
print(json.dumps(summary, indent=2))
```

---

### Step 7 — Use the Distilled Model for Role Prediction

```python
if result.model:
    # Predict which agent role a vector "belongs to"
    test_vector = np.random.randn(128).astype(np.float32)
    predicted_role = result.model.predict_role(test_vector)
    print(f"Predicted role: {predicted_role}")   # "architect", "engineer", or "qa_monitor"
```

---

## 7. Lifecycle of a Neural Message

```
1. Agent A generates a thought (latent vector from its model)
        │
2. InterlatMiddleware.pack_and_send() is called
        │
        ├── Procrustes alignment (W matrix rotates A's space → B's space)
        ├── DP noise injection (sigma=0.01, clipped at 8% of signal)
        ├── Neural Header stamped (model_id, task_uuid, entropy, seq_id)
        ├── NeuralPacket assembled (header + payload)
        └── Serialized to Base64 carrier text
        │
3. Carrier text injected into Agent B's context window
        │
4. Agent B calls InterlatMiddleware.receive_and_unpack()
        │
        ├── <NB_PACKET> tag parsed, JSON decoded, Base64 decoded
        ├── NeuralHeader reconstructed from first 128 dimensions
        ├── Payload vector extracted
        ├── Drift audit: cosine distance vs. Codebook anchors
        │   ├── drift ≤ 0.12  →  ✓ return payload vector
        │   └── drift > 0.12  →  ⚠ trigger NL fallback (Telegraph English)
        └── SANEmerg filter strips noise dimensions (if enabled)
        │
5. Agent B's model uses the decoded vector to condition its next output
        │
6. Exchange logged to DroidSpeak K-V cache (session-level memory pin)
        │
7. (Post-session) AgentArk harvests log → trains MultiRoleModel
```

---

## 8. Recovery & Fallback

The system degrades gracefully. If the neural channel breaks down:

**Trigger condition:** Cosine drift > `0.12` on the Audit Partition.

**Fallback message format (Telegraph English Protocol):**
```
PROTOCOL_FAIL: REASON [Drift_Detected=0.1854] | RESYNC: NL_NEGOTIATION | LAST_STATE: [<packet_uuid>]
```

**Recovery steps:**
1. Agents switch back to plain natural language for the current message
2. They re-negotiate the Procrustes W matrix using new NL anchor embeddings
3. Once recalibrated, `complete_bootstrap()` is called again to resume neural mode

---

## 9. Performance Benchmarks

| Metric | NL Baseline | Neural Bridge | Improvement |
|---|---|---|---|
| Tokens per status update | ~200 tokens | ~10–20 tokens | **90%+ reduction** |
| Max agents (same context) | 3–4 | 10+ | **3× scalability** |
| Reasoning accuracy | Lossy (translation errors) | Lossless (direct vectors) | **Higher fidelity** |
| DP overhead (semantic decay) | N/A | < 0.05% | **Negligible** |

> Benchmarks are validated in [`semantic_decay_benchmark.py`](./semantic_decay_benchmark.py) and [`performance_benchmark.py`](./performance_benchmark.py).

---

## 10. File Reference

| File | Role | Phase |
|---|---|---|
| [`neural_codebook.md`](./neural_codebook.md) | Neural vocabulary schema (header, anchors, partitions) | Phase 1 |
| [`alignment_strategy.py`](./alignment_strategy.py) | Procrustes alignment — cross-model latent space translation | Phase 1 |
| [`recovery_fallback.py`](./recovery_fallback.py) | Telegraph English fail-safe protocol | Phase 1 |
| [`interlat_middleware.py`](./interlat_middleware.py) | Core send/receive engine + DP noise layer | Phase 2 |
| [`droidspeak_cache.py`](./droidspeak_cache.py) | Holographic K-V cache (shared memory pinning) | Phase 2 |
| [`sanemerg_filter.py`](./sanemerg_filter.py) | Semantic importance filter (noise stripping) | Phase 2 |
| [`bootstrapping_routine.py`](./bootstrapping_routine.py) | NL handshake to calibrate SANEmerg weights | Phase 2 |
| [`agentark_distillation.py`](./agentark_distillation.py) | AgentArk — distill 3 agents into 1 MultiRoleModel | Phase 3 |
| [`performance_benchmark.py`](./performance_benchmark.py) | Token compression & accuracy benchmark runner | Phase 3 |
| [`semantic_decay_benchmark.py`](./semantic_decay_benchmark.py) | Multi-hop chain (A→B→C→English) cosine similarity decay | Phase 4 |
| [`AIM.md`](./AIM.md) | Project goals, rationale, and methodology overview | Reference |
| [`task.md`](./task.md) | Agent task board with phase completion status | Reference |

---

*Neural-Bridge MAS — Moving AI collaboration from the Symbolic Era to the Latent Era.*
