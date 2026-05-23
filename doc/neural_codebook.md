# Neural Codebook v1.0 (Draft)

The **Neural Codebook** is a Schema of Latent Tensors. It instructs agents on how to interpret high-density vector blocks.

## I. The Neural Header (The "Envelope")

Every neural transmission must begin with a 128-dimension **Context Vector** (The Header):

* **[DIM 0-31]:** **Model ID & Architecture** (Allows the Mapping Layer to select the correct Procrustes transform).
* **[DIM 32-63]:** **Task UUID** (Prevents "Context Bleed" from other ongoing projects).
* **[DIM 64-95]:** **Entropy Density Level** (Instructs the SANEmerg filter on noise vs. signal expectation).
* **[DIM 96-127]:** **Temporal Marker** (The sequence ID for DroidSpeak K-V cache alignment).

## II. Symbolic Reserved Tensors (The "Anchors")

Specific **Coordinate Points** in the latent space that represent universal agent states:

| Semantic Label | Vector Descriptor (Normalized) | Purpose |
| --- | --- | --- |
| `<INIT_SYNC>` | `[1.0, 0.0, ... 0.0]` | Requesting a DroidSpeak Cache partition. |
| `<ERR_BLOCK>` | `[-1.0, -1.0, ... 1.0]` | Critical logic failure; requires immediate NL fallback. |
| `<TASK_CMT>` | `[0.5, 0.5, ... 0.0]` | Signal that the following block is a finalized sub-task. |
| `<QUERY_REF>` | `[0.0, 0.5, ... 0.5]` | Requesting a similarity search in the shared vector memory. |

## III. Data Payloads (The "Logic Signals")

The Codebook defines how the variable block dimensions are partitioned:

1. **Code/Logic Partition:** High-precision dimensions (32-bit floats) focused on syntax and structural integrity.
2. **Creative/Abstract Partition:** High-variance dimensions (16-bit brain-floats) focused on broad conceptual mapping.
3. **Audit/Verification Partition:** Parity-check dimensions used to ensure the "Neural Bridge" hasn't introduced hallucinated drift.

## IV. Recovery Fallback (The "Telegraph English" Protocol)

If the **Audit Partition** detects a drift $> 0.12$ (Cosine Distance), the agents must immediately shift to this syntax:

> `PROTOCOL_FAIL: REASON [Drift_Detected] | RESYNC: NL_NEGOTIATION | LAST_STATE: [Vector_ID]`

---
**Note on Initialization:** During the Bootstrapping Phase, the **SANEmerg Filter** should be set to "Aggressive" for the first 50 cycles to prevent high-dimensional noise from overwhelming the communication pipe.
