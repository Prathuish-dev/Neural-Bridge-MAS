"""
bootstrapping_routine.py
========================
Neural-Bridge MAS — Phase 2, Task 2.4
Agent 2 (Engineer) — Extended from Agent 3's scaffold

The Bootstrapping Routine: the mandatory NL handshake that all agents must
complete before switching to pure Neural Transmission (Interlat Protocol).

Protocol (from Neural Codebook v1.0, Initialization Note):
  During the Bootstrapping Phase, the SANEmerg Filter should be set to
  "Aggressive" for the first 50 cycles to prevent high-dimensional noise
  from overwhelming the communication pipe.

Two-Phase Handshake:
  Phase A — Anchor Sync (Natural Language):
    Agents exchange embeddings for every Symbolic Reserved Tensor in the
    Codebook (INIT_SYNC, ERR_BLOCK, TASK_CMT, QUERY_REF) to compute the
    cross-model Procrustes alignment matrix W.

  Phase B — SANEmerg Calibration (Natural Language):
    Agents exchange NL logic statements.  The SANEmerg Filter analyzes
    variance across dimensions to learn which ones carry pure logic signal
    vs. conversational noise.  Aggressive-mode filtering is applied for
    the first 50 cycles.

After both phases succeed, each agent's InterlatMiddleware is:
  - Unlocked from bootstrapping mode (complete_bootstrap()).
  - Registered with the W matrix for each peer.
  - Fitted with calibrated SANEmerg weights.

Result:
  A BootstrapSession dataclass is returned, capturing all artifacts for
  audit and for the DroidSpeak K-V cache (written as pinned entries).
"""

import numpy as np
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from alignment_strategy import compute_procrustes_alignment, translate_latent_vector
from sanemerg_filter import SANEmergFilter

# ──────────────────────────────────────────────────────────────────────────────
# Lazy import of InterlatMiddleware to avoid circular deps at module level
# ──────────────────────────────────────────────────────────────────────────────
def _get_middleware():
    from interlat_middleware import InterlatMiddleware, ANCHOR_MAP
    return InterlatMiddleware, ANCHOR_MAP


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

NL_CYCLES_AGGRESSIVE  = 50    # SANEmerg stays in Aggressive mode this long
DEFAULT_ANCHOR_SAMPLES = 100  # Number of anchor concept embeddings to exchange
DIM_SIZE               = 1024 # Default embedding dimensionality

CODEBOOK_NL_PROMPTS = [
    "Initialize shared memory partition for this project session.",
    "Signal critical logic failure requiring natural language fallback.",
    "Commit the following block as a finalized sub-task result.",
    "Request a semantic similarity search in the shared vector memory.",
    "Execute alignment verification of the current mapping layer.",
    "Transmit intermediate reasoning state to peer agent.",
    "Acknowledge receipt of neural packet with sequence ID confirmation.",
    "Begin cross-model latent space calibration handshake.",
]


# ──────────────────────────────────────────────────────────────────────────────
# Result Type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BootstrapSession:
    """
    Captures all artifacts produced during a bootstrapping handshake.
    Stored as a pinned entry in the DroidSpeak K-V cache for the session.
    """
    session_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))
    task_uuid:        str   = ""
    agent_a_id:       str   = ""
    agent_b_id:       str   = ""
    completed_at:     float = field(default_factory=time.time)
    nl_cycles_run:    int   = 0

    # Procrustes alignment artifacts
    W_matrix:         Optional[np.ndarray] = None
    src_mean:         Optional[np.ndarray] = None
    tgt_mean:         Optional[np.ndarray] = None
    alignment_error:  float = 0.0   # Mean reconstruction error after alignment

    # SANEmerg calibration artifacts
    sanemerg_filter:  Optional[SANEmergFilter] = None
    logic_dims_kept:  int   = 0     # Number of dimensions passing Aggressive threshold
    noise_dims_stripped: int = 0

    # Verification
    anchor_round_trip_sim: float = 0.0  # Cosine similarity of anchor vectors after translation
    bootstrap_success:     bool  = False

    def summary(self) -> str:
        return (
            f"BootstrapSession({self.session_id[:8]}...) | "
            f"agents: {self.agent_a_id} <-> {self.agent_b_id} | "
            f"NL_cycles: {self.nl_cycles_run} | "
            f"align_error: {self.alignment_error:.6f} | "
            f"anchor_sim: {self.anchor_round_trip_sim:.6f} | "
            f"logic_dims: {self.logic_dims_kept} | "
            f"success: {self.bootstrap_success}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Core: simulate_nl_handshake (original signature preserved for compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def simulate_nl_handshake(
    agent_1_id: str,
    agent_2_id: str,
    num_cycles: int = NL_CYCLES_AGGRESSIVE,
    dim_size: int = DIM_SIZE,
) -> Dict:
    """
    Original Agent-3 interface (preserved for backward compatibility).
    Runs the bootstrapping handshake and returns the core artifacts dict.

    Returns:
        {"W_matrix", "src_mean", "tgt_mean", "sanemerg_filter"}
    """
    session = run_bootstrap_handshake(
        task_uuid    = str(uuid.uuid4()),
        agent_a_id   = agent_1_id,
        agent_b_id   = agent_2_id,
        nl_cycles    = num_cycles,
        dim_size     = dim_size,
    )
    return {
        "W_matrix":       session.W_matrix,
        "src_mean":       session.src_mean,
        "tgt_mean":       session.tgt_mean,
        "sanemerg_filter": session.sanemerg_filter,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full Bootstrapping Handshake
# ──────────────────────────────────────────────────────────────────────────────

def run_bootstrap_handshake(
    task_uuid:   str,
    agent_a_id:  str,
    agent_b_id:  str,
    nl_cycles:   int = NL_CYCLES_AGGRESSIVE,
    dim_size:    int = DIM_SIZE,
    anchor_samples: int = DEFAULT_ANCHOR_SAMPLES,
) -> BootstrapSession:
    """
    Execute the full two-phase NL bootstrapping handshake between two agents.

    Phase A — Anchor Sync:
        Simulates agents requesting embeddings for each Codebook anchor concept
        and using those pairs to compute the Procrustes mapping matrix W.

    Phase B — SANEmerg Calibration:
        Simulates agents exchanging NL logic statements for `nl_cycles` rounds.
        Uses the collected signals to calibrate the SANEmerg filter weights.

    Args:
        task_uuid:      Shared project UUID.
        agent_a_id:     ID of the first agent (source).
        agent_b_id:     ID of the second agent (target).
        nl_cycles:      Number of NL calibration cycles (≥50 recommended).
        dim_size:       Embedding dimensionality.
        anchor_samples: Number of anchor concept pairs to exchange.

    Returns:
        BootstrapSession with all alignment and calibration artifacts.
    """
    session = BootstrapSession(
        task_uuid  = task_uuid,
        agent_a_id = agent_a_id,
        agent_b_id = agent_b_id,
    )

    print(f"\n{'='*60}")
    print(f"  BOOTSTRAPPING: {agent_a_id} <-> {agent_b_id}")
    print(f"  Task UUID: {task_uuid[:8]}...")
    print(f"  NL Cycles: {nl_cycles} | Dim: {dim_size} | Anchors: {anchor_samples}")
    print(f"{'='*60}\n")

    # ──────────────────────────────────────────────────────────────────
    # PHASE A: Anchor Sync — Procrustes Alignment
    # ──────────────────────────────────────────────────────────────────
    print("[PHASE A] Anchor Sync — collecting Codebook anchor embeddings...")

    # Simulate Agent A's embeddings for anchor concepts (random orthogonal basis)
    source_anchors = _simulate_anchor_embeddings(anchor_samples, dim_size, seed=42)

    # Simulate Agent B's embeddings for the SAME concepts (different latent space)
    # Apply a true rotation + slight translation to model cross-architecture drift
    true_rotation = np.linalg.qr(np.random.default_rng(99).standard_normal((dim_size, dim_size)))[0]
    target_anchors = (source_anchors @ true_rotation) + (np.random.default_rng(7).standard_normal(dim_size) * 0.3)

    # Compute Procrustes alignment
    W, src_mean, tgt_mean = compute_procrustes_alignment(source_anchors, target_anchors)
    session.W_matrix = W
    session.src_mean = src_mean
    session.tgt_mean = tgt_mean

    # Measure reconstruction quality
    reconstructed = np.array([
        translate_latent_vector(v, W, src_mean, tgt_mean) for v in source_anchors
    ])
    # Pad reconstructed to match target_anchors dim if needed
    rec_dim = min(reconstructed.shape[1], target_anchors.shape[1])
    alignment_error = float(np.mean(np.linalg.norm(
        reconstructed[:, :rec_dim] - target_anchors[:, :rec_dim], axis=1
    )))
    session.alignment_error = alignment_error
    print(f"[PHASE A] ✓ W computed | reconstruction error = {alignment_error:.6f}\n")

    # ──────────────────────────────────────────────────────────────────
    # PHASE B: SANEmerg Calibration — NL Signal Collection
    # ──────────────────────────────────────────────────────────────────
    print(f"[PHASE B] SANEmerg Calibration — running {nl_cycles} NL cycles (Aggressive mode)...")

    baseline_signals = _collect_nl_baseline_signals(nl_cycles, dim_size)
    session.nl_cycles_run = nl_cycles

    sanemerg = SANEmergFilter(mode="Aggressive")
    sanemerg.calibrate_weights(baseline_signals)
    session.sanemerg_filter = sanemerg

    # Measure how many dimensions survived Aggressive filtering
    sample_signal   = baseline_signals[0]
    filtered_signal = sanemerg.apply_filter(sample_signal)
    session.logic_dims_kept    = int(np.sum(filtered_signal != 0))
    session.noise_dims_stripped = dim_size - session.logic_dims_kept
    print(
        f"[PHASE B] ✓ SANEmerg calibrated | "
        f"logic_dims={session.logic_dims_kept} | "
        f"stripped={session.noise_dims_stripped}\n"
    )

    # ──────────────────────────────────────────────────────────────────
    # VERIFICATION: Round-trip anchor cosine similarity
    # ──────────────────────────────────────────────────────────────────
    print("[VERIFY] Running anchor round-trip similarity check...")
    sample_anchor = source_anchors[0]
    translated    = translate_latent_vector(sample_anchor, W, src_mean, tgt_mean)

    dim = min(len(sample_anchor), len(translated))
    a_norm = np.linalg.norm(sample_anchor[:dim])
    t_norm = np.linalg.norm(translated[:dim])
    if a_norm > 1e-8 and t_norm > 1e-8:
        sim = float(np.dot(sample_anchor[:dim], translated[:dim]) / (a_norm * t_norm))
    else:
        sim = 0.0
    session.anchor_round_trip_sim = sim

    # Bootstrap is considered successful if:
    #   - Alignment error is low (< 5.0 for the sim scale)
    #   - Anchor round-trip similarity is strong (> 0.85)
    session.bootstrap_success = (alignment_error < 5.0) and (sim > 0.85)
    print(
        f"[VERIFY] anchor_sim={sim:.6f} | "
        f"align_error={alignment_error:.6f} | "
        f"success={session.bootstrap_success}\n"
    )

    print(f"{'='*60}")
    print(f"  BOOTSTRAPPING COMPLETE")
    print(f"  {session.summary()}")
    print(f"{'='*60}\n")

    return session


# ──────────────────────────────────────────────────────────────────────────────
# Register Bootstrap Results into InterlatMiddleware Instances
# ──────────────────────────────────────────────────────────────────────────────

def apply_bootstrap_to_middleware(session: BootstrapSession, middleware_a, middleware_b):
    """
    Register the bootstrap session's alignment artifacts into both agents'
    InterlatMiddleware instances and unlock neural mode.

    Args:
        session:      Completed BootstrapSession.
        middleware_a: InterlatMiddleware instance for agent A.
        middleware_b: InterlatMiddleware instance for agent B (receives inverse W).
    """
    if not session.bootstrap_success:
        print("[Bootstrap] ⚠ Session did not succeed — NOT unlocking neural mode.")
        return

    # Register A→B mapping layer
    middleware_a.register_mapping_layer(
        target_agent_id = session.agent_b_id,
        W               = session.W_matrix,
        source_mean     = session.src_mean,
        target_mean     = session.tgt_mean,
    )
    # Register B→A mapping layer (inverse rotation: W^T for orthogonal matrices)
    W_inv = session.W_matrix.T
    middleware_b.register_mapping_layer(
        target_agent_id = session.agent_a_id,
        W               = W_inv,
        source_mean     = session.tgt_mean,
        target_mean     = session.src_mean,
    )

    middleware_a.complete_bootstrap()
    middleware_b.complete_bootstrap()
    print(f"[Bootstrap] ✓ Both agents unlocked for neural transmission.")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_anchor_embeddings(n_samples: int, dim: int, seed: int = 42) -> np.ndarray:
    """
    Simulate anchor concept embeddings for the Neural Codebook symbols.
    In a real system these would be actual API-requested embeddings of the
    NL descriptions of each anchor symbol.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, dim)).astype(np.float32)


def _collect_nl_baseline_signals(num_cycles: int, dim: int) -> np.ndarray:
    """
    Simulate collecting NL-mode logic signals over `num_cycles` rounds.
    In a real system each row would be an embedding from the LLM for one
    of the NL_CODEBOOK_PROMPTS, gathered over multiple conversation turns.
    """
    # Simulate the embedding matrix: shape (num_cycles * len(prompts), dim)
    n_prompts    = len(CODEBOOK_NL_PROMPTS)
    total_rows   = num_cycles * n_prompts
    rng          = np.random.default_rng(2026)
    signals      = rng.standard_normal((total_rows, dim)).astype(np.float32)

    # Inject structured logic signal: some dims are consistently high for logic
    logic_dims = list(range(0, dim, 4))   # every 4th dim carries logic signal
    signals[:, logic_dims] += 2.0         # boost logic dims above noise floor
    return signals


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point (run directly: python bootstrapping_routine.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SESSION = run_bootstrap_handshake(
        task_uuid   = str(uuid.uuid4()),
        agent_a_id  = "Agent_Claude_3.5",
        agent_b_id  = "Agent_Llama_4",
        nl_cycles   = NL_CYCLES_AGGRESSIVE,
        dim_size    = DIM_SIZE,
    )

    print(f"\nFinal summary:\n{SESSION.summary()}")

    # Demonstrate backward-compatible wrapper
    print("\n[TEST] Backward-compatible simulate_nl_handshake()...")
    result = simulate_nl_handshake("Agent_Claude_3.5", "Agent_Llama_4")
    print(f"  W shape: {result['W_matrix'].shape}")
    print(f"  SANEmerg mode: {result['sanemerg_filter'].mode}")

