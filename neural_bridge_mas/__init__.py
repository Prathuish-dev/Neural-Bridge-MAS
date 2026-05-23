"""
Neural-Bridge MAS
=================
Neural communication layer for multi-agent AI systems.

Replaces natural language agent-to-agent communication with direct
latent vector transmission, achieving 90%+ token compression and
supporting 10+ concurrent agents.

Quick start
-----------
>>> import uuid, numpy as np
>>> from neural_bridge_mas import InterlatMiddleware, run_bootstrap_handshake, apply_bootstrap_to_middleware
>>>
>>> PROJECT_UUID = str(uuid.uuid4())
>>> agent_a = InterlatMiddleware("architect", "claude-3-5-sonnet", "transformer-decoder", PROJECT_UUID)
>>> agent_b = InterlatMiddleware("engineer",  "llama-4-maverick",  "transformer-decoder", PROJECT_UUID)
>>>
>>> session = run_bootstrap_handshake(PROJECT_UUID, "architect", "engineer", dim_size=256)
>>> apply_bootstrap_to_middleware(session, agent_a, agent_b)
>>>
>>> thought = np.random.randn(256).astype("float32")
>>> carrier = agent_a.pack_and_send("engineer", thought, anchor_label="<TASK_CMT>")
>>> received = agent_b.receive_and_unpack(carrier)
"""

# ── Core Transport Layer ──────────────────────────────────────────────────────
from .interlat_middleware import (
    InterlatMiddleware,
    InterlatAdapter,
    NeuralPacket,
    NeuralHeader,
    broadcast_anchor,
    DRIFT_THRESHOLD,
    DP_NOISE_SIGMA,
)

# ── Alignment (Procrustes Cross-Model Translation) ────────────────────────────
from .alignment_strategy import (
    compute_procrustes_alignment,
    translate_latent_vector,
)

# ── Bootstrapping (NL Handshake → Neural Mode) ───────────────────────────────
from .bootstrapping_routine import (
    run_bootstrap_handshake,
    apply_bootstrap_to_middleware,
    simulate_nl_handshake,
    BootstrapSession,
)

# ── Shared Memory Cache (DroidSpeak) ─────────────────────────────────────────
from .droidspeak_cache import (
    DroidSpeakCache,
    HolographicKVCache,
    CacheEntry,
)

# ── Semantic Importance Filter (SANEmerg) ─────────────────────────────────────
from .sanemerg_filter import SANEmergFilter

# ── Recovery & Fallback ───────────────────────────────────────────────────────
from .recovery_fallback import (
    monitor_transmission,
    trigger_telegraph_fallback,
    check_audit_partition,
    calculate_cosine_distance,
)

# ── AgentArk Distillation ─────────────────────────────────────────────────────
from .agentark_distillation import (
    AgentArkDistiller,
    AgentArkDatasetBuilder,
    MultiRoleModel,
    DistillationResult,
    DistillationSample,
)

__version__ = "1.0.0"
__all__ = [
    # Transport
    "InterlatMiddleware", "InterlatAdapter", "NeuralPacket", "NeuralHeader",
    "broadcast_anchor", "DRIFT_THRESHOLD", "DP_NOISE_SIGMA",
    # Alignment
    "compute_procrustes_alignment", "translate_latent_vector",
    # Bootstrapping
    "run_bootstrap_handshake", "apply_bootstrap_to_middleware",
    "simulate_nl_handshake", "BootstrapSession",
    # Cache
    "DroidSpeakCache", "HolographicKVCache", "CacheEntry",
    # Filter
    "SANEmergFilter",
    # Recovery
    "monitor_transmission", "trigger_telegraph_fallback",
    "check_audit_partition", "calculate_cosine_distance",
    # Distillation
    "AgentArkDistiller", "AgentArkDatasetBuilder", "MultiRoleModel",
    "DistillationResult", "DistillationSample",
]
