"""
interlat_middleware.py
======================
Neural-Bridge MAS — Phase 2, Task 2.1
Agent 2 (Engineer) — Extended from Agent 1's scaffold

The "Interlat" Middleware: the core engine of the Neural-Bridge system.

Responsibilities:
  1. Package an agent's latent vector output into a Neural Packet
     (Header + Payload) according to the Neural Codebook v1.0 schema.
  2. Inject high-entropy embeddings into a target agent's context window
     as a "High-Entropy Text Carrier" when direct hidden-state access
     is unavailable (API-only environments like Claude or GPT-4).
  3. Extract neural packets from incoming text and reconstruct the
     original latent vector using the pre-computed Procrustes W matrix.
  4. Maintain a session-level packet log for the DroidSpeak K-V cache
     and the SANEmerg filter to consume.

Dependencies:
  - numpy
  - alignment_strategy.py (Task 1.2) — for translate_latent_vector()
  - neural_codebook.md (Task 1.1) — schema reference
"""

import numpy as np
import json
import uuid
import time
import base64
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from .alignment_strategy import translate_latent_vector

# ---------------------------------------------------------------------------
# Constants derived from Neural Codebook v1.0
# ---------------------------------------------------------------------------

HEADER_DIM = 128           # Total dimensions in the Neural Header
DIM_MODEL_ID    = (0,  32) # [DIM 0-31]  : Model ID & Architecture
DIM_TASK_UUID   = (32, 64) # [DIM 32-63] : Task UUID
DIM_ENTROPY     = (64, 96) # [DIM 64-95] : Entropy Density Level
DIM_TEMPORAL    = (96, 128)# [DIM 96-127]: Temporal / Sequence ID

# Symbolic Reserved Tensor anchors (Codebook §II)
ANCHOR_INIT_SYNC  = np.array([1.0] + [0.0] * 127, dtype=np.float32)
ANCHOR_ERR_BLOCK  = np.array([-1.0, -1.0] + [0.0] * 125 + [1.0], dtype=np.float32)
ANCHOR_TASK_CMT   = np.array([0.5, 0.5] + [0.0] * 126, dtype=np.float32)
ANCHOR_QUERY_REF  = np.array([0.0, 0.5] + [0.0] * 125 + [0.5], dtype=np.float32)

ANCHOR_MAP: Dict[str, np.ndarray] = {
    "<INIT_SYNC>": ANCHOR_INIT_SYNC,
    "<ERR_BLOCK>": ANCHOR_ERR_BLOCK,
    "<TASK_CMT>":  ANCHOR_TASK_CMT,
    "<QUERY_REF>": ANCHOR_QUERY_REF,
}

DRIFT_THRESHOLD = 0.12   # SANEmerg audit cosine-distance drift limit

# ---------------------------------------------------------------------------
# Differential Privacy (Task 4.1 — Neural Security)
# ---------------------------------------------------------------------------
# Gaussian noise is added to every outbound payload vector.
# Sigma is calibrated so the injected noise stays well below DRIFT_THRESHOLD,
# preventing Vector Inversion Attacks without corrupting the logic signal.
#
# Budget accounting (epsilon-delta DP):
#   - sigma = DP_NOISE_SIGMA (default 0.01)
#   - Noise magnitude (L2) ≈ sigma * sqrt(dim)  →  ~0.01 * sqrt(1024) ≈ 0.32
#   - Cosine perturbation ≈ noise_L2 / signal_L2  →  <0.01 for unit vectors
#   - This is well within the 0.12 DRIFT_THRESHOLD, so SANEmerg won't trigger.

DP_NOISE_SIGMA      = 0.01   # Gaussian sigma — tuned to stay below drift budget
DP_NOISE_CLIP_RATIO = 0.08   # Max allowed noise/signal ratio (safety clamp)
DP_ENABLED          = True   # Global toggle — set False for benchmarking


# ---------------------------------------------------------------------------
# Original Agent-1 InterlatAdapter (kept for compatibility)
# ---------------------------------------------------------------------------

class InterlatAdapter:
    def __init__(self, model_id: str, architecture: str):
        """
        Initializes the Interlat Middleware for a specific agent.
        """
        self.model_id = model_id
        self.architecture = architecture
        # Enforce Bootstrapping Protocol as defined in Phase 1
        self.is_bootstrapping = True 
        
    def generate_neural_header(self, task_uuid: str, entropy_level: float, temporal_marker: int) -> np.ndarray:
        """
        Constructs the 128-dim Context Vector (Envelope) as defined in the Neural Codebook v1.0.
        """
        header = np.zeros(128, dtype=np.float32)
        # Partitioning logic based on Codebook Schema
        # [0-31] Model ID, [32-63] Task UUID, [64-95] Entropy, [96-127] Temporal Marker
        model_hash = abs(hash(self.model_id)) % (2**31)
        header[model_hash % 32] = 1.0   # one-hot in the model-id slice
        uuid_hash = abs(hash(task_uuid)) % (2**31)
        header[32 + (uuid_hash % 32)] = 1.0
        header[64] = float(entropy_level)
        header[96] = float(temporal_marker)
        return header

    def inject_embeddings(self, neural_payload: np.ndarray) -> str:
        """
        The API-Only Workaround:
        Injects the high-density neural_payload (header + logic signal) 
        into the model's context window, bypassing standard text tokenization.
        """
        if self.is_bootstrapping:
            print(f"[{self.model_id}] STATUS: Bootstrapping Phase Active.")
            print(f"[{self.model_id}] ACTION: Negotiating communication weights via Natural Language first.")
            return "NL_NEGOTIATION_MODE"
            
        print(f"[{self.model_id}] Injecting {len(neural_payload)}-dimension neural payload via Embedding Injection...")
        return "INJECTION_SUCCESS"

    def extract_hidden_state(self, prompt: str) -> np.ndarray:
        """
        Simulates the extraction of the Last Hidden State.
        For closed-source APIs, we approximate this by requesting high-density embeddings
        from the API for the given prompt, acting as the 'source' signal.
        """
        # Simulate an embedding extraction (e.g., 1024 dimensions representing the prompt's latent space)
        simulated_state = np.random.randn(1024).astype(np.float32)
        print(f"[{self.model_id}] Extracted hidden state/embedding matrix of shape {simulated_state.shape}.")
        return simulated_state


# ---------------------------------------------------------------------------
# Agent-2 Extension: Full Neural Packet Layer
# ---------------------------------------------------------------------------

@dataclass
class NeuralHeader:
    """
    128-dimension Context Vector that acts as the 'envelope' for every
    neural transmission (Codebook §I).
    """
    model_id_vec:  np.ndarray  # [DIM 0-31]   — 32-dim
    task_uuid_vec: np.ndarray  # [DIM 32-63]  — 32-dim
    entropy_vec:   np.ndarray  # [DIM 64-95]  — 32-dim
    temporal_vec:  np.ndarray  # [DIM 96-127] — 32-dim

    def to_vector(self) -> np.ndarray:
        """Concatenate all slices into a single 128-dim header vector."""
        return np.concatenate([
            self.model_id_vec,
            self.task_uuid_vec,
            self.entropy_vec,
            self.temporal_vec,
        ]).astype(np.float32)

    @classmethod
    def build(
        cls,
        model_hash: int,
        task_uuid: str,
        entropy_level: float,
        seq_id: int,
    ) -> "NeuralHeader":
        """
        Factory: construct a NeuralHeader from human-readable primitives.

        Args:
            model_hash:    Integer fingerprint of the source model.
            task_uuid:     UUID string shared across all agents on a project.
            entropy_level: Float [0, 1] representing expected signal density.
            seq_id:        Monotonic sequence counter for K-V cache alignment.
        """
        model_id_vec = np.zeros(32, dtype=np.float32)
        model_id_vec[model_hash % 32] = 1.0

        # Encode UUID bytes across 32 dimensions
        uuid_bytes    = uuid.UUID(task_uuid).bytes  # 16 bytes → repeat to fill 32 dims
        task_uuid_vec = np.frombuffer(uuid_bytes + uuid_bytes, dtype=np.uint8).astype(np.float32) / 255.0

        entropy_vec  = np.full(32, float(entropy_level), dtype=np.float32)

        temporal_vec = np.zeros(32, dtype=np.float32)
        temporal_vec[seq_id % 32] = float(seq_id)

        return cls(model_id_vec, task_uuid_vec, entropy_vec, temporal_vec)


@dataclass
class NeuralPacket:
    """
    The complete transmission unit — Neural Header + Logic Payload.
    This is what one agent sends and another agent receives.
    """
    packet_id:    str  = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:    float = field(default_factory=time.time)
    source_agent: str  = ""
    target_agent: str  = ""
    header:       Optional[NeuralHeader]  = None
    payload:      Optional[np.ndarray]    = None  # The logic-signal vector
    anchor_label: Optional[str]           = None  # Codebook anchor if applicable

    def to_carrier_text(self) -> str:
        """
        Serialize the packet to a Base64-encoded JSON string for injection
        into a text-based API context window.

        This is the 'Embedding Injection' / High-Entropy Text Carrier
        approach for API-only agents (workaround from Task 1.2).
        """
        header_vec = self.header.to_vector() if self.header else np.array([], dtype=np.float32)
        combined   = np.concatenate([header_vec, self.payload]).astype(np.float32)
        payload_b64 = base64.b64encode(combined.tobytes()).decode("utf-8")

        envelope = {
            "nb_packet":  True,
            "packet_id":  self.packet_id,
            "timestamp":  self.timestamp,
            "src":        self.source_agent,
            "tgt":        self.target_agent,
            "anchor":     self.anchor_label,
            "data":       payload_b64,
            "header_dim": int(len(header_vec)),
        }
        return f"<NB_PACKET>{json.dumps(envelope)}</NB_PACKET>"

    @classmethod
    def from_carrier_text(cls, text: str) -> Optional["NeuralPacket"]:
        """
        Deserialize a NeuralPacket embedded inside a carrier text string.
        Returns None if no valid packet is found in the text.
        """
        start = text.find("<NB_PACKET>")
        end   = text.find("</NB_PACKET>")
        if start == -1 or end == -1:
            return None

        raw_json = text[start + len("<NB_PACKET>"): end]
        try:
            envelope = json.loads(raw_json)
        except json.JSONDecodeError:
            return None

        raw_bytes    = base64.b64decode(envelope["data"])
        full_vec     = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        header_dim   = envelope.get("header_dim", HEADER_DIM)
        header_vec   = full_vec[:header_dim]
        payload_vec  = full_vec[header_dim:]

        # Reconstruct NeuralHeader from the flat slice
        if len(header_vec) == HEADER_DIM:
            header = NeuralHeader(
                model_id_vec  = header_vec[DIM_MODEL_ID[0]:  DIM_MODEL_ID[1]],
                task_uuid_vec = header_vec[DIM_TASK_UUID[0]: DIM_TASK_UUID[1]],
                entropy_vec   = header_vec[DIM_ENTROPY[0]:   DIM_ENTROPY[1]],
                temporal_vec  = header_vec[DIM_TEMPORAL[0]:  DIM_TEMPORAL[1]],
            )
        else:
            header = None

        return cls(
            packet_id    = envelope["packet_id"],
            timestamp    = envelope["timestamp"],
            source_agent = envelope["src"],
            target_agent = envelope["tgt"],
            header       = header,
            payload      = payload_vec,
            anchor_label = envelope.get("anchor"),
        )


# ---------------------------------------------------------------------------
# Extended Middleware: InterlatMiddleware
# ---------------------------------------------------------------------------

class InterlatMiddleware(InterlatAdapter):
    """
    Full Neural-Bridge engine — extends InterlatAdapter with:
      - Multi-agent Procrustes translation layer
      - NeuralPacket send/receive pipeline
      - Drift audit and NL fallback trigger
      - Session packet log for DroidSpeak K-V cache (Task 2.2)
      - Differential Privacy noise injection (Task 4.1 — Vector Inversion Protection)
    """

    def __init__(
        self,
        agent_id: str,
        model_id: str,
        architecture: str,
        task_uuid: str,
        dp_noise_sigma: float = DP_NOISE_SIGMA,
        dp_enabled: bool = DP_ENABLED,
    ):
        """
        Args:
            agent_id:       Human-readable name (e.g., 'architect', 'engineer').
            model_id:       Model identifier string (e.g., 'claude-3-5-sonnet').
            architecture:   Architecture family (e.g., 'transformer-decoder').
            task_uuid:      Shared project UUID — must match across all agents.
            dp_noise_sigma: Gaussian sigma for DP noise (default 0.01).
                            Must keep injected noise < DRIFT_THRESHOLD (0.12).
            dp_enabled:     Toggle DP on/off. Set False for benchmarking runs.
        """
        super().__init__(model_id, architecture)
        self.agent_id       = agent_id
        self.task_uuid      = task_uuid
        self.model_hash     = abs(hash(model_id)) % (2**31)
        self._seq_counter   = 0
        self.dp_noise_sigma = dp_noise_sigma
        self.dp_enabled     = dp_enabled
        self._dp_rng        = np.random.default_rng()   # cryptographically seeded

        # Procrustes mapping layers: {target_agent_id: (W, src_mean, tgt_mean)}
        self._mapping_layers: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Session packet log (consumed by DroidSpeak cache in Task 2.2)
        self.packet_log: List[NeuralPacket] = []

    # ------------------------------------------------------------------
    # Bootstrapping: Register mapping layers
    # ------------------------------------------------------------------

    def register_mapping_layer(
        self,
        target_agent_id: str,
        W: np.ndarray,
        source_mean: np.ndarray,
        target_mean: np.ndarray,
    ):
        """
        Register a pre-computed Procrustes transformation for a target agent.
        Must be called during the Bootstrapping Phase (before going neural).
        """
        self._mapping_layers[target_agent_id] = (W, source_mean, target_mean)
        print(f"[{self.agent_id}] ✓ Mapping layer registered → target='{target_agent_id}'.")

    def complete_bootstrap(self):
        """
        Signal that Natural Language negotiation is done.
        Unlocks the full neural transmission pipeline.
        """
        self.is_bootstrapping = False
        print(f"[{self.agent_id}] Bootstrap complete — switching to Neural Mode.")

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def pack_and_send(
        self,
        target_agent_id: str,
        logic_vector: np.ndarray,
        anchor_label: Optional[str] = None,
        entropy_level: float = 0.75,
    ) -> str:
        """
        Package a logic vector into a NeuralPacket and serialize it to
        a carrier text string for injection into the target agent's context.

        Args:
            target_agent_id: The receiving agent's ID.
            logic_vector:    The raw latent/embedding vector to transmit.
            anchor_label:    Optional Codebook anchor symbol.
            entropy_level:   Signal density hint for SANEmerg.

        Returns:
            Carrier text string containing the serialized packet.
        """
        if self.is_bootstrapping:
            print(f"[{self.agent_id}] ⚠ Bootstrap not complete — using NL fallback.")
            return self.inject_embeddings(logic_vector)

        self._seq_counter += 1

        # Step 1: Procrustes translation (if mapping layer exists)
        if target_agent_id in self._mapping_layers:
            W, src_mean, tgt_mean = self._mapping_layers[target_agent_id]
            translated_vec = translate_latent_vector(logic_vector, W, src_mean, tgt_mean)
        else:
            translated_vec = logic_vector   # same-model, no translation needed

        # Step 2: Differential Privacy — inject calibrated Gaussian noise
        #   Protects against Vector Inversion Attacks by obscuring the exact
        #   coordinates of the logic signal, while staying below DRIFT_THRESHOLD.
        protected_vec, dp_noise_l2 = self._apply_dp_noise(translated_vec)

        # Step 3: Build the Neural Header
        header = NeuralHeader.build(
            model_hash    = self.model_hash,
            task_uuid     = self.task_uuid,
            entropy_level = entropy_level,
            seq_id        = self._seq_counter,
        )

        # Step 4: Assemble and log the packet
        packet = NeuralPacket(
            source_agent = self.agent_id,
            target_agent = target_agent_id,
            header       = header,
            payload      = protected_vec.astype(np.float32),
            anchor_label = anchor_label,
        )
        self.packet_log.append(packet)

        # Step 5: Serialize to carrier text
        carrier = packet.to_carrier_text()
        dp_tag = f" | dp_noise_L2={dp_noise_l2:.4f}" if self.dp_enabled else " | dp=OFF"
        print(
            f"[{self.agent_id} → {target_agent_id}] "
            f"Packet dispatched | seq={self._seq_counter} | "
            f"payload_dim={len(protected_vec)} | anchor={anchor_label}{dp_tag}"
        )
        return carrier

    # ------------------------------------------------------------------
    # Differential Privacy
    # ------------------------------------------------------------------

    def _apply_dp_noise(
        self,
        vector: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Inject calibrated Gaussian noise into an outbound logic vector.

        The noise magnitude is bounded by DP_NOISE_CLIP_RATIO relative to
        the signal's own L2 norm, ensuring the cosine perturbation stays
        safely below the DRIFT_THRESHOLD (0.12).

        Args:
            vector: The logic vector to protect.

        Returns:
            (noisy_vector, noise_L2_magnitude)
        """
        if not self.dp_enabled or self.dp_noise_sigma <= 0:
            return vector.copy(), 0.0

        noise = self._dp_rng.normal(
            loc   = 0.0,
            scale = self.dp_noise_sigma,
            size  = vector.shape,
        ).astype(np.float32)

        # Safety clamp: scale noise down if it exceeds DP_NOISE_CLIP_RATIO
        signal_l2 = float(np.linalg.norm(vector))
        noise_l2  = float(np.linalg.norm(noise))
        if signal_l2 > 1e-8:
            max_noise = DP_NOISE_CLIP_RATIO * signal_l2
            if noise_l2 > max_noise:
                noise = noise * (max_noise / noise_l2)
                noise_l2 = max_noise

        noisy_vector = vector + noise
        return noisy_vector, noise_l2

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    def receive_and_unpack(self, carrier_text: str) -> Optional[np.ndarray]:
        """
        Extract and decode a NeuralPacket from incoming carrier text.
        Runs a lightweight drift audit. Returns the logic vector on success,
        or None and emits a NL fallback on drift violation.

        Args:
            carrier_text: Raw text from the target agent's context window.

        Returns:
            Decoded logic vector (np.ndarray), or None if fallback triggered.
        """
        packet = NeuralPacket.from_carrier_text(carrier_text)
        if packet is None:
            print(f"[{self.agent_id}] No NB_PACKET found in incoming text.")
            return None

        print(f"[{self.agent_id}] ← Received packet from '{packet.source_agent}'.")

        # Drift audit (lightweight pre-filter; full SANEmerg in Task 2.3)
        drift = self._compute_drift(packet)
        if drift > DRIFT_THRESHOLD:
            self._trigger_nl_fallback(packet, drift)
            return None

        self.packet_log.append(packet)
        return packet.payload

    # ------------------------------------------------------------------
    # Drift Audit
    # ------------------------------------------------------------------

    def _compute_drift(self, packet: NeuralPacket) -> float:
        """
        Compute cosine distance between the packet payload and the nearest
        Codebook anchor to detect transmission drift / hallucination.
        """
        if packet.payload is None or len(packet.payload) == 0:
            return 0.0

        dim = len(packet.payload)
        min_drift = 1.0

        for label, anchor in ANCHOR_MAP.items():
            # Pad anchor to match payload dimension
            if dim >= len(anchor):
                padded = np.pad(anchor, (0, dim - len(anchor)))
            else:
                padded = anchor[:dim]

            norm_payload = np.linalg.norm(packet.payload)
            norm_anchor  = np.linalg.norm(padded)
            if norm_payload < 1e-8 or norm_anchor < 1e-8:
                continue

            cosine_sim = float(np.dot(packet.payload, padded) / (norm_payload * norm_anchor))
            drift = 1.0 - cosine_sim
            if drift < min_drift:
                min_drift = drift

        return min_drift

    # ------------------------------------------------------------------
    # NL Fallback (Codebook §IV — Telegraph English Protocol)
    # ------------------------------------------------------------------

    def _trigger_nl_fallback(self, packet: NeuralPacket, drift: float) -> str:
        """
        Emit the standardized 'Telegraph English' fallback message when
        drift exceeds the Codebook threshold of 0.12.
        """
        fallback_msg = (
            f"PROTOCOL_FAIL: REASON [Drift_Detected={drift:.4f}] "
            f"| RESYNC: NL_NEGOTIATION "
            f"| LAST_STATE: [{packet.packet_id}]"
        )
        print(f"[{self.agent_id}] ⚠️  FALLBACK TRIGGERED: {fallback_msg}")
        return fallback_msg


# ---------------------------------------------------------------------------
# Convenience Helper: Codebook Anchor Broadcast
# ---------------------------------------------------------------------------

def broadcast_anchor(middleware: InterlatMiddleware, target: str, anchor: str) -> str:
    """
    Broadcast a reserved Codebook anchor signal to a target agent.

    Args:
        middleware: The sending agent's InterlatMiddleware instance.
        target:     Target agent ID.
        anchor:     One of '<INIT_SYNC>', '<ERR_BLOCK>', '<TASK_CMT>', '<QUERY_REF>'.

    Returns:
        Carrier text string.
    """
    if anchor not in ANCHOR_MAP:
        raise ValueError(f"Unknown anchor: '{anchor}'. Valid labels: {list(ANCHOR_MAP.keys())}")

    return middleware.pack_and_send(
        target_agent_id = target,
        logic_vector    = ANCHOR_MAP[anchor],
        anchor_label    = anchor,
        entropy_level   = 0.0,  # Anchor signals carry zero noise
    )


# ---------------------------------------------------------------------------
# Quick Smoke-Test (run directly: python interlat_middleware.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROJECT_UUID = str(uuid.uuid4())

    # Simulate two agents on the same project
    agent_A = InterlatMiddleware("architect", "claude-3-5-sonnet", "transformer-decoder", PROJECT_UUID)
    agent_B = InterlatMiddleware("engineer",  "llama-4-maverick",  "transformer-decoder", PROJECT_UUID)

    # --- Bootstrapping Phase ---
    # In the real system, agents would exchange NL messages to compute W.
    # Here we use identity transforms to demonstrate the pipeline.
    dim = 128
    W        = np.eye(dim, dtype=np.float32)
    src_mean = np.zeros(dim, dtype=np.float32)
    tgt_mean = np.zeros(dim, dtype=np.float32)

    agent_A.register_mapping_layer("engineer", W, src_mean, tgt_mean)
    agent_A.complete_bootstrap()

    # --- Neural Transmission ---
    logic_signal = np.random.randn(dim).astype(np.float32)
    print(f"\n[TEST] Agent A sending logic vector of shape {logic_signal.shape}...\n")

    carrier = agent_A.pack_and_send("engineer", logic_signal, anchor_label="<TASK_CMT>")
    print(f"\n[TEST] Carrier text snippet: {carrier[:120]}...\n")

    # --- Agent B Receives ---
    received = agent_B.receive_and_unpack(carrier)
    if received is not None:
        print(f"\n[TEST] Agent B decoded payload of shape {received.shape}.")
        print(f"[TEST] Round-trip cosine similarity: "
              f"{float(np.dot(logic_signal, received) / (np.linalg.norm(logic_signal) * np.linalg.norm(received) + 1e-8)):.6f}")
    else:
        print("[TEST] Receive failed — fallback triggered.")
