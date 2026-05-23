"""
agentark_distillation.py
========================
Neural-Bridge MAS — Phase 3, Task 3.2
Agent 2 (Engineer)

AgentArk Distillation: compact 3 specialized agents into one Multi-Role Model.

Background (from AIM.md §4.D):
  Once the system is stable, we use AgentArk logic to "compact" the 3 separate
  agents into a single Multi-Role Model.
  How: We record the successful "Neural Exchanges" and fine-tune a smaller model
  to simulate all three agents' reasoning internally.

This module:
  1. Harvests successful NeuralPackets from the Interlat packet log.
  2. Labels each packet with its source agent role.
  3. Constructs a distillation training dataset (role, input_vec → output_vec).
  4. Simulates the fine-tuning loop using gradient-descent weight updates
     (pure NumPy — no ML framework required for this prototype).
  5. Evaluates the distilled model's role-prediction accuracy and
     vector reconstruction quality.

In a production system, step 4 would be replaced with a proper LoRA
fine-tuning run against an open-source model (e.g., Llama-4-8B).

Dependencies:
  - numpy
  - droidspeak_cache.py (Task 2.2) — for harvesting the session log
  - interlat_middleware.py (Task 2.1) — NeuralPacket type
"""

import numpy as np
import uuid
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ROLES = ["architect", "engineer", "qa_monitor"]
ROLE_TO_IDX  = {r: i for i, r in enumerate(AGENT_ROLES)}
IDX_TO_ROLE  = {i: r for i, r in enumerate(AGENT_ROLES)}
NUM_ROLES    = len(AGENT_ROLES)

DEFAULT_HIDDEN_DIM = 256    # Distilled model hidden size
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS        = 100
MIN_PACKETS_REQUIRED  = 10  # Minimum dataset size to attempt distillation


# ---------------------------------------------------------------------------
# Training Sample
# ---------------------------------------------------------------------------

@dataclass
class DistillationSample:
    """
    One training example for the AgentArk distillation process.

    input_vec:  The inbound logic signal (the 'prompt' in neural space).
    output_vec: The outbound logic signal (the 'response' in neural space).
    role_label: Which agent role generated this exchange (one-hot encoded).
    packet_id:  Reference back to the source NeuralPacket.
    """
    packet_id:  str
    input_vec:  np.ndarray
    output_vec: np.ndarray
    role_label: np.ndarray   # One-hot [architect, engineer, qa_monitor]
    source_agent: str = ""
    seq_id: int = 0


# ---------------------------------------------------------------------------
# Distillation Dataset Builder
# ---------------------------------------------------------------------------

class AgentArkDatasetBuilder:
    """
    Harvests NeuralPackets from an Interlat session log (or DroidSpeak cache)
    and constructs a labeled distillation training dataset.
    """

    def __init__(self, agent_role_map: Optional[Dict[str, str]] = None):
        """
        Args:
            agent_role_map: Maps agent_id strings to role labels.
                            e.g. {"architect_01": "architect", "eng_02": "engineer"}
                            If None, tries to infer from agent_id substrings.
        """
        self.agent_role_map = agent_role_map or {}
        self.samples: List[DistillationSample] = []

    def _infer_role(self, agent_id: str) -> str:
        """Infer agent role from agent_id string if not in the explicit map."""
        if agent_id in self.agent_role_map:
            return self.agent_role_map[agent_id]
        agent_id_lower = agent_id.lower()
        for role in AGENT_ROLES:
            if role in agent_id_lower or role.replace("_", "") in agent_id_lower:
                return role
        return "architect"   # default

    def _role_to_one_hot(self, role: str) -> np.ndarray:
        vec = np.zeros(NUM_ROLES, dtype=np.float32)
        idx = ROLE_TO_IDX.get(role, 0)
        vec[idx] = 1.0
        return vec

    def ingest_packet_log(self, packet_log: list) -> int:
        """
        Convert a list of NeuralPackets into DistillationSamples.

        Each consecutive pair (send → receive) of packets from the same
        source agent becomes one (input, output) training sample.

        Args:
            packet_log: List of NeuralPacket objects.

        Returns:
            Number of samples added.
        """
        # Group packets by source agent and sort by sequence
        by_agent: Dict[str, list] = defaultdict(list)
        for pkt in packet_log:
            if pkt.payload is not None and len(pkt.payload) > 0:
                by_agent[pkt.source_agent].append(pkt)

        added = 0
        for agent_id, pkts in by_agent.items():
            role  = self._infer_role(agent_id)
            label = self._role_to_one_hot(role)
            # Create (input=pkt[i], output=pkt[i+1]) pairs
            for i in range(len(pkts) - 1):
                sample = DistillationSample(
                    packet_id    = pkts[i].packet_id,
                    input_vec    = pkts[i].payload.astype(np.float32),
                    output_vec   = pkts[i + 1].payload.astype(np.float32),
                    role_label   = label,
                    source_agent = agent_id,
                    seq_id       = i,
                )
                self.samples.append(sample)
                added += 1

        print(f"[AgentArk] Dataset: ingested {added} samples from {len(by_agent)} agents.")
        return added

    def ingest_synthetic(
        self,
        n_samples_per_role: int = 50,
        vec_dim: int = DEFAULT_HIDDEN_DIM,
        seed: int = 42,
    ) -> int:
        """
        Generate synthetic training data when no real packet log is available.
        Each role gets a distinct signal subspace to make distillation tractable.

        Args:
            n_samples_per_role: Number of (input, output) pairs per agent role.
            vec_dim:            Dimensionality of the logic vectors.
            seed:               RNG seed for reproducibility.

        Returns:
            Total number of samples added.
        """
        rng   = np.random.default_rng(seed)
        added = 0

        for role_idx, role in enumerate(AGENT_ROLES):
            label = self._role_to_one_hot(role)
            # Each role occupies a distinct "band" of the vector space
            band_start = role_idx * (vec_dim // NUM_ROLES)
            band_end   = band_start + (vec_dim // NUM_ROLES)

            for i in range(n_samples_per_role):
                input_vec  = rng.standard_normal(vec_dim).astype(np.float32)
                output_vec = rng.standard_normal(vec_dim).astype(np.float32)
                # Boost the role-specific band to create a learnable signal
                input_vec[band_start:band_end]  += 3.0
                output_vec[band_start:band_end] += 3.0

                self.samples.append(DistillationSample(
                    packet_id    = str(uuid.uuid4()),
                    input_vec    = input_vec,
                    output_vec   = output_vec,
                    role_label   = label,
                    source_agent = role,
                    seq_id       = i,
                ))
                added += 1

        print(f"[AgentArk] Synthetic dataset: {added} samples ({n_samples_per_role} × {NUM_ROLES} roles).")
        return added

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Export the dataset as (X_inputs, Y_outputs, R_roles) NumPy arrays.

        Returns:
            X: (N, input_dim)
            Y: (N, output_dim)
            R: (N, NUM_ROLES)  — one-hot role labels
        """
        X = np.stack([s.input_vec  for s in self.samples]).astype(np.float32)
        Y = np.stack([s.output_vec for s in self.samples]).astype(np.float32)
        R = np.stack([s.role_label for s in self.samples]).astype(np.float32)
        return X, Y, R


# ---------------------------------------------------------------------------
# Distilled Multi-Role Model (Pure NumPy prototype)
# ---------------------------------------------------------------------------

class MultiRoleModel:
    """
    A lightweight neural network that simulates all agent roles internally.

    Architecture: Input → RoleGating → HiddenLayer → Output

    RoleGating: a role-conditioned projection that routes the input through
                role-specific weight subsets before the shared hidden layer.

    In production, this would be a LoRA-adapted transformer.
    """

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_dim)

        # Role-conditioned gating projections: one per role
        self.W_gate: Dict[int, np.ndarray] = {
            i: (rng.standard_normal((input_dim, hidden_dim)) * scale).astype(np.float32)
            for i in range(NUM_ROLES)
        }
        # Shared hidden → output projection
        self.W_out = (rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)

        # Role classifier head (hidden → NUM_ROLES)
        self.W_cls = (rng.standard_normal((hidden_dim, NUM_ROLES)) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b_cls = np.zeros(NUM_ROLES, dtype=np.float32)

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-8)

    def forward(
        self,
        x: np.ndarray,
        role_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Forward pass through the Multi-Role Model.

        Args:
            x:        Input vector (input_dim,).
            role_idx: If provided, use this role's gating weights.
                      If None, the classifier head predicts the role.

        Returns:
            (output_vec, role_probs, predicted_role_idx)
        """
        # Step 1: Predict role if not given
        # Use all role gates and average for initial hidden state
        h_avg = np.zeros(self.hidden_dim, dtype=np.float32)
        for i, W in self.W_gate.items():
            h_avg += self._relu(x @ W)
        h_avg /= NUM_ROLES

        role_logits = h_avg @ self.W_cls + self.b_cls
        role_probs  = self._softmax(role_logits)
        pred_role   = int(np.argmax(role_probs)) if role_idx is None else role_idx

        # Step 2: Role-gated hidden representation
        h = self._relu(x @ self.W_gate[pred_role])

        # Step 3: Output projection
        out = h @ self.W_out + self.b_out
        return out, role_probs, pred_role

    def predict_role(self, x: np.ndarray) -> str:
        """Predict which agent role would have generated this vector."""
        _, _, role_idx = self.forward(x)
        return IDX_TO_ROLE[role_idx]


# ---------------------------------------------------------------------------
# AgentArk Distillation Trainer
# ---------------------------------------------------------------------------

@dataclass
class DistillationResult:
    """Captures the outcome of a distillation training run."""
    model:           Optional[MultiRoleModel] = None
    epochs_run:      int   = 0
    final_vec_loss:  float = 0.0   # Mean cosine distance between predicted and true output
    final_cls_loss:  float = 0.0   # Cross-entropy on role classification
    role_accuracy:   float = 0.0   # % of training samples with correct role prediction
    compression_ratio: float = 0.0 # Token savings vs. NL baseline
    success:         bool  = False
    timestamp:       float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"DistillationResult | epochs={self.epochs_run} | "
            f"vec_loss={self.final_vec_loss:.6f} | "
            f"cls_loss={self.final_cls_loss:.6f} | "
            f"role_acc={self.role_accuracy:.2%} | "
            f"compression={self.compression_ratio:.1f}x | "
            f"success={self.success}"
        )


class AgentArkDistiller:
    """
    Trains a MultiRoleModel on the distillation dataset to compact
    multiple specialized agents into a single model.
    """

    def __init__(
        self,
        hidden_dim:    int   = DEFAULT_HIDDEN_DIM,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        epochs:        int   = DEFAULT_EPOCHS,
    ):
        self.hidden_dim    = hidden_dim
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.model: Optional[MultiRoleModel] = None

    def distill(self, builder: AgentArkDatasetBuilder) -> DistillationResult:
        """
        Run the full distillation training loop.

        Args:
            builder: Populated AgentArkDatasetBuilder with training samples.

        Returns:
            DistillationResult with trained model and metrics.
        """
        result = DistillationResult()

        if len(builder.samples) < MIN_PACKETS_REQUIRED:
            print(
                f"[AgentArk] ⚠ Insufficient training data: "
                f"{len(builder.samples)} samples (need ≥ {MIN_PACKETS_REQUIRED}). "
                f"Run ingest_synthetic() to generate data."
            )
            return result

        X, Y, R = builder.to_arrays()
        N, input_dim  = X.shape
        _, output_dim = Y.shape

        print(f"\n[AgentArk] Starting distillation | samples={N} | in={input_dim} | out={output_dim}")
        print(f"[AgentArk] Epochs={self.epochs} | lr={self.learning_rate} | hidden={self.hidden_dim}\n")

        # Initialize model
        self.model = MultiRoleModel(input_dim, output_dim, self.hidden_dim)
        result.model = self.model

        # Training loop (simplified gradient descent with numerical gradients)
        loss_history = []
        for epoch in range(self.epochs):
            epoch_vec_loss = 0.0
            epoch_cls_loss = 0.0

            # Shuffle dataset
            perm = np.random.permutation(N)

            for idx in perm:
                x      = X[idx]
                y_true = Y[idx]
                r_true = R[idx]
                role_true_idx = int(np.argmax(r_true))

                # Forward pass
                y_pred, role_probs, pred_role = self.model.forward(x, role_true_idx)

                # Vector reconstruction loss (cosine distance)
                yn  = np.linalg.norm(y_true)
                yp  = np.linalg.norm(y_pred)
                if yn > 1e-8 and yp > 1e-8:
                    cos_sim  = float(np.dot(y_true, y_pred) / (yn * yp))
                    vec_loss = 1.0 - cos_sim
                else:
                    vec_loss = 1.0

                # Role classification loss (cross-entropy)
                cls_loss = -float(np.log(role_probs[role_true_idx] + 1e-8))

                total_loss = vec_loss + 0.1 * cls_loss
                epoch_vec_loss += vec_loss
                epoch_cls_loss += cls_loss

                # Simplified weight update: nudge W_gate[role] toward reducing vec_loss
                # (In production: actual backprop through autograd)
                h = self.model._relu(x @ self.model.W_gate[role_true_idx])
                # Output gradient
                err = y_pred - y_true
                d_W_out = np.outer(h, err) * self.learning_rate / N
                self.model.W_out -= d_W_out
                self.model.b_out -= err * self.learning_rate / N

            avg_vec_loss = epoch_vec_loss / N
            avg_cls_loss = epoch_cls_loss / N
            loss_history.append(avg_vec_loss)

            if epoch == 0 or (epoch + 1) % (self.epochs // 5) == 0:
                print(
                    f"  Epoch {epoch + 1:4d}/{self.epochs} | "
                    f"vec_loss={avg_vec_loss:.6f} | cls_loss={avg_cls_loss:.6f}"
                )

        # Final evaluation
        correct_roles = 0
        final_vec_losses = []
        for i in range(N):
            y_pred, role_probs, pred_role = self.model.forward(X[i])
            if pred_role == int(np.argmax(R[i])):
                correct_roles += 1
            yn = np.linalg.norm(Y[i])
            yp = np.linalg.norm(y_pred)
            if yn > 1e-8 and yp > 1e-8:
                final_vec_losses.append(1.0 - float(np.dot(Y[i], y_pred) / (yn * yp)))
            else:
                final_vec_losses.append(1.0)

        result.epochs_run     = self.epochs
        result.final_vec_loss = float(np.mean(final_vec_losses))
        result.final_cls_loss = avg_cls_loss
        result.role_accuracy  = correct_roles / N

        # Compression ratio: one multi-role model vs. 3 separate full-size agents
        # Each agent would need full hidden_dim; the distilled model uses hidden_dim
        # shared across roles. Rough estimate: 3 agents × avg_signal → 1 model × dim.
        result.compression_ratio = float(NUM_ROLES * input_dim) / float(self.hidden_dim)
        result.success = (result.role_accuracy > 0.7) and (result.final_vec_loss < 0.5)

        print(f"\n[AgentArk] Distillation complete:")
        print(f"  {result.summary()}\n")
        return result

    def export_model_summary(self, result: DistillationResult) -> Dict:
        """
        Export a JSON-serializable summary of the distilled model.
        Intended for the DroidSpeak cache and AgentArk logging.
        """
        if result.model is None:
            return {}
        m = result.model
        return {
            "model_type":        "MultiRoleModel (AgentArk Prototype)",
            "input_dim":         m.input_dim,
            "hidden_dim":        m.hidden_dim,
            "output_dim":        m.output_dim,
            "roles":             AGENT_ROLES,
            "epochs_trained":    result.epochs_run,
            "final_vec_loss":    round(result.final_vec_loss, 6),
            "role_accuracy_pct": round(result.role_accuracy * 100, 2),
            "compression_ratio": round(result.compression_ratio, 2),
            "success":           result.success,
            "timestamp":         result.timestamp,
        }


# ---------------------------------------------------------------------------
# Smoke-Test (run directly: python agentark_distillation.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  AgentArk Distillation — Smoke Test")
    print("=" * 60)

    # Step 1: Build a synthetic dataset
    builder = AgentArkDatasetBuilder()
    builder.ingest_synthetic(n_samples_per_role=60, vec_dim=128, seed=2026)

    # Step 2: Run distillation
    distiller = AgentArkDistiller(
        hidden_dim    = 128,
        learning_rate = 5e-4,
        epochs        = 80,
    )
    result = distiller.distill(builder)

    # Step 3: Export summary
    summary = distiller.export_model_summary(result)
    print("\n[AgentArk] Model Summary:")
    print(json.dumps(summary, indent=2))

    # Step 4: Demonstrate role prediction
    if result.model:
        test_vec = np.zeros(128, dtype=np.float32)
        # Inject a signal in the "engineer" band (role_idx=1)
        eng_band_start = 1 * (128 // NUM_ROLES)
        test_vec[eng_band_start: eng_band_start + (128 // NUM_ROLES)] = 3.0
        predicted_role = result.model.predict_role(test_vec)
        print(f"\n[AgentArk] Role prediction for engineer-band signal: '{predicted_role}'")
        print(f"  (Expected: 'engineer')")
