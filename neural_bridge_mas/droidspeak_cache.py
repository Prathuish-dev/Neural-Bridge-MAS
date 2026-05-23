"""
droidspeak_cache.py
===================
Neural-Bridge MAS — Phase 2, Task 2.2
Agent 2 (Engineer) — Extended from Agent 1's scaffold

The DroidSpeak Shared K-V Cache: a holographic prefix cache that allows
multiple agents to share a single pinned memory partition.

Background (from AIM.md §4.B):
  Using the DroidSpeak method, all agents share a single "prefix cache."
  If Agent A reads a 50MB documentation file, that information is "pinned"
  so Agent B and C can access it instantly without spending a single token
  to re-read it.

Architecture:
  - HolographicKVCache   : Agent 1's original simple string-key store (kept).
  - CacheEntry           : Rich record with seq_id, key/value vectors, LRU metadata.
  - CachePartition       : Per-project (Task UUID) memory region.
  - DroidSpeakCache      : Full orchestrator — cosine search, LRU eviction,
                           pinning, snapshots, and NeuralPacket log integration.
"""

import numpy as np
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARTITION_CAPACITY = 1024   # Max entries per project partition
DEFAULT_TOP_K              = 3      # Default nearest-neighbor count
MAX_GLOBAL_ENTRIES         = 8192   # Absolute ceiling across all partitions


# ---------------------------------------------------------------------------
# Original Agent-1 HolographicKVCache (kept for compatibility)
# ---------------------------------------------------------------------------

class HolographicKVCache:
    def __init__(self, cache_size_mb: int = 1024):
        """
        Initializes the DroidSpeak Shared K-V Cache.
        This acts as the shared GPU memory partition where agents can 'pin'
        massive documents or context states for instantaneous shared access
        without spending tokens to re-read them.
        """
        self.max_size = cache_size_mb
        self.memory_partition: Dict[str, np.ndarray] = {}
        self.access_logs = []

    def pin_context(self, agent_id: str, context_key: str, neural_state: np.ndarray) -> bool:
        """
        Allows an agent to 'pin' a read document or state to the shared cache.
        """
        self.memory_partition[context_key] = neural_state
        print(f"[CACHE] Agent {agent_id} pinned neural state '{context_key}' to shared memory.")
        self.access_logs.append(f"PIN: {agent_id} -> {context_key}")
        return True

    def retrieve_context(self, agent_id: str, context_key: str) -> Optional[np.ndarray]:
        """
        Allows another agent to instantly access a pinned neural state,
        bypassing the need to process the original text.
        """
        if context_key in self.memory_partition:
            print(f"[CACHE] Agent {agent_id} successfully retrieved '{context_key}'. Token cost: 0.")
            self.access_logs.append(f"READ: {agent_id} <- {context_key}")
            return self.memory_partition[context_key]
        else:
            print(f"[CACHE ERR] Agent {agent_id} attempted to read missing key '{context_key}'.")
            return None

    def flush_partition(self, task_uuid: str):
        """
        Clears the cache for a specific task to prevent Context Bleed across sessions.
        """
        self.memory_partition.clear()
        print(f"[CACHE] Partition flushed for Task UUID: {task_uuid}")


# ---------------------------------------------------------------------------
# Agent-2 Extension: Vector-Search K-V Cache
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """
    A rich key-value record in the DroidSpeak semantic cache.

    'key'   — high-entropy vector encoding the semantic identity of the knowledge.
    'value' — latent payload (e.g., compressed document embedding, agent reasoning state).
    """
    entry_id:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    seq_id:       int   = 0        # Aligns with Neural Header Temporal Marker (DIM 96-127)
    written_by:   str   = ""       # Agent ID that created this entry
    timestamp:    float = field(default_factory=time.time)
    key:          Optional[np.ndarray] = None
    value:        Optional[np.ndarray] = None
    is_pinned:    bool  = False    # Pinned entries survive LRU eviction
    access_count: int   = 0


class CachePartition:
    """
    A named memory region for one project (Task UUID).
    Maintains insertion order for LRU eviction via OrderedDict.
    """
    def __init__(self, task_uuid: str, capacity: int = DEFAULT_PARTITION_CAPACITY):
        self.task_uuid  = task_uuid
        self.capacity   = capacity
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._seq = 0

    def next_seq(self) -> int:
        s = self._seq; self._seq += 1; return s

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity


class DroidSpeakCache:
    """
    Full Neural-Bridge holographic K-V cache orchestrator.

    Provides:
      - write()        — store a (key, value) vector pair in a partition.
      - query()        — cosine-similarity nearest-neighbor search.
      - pin_entry()    — mark an entry immune to eviction.
      - flush()        — clear an entire project partition.
      - snapshot()     — diagnostic summary for benchmarking (Task 3.1).
      - from_packet_log() — bulk-ingest packets from the Interlat session log.
    """

    def __init__(self, global_capacity: int = MAX_GLOBAL_ENTRIES):
        self.global_capacity = global_capacity
        self._partitions: Dict[str, CachePartition] = {}
        # Global LRU log: entry_id → task_uuid (oldest-first)
        self._lru_log: OrderedDict[str, str] = OrderedDict()
        self._total_entries: int = 0

    # ------------------------------------------------------------------
    # Partition Management
    # ------------------------------------------------------------------

    def init_partition(self, task_uuid: str, capacity: int = DEFAULT_PARTITION_CAPACITY):
        """
        Allocate a dedicated cache region for a project.
        Called automatically on the first write, or explicitly on <INIT_SYNC>.
        """
        if task_uuid not in self._partitions:
            self._partitions[task_uuid] = CachePartition(task_uuid, capacity)
            print(f"[DroidSpeak] Partition allocated | task={task_uuid[:8]}... | cap={capacity}")

    def _partition(self, task_uuid: str) -> CachePartition:
        self.init_partition(task_uuid)
        return self._partitions[task_uuid]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        task_uuid: str,
        key: np.ndarray,
        value: np.ndarray,
        agent_id: str,
        pin: bool = False,
    ) -> str:
        """
        Store a (key, value) pair. Evicts LRU entries if at capacity.

        Returns:
            The entry_id of the stored record.
        """
        if self._total_entries >= self.global_capacity:
            self._evict_global()

        p = self._partition(task_uuid)
        if p.is_full:
            self._evict_partition(p)

        entry = CacheEntry(
            seq_id    = p.next_seq(),
            written_by= agent_id,
            key       = key.astype(np.float32),
            value     = value.astype(np.float32),
            is_pinned = pin,
        )
        p.entries[entry.entry_id] = entry
        self._lru_log[entry.entry_id] = task_uuid
        self._total_entries += 1

        print(
            f"[DroidSpeak] Write | agent={agent_id} | task={task_uuid[:8]}... | "
            f"seq={entry.seq_id} | key_dim={len(key)} | val_dim={len(value)} | pinned={pin}"
        )
        return entry.entry_id

    # ------------------------------------------------------------------
    # Query (Semantic Nearest-Neighbor Search)
    # ------------------------------------------------------------------

    def query(
        self,
        task_uuid: str,
        query_vec: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        seq_min: Optional[int] = None,
        seq_max: Optional[int] = None,
    ) -> List[Tuple[float, CacheEntry]]:
        """
        Cosine-similarity search within a partition.
        Triggered by a <QUERY_REF> anchor from an agent.

        Args:
            task_uuid:  Project identifier.
            query_vec:  Query vector to match against stored keys.
            top_k:      Number of nearest entries to return.
            seq_min/max: Optional temporal filter on seq_id.

        Returns:
            List of (cosine_similarity, CacheEntry) sorted descending.
        """
        p = self._partition(task_uuid)
        if p.size == 0:
            print(f"[DroidSpeak] Query on empty partition | task={task_uuid[:8]}...")
            return []

        q = query_vec.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return []

        results: List[Tuple[float, CacheEntry]] = []

        for entry in p.entries.values():
            if seq_min is not None and entry.seq_id < seq_min: continue
            if seq_max is not None and entry.seq_id > seq_max: continue
            if entry.key is None: continue

            dim = max(len(q), len(entry.key))
            q_p = np.pad(q,          (0, dim - len(q)))          if len(q)          < dim else q
            k_p = np.pad(entry.key,  (0, dim - len(entry.key)))  if len(entry.key)  < dim else entry.key
            k_norm = np.linalg.norm(k_p)
            if k_norm < 1e-8: continue

            sim = float(np.dot(q_p, k_p) / (q_norm * k_norm))
            results.append((sim, entry))

            entry.access_count += 1
            self._lru_log.move_to_end(entry.entry_id)   # mark as recently used

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:top_k]
        if top:
            print(
                f"[DroidSpeak] Query | task={task_uuid[:8]}... | "
                f"hits={len(results)} | top_k={top_k} | best_sim={top[0][0]:.4f}"
            )
        return top

    # ------------------------------------------------------------------
    # Pinning
    # ------------------------------------------------------------------

    def pin_entry(self, task_uuid: str, entry_id: str):
        """Mark an entry as immune to LRU eviction."""
        p = self._partition(task_uuid)
        if entry_id in p.entries:
            p.entries[entry_id].is_pinned = True
            print(f"[DroidSpeak] Pinned | entry={entry_id[:8]}...")

    def unpin_entry(self, task_uuid: str, entry_id: str):
        """Allow a previously pinned entry to be evicted."""
        p = self._partition(task_uuid)
        if entry_id in p.entries:
            p.entries[entry_id].is_pinned = False

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self, task_uuid: str):
        """
        Completely clear a project partition to prevent Context Bleed
        between sessions (equivalent to HolographicKVCache.flush_partition).
        """
        if task_uuid in self._partitions:
            p = self._partitions[task_uuid]
            for eid in list(p.entries.keys()):
                self._lru_log.pop(eid, None)
                self._total_entries -= 1
            p.entries.clear()
            print(f"[DroidSpeak] Partition flushed | task={task_uuid[:8]}...")

    # ------------------------------------------------------------------
    # Integration: Ingest from Interlat Packet Log
    # ------------------------------------------------------------------

    def from_packet_log(self, task_uuid: str, packet_log: list, agent_id: str = "system"):
        """
        Bulk-ingest NeuralPackets from an Interlat session log into the cache.
        Each packet's (header_vec, payload) becomes a (key, value) cache entry.

        Args:
            task_uuid:   Project identifier.
            packet_log:  List of NeuralPacket objects from InterlatMiddleware.
            agent_id:    Label for the ingesting agent.
        """
        ingested = 0
        for packet in packet_log:
            if packet.header is not None and packet.payload is not None:
                key   = packet.header.to_vector()
                value = packet.payload
                self.write(task_uuid, key, value, agent_id=agent_id)
                ingested += 1
        print(f"[DroidSpeak] Ingested {ingested} packets from Interlat log.")

    # ------------------------------------------------------------------
    # Eviction (LRU)
    # ------------------------------------------------------------------

    def _evict_partition(self, p: CachePartition) -> bool:
        """Evict the oldest non-pinned entry from a specific partition."""
        for eid, entry in list(p.entries.items()):
            if not entry.is_pinned:
                del p.entries[eid]
                self._lru_log.pop(eid, None)
                self._total_entries -= 1
                print(f"[DroidSpeak] Evicted (partition) | entry={eid[:8]}...")
                return True
        return False

    def _evict_global(self) -> bool:
        """Evict the globally least-recently-used non-pinned entry."""
        for eid, task_uuid in list(self._lru_log.items()):
            p = self._partitions.get(task_uuid)
            if p and eid in p.entries and not p.entries[eid].is_pinned:
                del p.entries[eid]
                del self._lru_log[eid]
                self._total_entries -= 1
                print(f"[DroidSpeak] Evicted (global LRU) | entry={eid[:8]}...")
                return True
        return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a utilization summary (consumed by Task 3.1 benchmarker)."""
        out: Dict[str, Any] = {
            "total_entries":   self._total_entries,
            "global_capacity": self.global_capacity,
            "utilization_pct": round(100.0 * self._total_entries / self.global_capacity, 2),
            "partitions": {}
        }
        for tuuid, p in self._partitions.items():
            pinned = sum(1 for e in p.entries.values() if e.is_pinned)
            out["partitions"][tuuid] = {
                "entries":   p.size,
                "capacity":  p.capacity,
                "pinned":    pinned,
                "evictable": p.size - pinned,
                "next_seq":  p._seq,
            }
        return out


# ---------------------------------------------------------------------------
# Smoke-Test (run directly: python droidspeak_cache.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROJECT_UUID = str(uuid.uuid4())
    cache = DroidSpeakCache(global_capacity=512)

    # Agent A pins a large documentation embedding
    doc_key   = np.random.randn(256).astype(np.float32)
    doc_value = np.random.randn(256).astype(np.float32)
    eid = cache.write(PROJECT_UUID, doc_key, doc_value, agent_id="architect", pin=True)
    print(f"\n[TEST] Pinned doc entry: {eid[:8]}...\n")

    # Agent B queries with a slightly noisy version of the same key
    noisy_query = doc_key + np.random.randn(256).astype(np.float32) * 0.05
    results = cache.query(PROJECT_UUID, noisy_query, top_k=1)

    if results:
        sim, entry = results[0]
        print(f"\n[TEST] Best match | sim={sim:.6f} | seq={entry.seq_id} | pinned={entry.is_pinned}")

    print(f"\n[TEST] Snapshot: {cache.snapshot()}")

    # Demonstrate legacy API compatibility
    legacy = HolographicKVCache()
    legacy.pin_context("architect", "doc_chunk_1", doc_value)
    retrieved = legacy.retrieve_context("engineer", "doc_chunk_1")
    print(f"\n[TEST] Legacy retrieve shape: {retrieved.shape}")
