import numpy as np

# Threshold defined in Neural Codebook v1.0
MAX_DRIFT_THRESHOLD = 0.12

def calculate_cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates the cosine distance between two vectors.
    Used by the Audit/Verification Partition to check for semantic drift.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # Cosine similarity is between -1 and 1
    # Cosine distance = 1 - Cosine Similarity
    cosine_sim = dot_product / (norm_a * norm_b)
    return 1.0 - cosine_sim

def check_audit_partition(source_audit_vector: np.ndarray, received_audit_vector: np.ndarray) -> bool:
    """
    Verifies if the neural signal has degraded beyond the acceptable threshold
    during the cross-model latent translation.
    
    Returns True if the signal is safe, False if drift is detected.
    """
    drift = calculate_cosine_distance(source_audit_vector, received_audit_vector)
    
    if drift > MAX_DRIFT_THRESHOLD:
        return False
    return True

def detect_err_block(received_vector: np.ndarray) -> bool:
    """
    Checks if the received vector matches the symbolic <ERR_BLOCK> reserved tensor.
    Defined in Codebook as [-1.0, -1.0, ... 1.0].
    """
    # Assuming the first two elements are -1.0 as a simplified heuristic
    # or checking for the general pattern.
    if len(received_vector) >= 3 and received_vector[0] < -0.9 and received_vector[1] < -0.9 and received_vector[-1] > 0.9:
        return True
    return False

def trigger_telegraph_fallback(reason: str, last_state_id: str) -> str:
    """
    Triggers the Telegraph English fallback protocol when neural transmission fails.
    This forces the agents back into natural language negotiation to resync their weights.
    """
    fallback_message = f"PROTOCOL_FAIL: REASON [{reason}] | RESYNC: NL_NEGOTIATION | LAST_STATE: [{last_state_id}]"
    print(f"[SYSTEM ALERT] Neural Bridge Collapsed. Initiating Fallback:\n>> {fallback_message}")
    
    # In a live system, this function would interrupt the vector injection and inject
    # this structured text string into the standard API context window instead.
    return fallback_message

def monitor_transmission(source_vector: np.ndarray, received_vector: np.ndarray, state_id: str):
    """
    The main monitoring loop for Agent 3 (QA). 
    """
    if detect_err_block(received_vector):
        return trigger_telegraph_fallback("Critical_Logic_Failure_ERR_BLOCK", state_id)

    is_safe = check_audit_partition(source_vector, received_vector)
    
    if not is_safe:
        return trigger_telegraph_fallback("Drift_Detected_Above_0.12", state_id)
    
    return "TRANSMISSION_OK"
