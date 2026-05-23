import numpy as np

def compute_procrustes_alignment(source_embeddings: np.ndarray, target_embeddings: np.ndarray):
    """
    Computes the orthogonal transformation matrix to align a source latent space
    to a target latent space using Orthogonal Procrustes.
    
    This is necessary because different models (e.g., Claude vs Llama) have entirely
    different embedding/latent spaces. A concept vector in one space must be translated
    (rotated and scaled) to make sense in the other.
    
    Args:
        source_embeddings: Array of shape (N, D_source) representing anchor concepts.
        target_embeddings: Array of shape (N, D_target) representing the exact same concepts.
                           Note: D_source and D_target must be equalized/padded before this step
                           if they differ in dimension size.
                           
    Returns:
        W: The optimal rotation/orthogonal transformation matrix (D x D).
        source_mean: The translation offset for the source.
        target_mean: The translation offset for the target.
    """
    # 1. Center the embeddings to remove translation bias
    source_mean = np.mean(source_embeddings, axis=0)
    target_mean = np.mean(target_embeddings, axis=0)
    
    X_centered = source_embeddings - source_mean
    Y_centered = target_embeddings - target_mean
    
    # 2. Compute the cross-covariance matrix (X^T * Y)
    C = np.dot(X_centered.T, Y_centered)
    
    # 3. Perform Singular Value Decomposition (SVD)
    # This separates the cross-covariance into rotational components
    U, S, Vt = np.linalg.svd(C)
    
    # 4. Compute the optimal rotation matrix W
    W = np.dot(U, Vt)
    
    return W, source_mean, target_mean

def translate_latent_vector(vector: np.ndarray, W: np.ndarray, source_mean: np.ndarray, target_mean: np.ndarray) -> np.ndarray:
    """
    Translates a single high-entropy neural signal (vector) from the source agent's
    latent space into the target agent's latent space.
    
    Args:
        vector: The incoming latent vector.
        W: The pre-computed orthogonal transformation matrix.
        source_mean: The pre-computed source center.
        target_mean: The pre-computed target center.
        
    Returns:
        The aligned vector, ready to be injected into the target agent's context window.
    """
    # Apply translation to center, rotate via W, then translate to target center
    aligned_vector = np.dot((vector - source_mean), W) + target_mean
    return aligned_vector

# --- Bootstrapping Example Usage ---
# During the Bootstrapping Phase, agents will output standard embeddings for a shared 
# set of words (e.g., the Neural Codebook "Anchors"). 
# We capture these to compute 'W', locking in the mapping layer for the session.
