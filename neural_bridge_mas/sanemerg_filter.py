import numpy as np

class SANEmergFilter:
    def __init__(self, mode: str = "Aggressive"):
        """
        Initializes the Semantic Importance (SANEmerg) Filter.
        Acts as the 'Neural Secretary' to strip away politeness, formatting, 
        and other high-dimensional noise, leaving only the dense Logic Signal.
        """
        self.mode = mode
        # The filter weights are calibrated during the Bootstrapping Phase
        self.importance_weights = None

    def calibrate_weights(self, baseline_signals: np.ndarray):
        """
        Calibrates the filter during the NL Negotiation Phase.
        Learns which dimensions correlate with pure logic vs. conversational fluff.
        """
        # Mock calibration: analyzing variance to find "noisy" dimensions
        # In a real setup, this involves analyzing the difference between standard text 
        # embeddings and dense logic embeddings established during the handshake.
        variance = np.var(baseline_signals, axis=0)
        
        # Dimensions with stable variance across logic statements get higher weight
        self.importance_weights = 1.0 / (variance + 1e-6)
        
        # Normalize weights between 0 and 1
        self.importance_weights /= np.max(self.importance_weights)
        print(f"[SANEmerg] Filter weights calibrated based on bootstrapping data.")

    def apply_filter(self, neural_payload: np.ndarray) -> np.ndarray:
        """
        Strips away noise dimensions based on the calibrated weights and current mode.
        """
        if self.importance_weights is None:
            raise ValueError("SANEmerg Filter must be calibrated before use.")

        filtered_signal = np.copy(neural_payload)
        
        # Apply the learned soft-weights
        filtered_signal = filtered_signal * self.importance_weights

        # In "Aggressive" mode (recommended for the first 50 cycles), 
        # we strictly zero out any dimension that falls below a high importance threshold.
        if self.mode == "Aggressive":
            threshold = np.percentile(self.importance_weights, 75) # Keep only top 25% dims
            mask = self.importance_weights >= threshold
            filtered_signal = filtered_signal * mask
            stripped_count = np.sum(~mask)
            print(f"[SANEmerg] Aggressive Filter applied. Stripped {stripped_count} noisy dimensions.")
        else:
            print(f"[SANEmerg] Standard soft-weight filter applied.")
            
        return filtered_signal
