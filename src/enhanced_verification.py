"""Enhanced verification with Content ID-inspired techniques."""

import numpy as np
from typing import List, Tuple, Dict


class EnhancedVerifier:
    """Improved verification using multi-scale features and adaptive thresholds."""
    
    def __init__(self, base_hasher):
        """Initialize with base perceptual hasher."""
        self.hasher = base_hasher
    
    def estimate_signal_quality(self, audio_window: np.ndarray) -> Dict[str, float]:
        """Estimate audio signal quality metrics."""
        # RMS energy
        rms = np.sqrt(np.mean(audio_window**2))
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio_window)) != 0)
        zcr = zero_crossings / len(audio_window)
        
        # Dynamic range
        dynamic_range = np.max(np.abs(audio_window)) - np.min(np.abs(audio_window))
        
        # Estimate SNR (simple method)
        # High energy + low ZCR = clean signal
        # Low energy or high ZCR = noisy signal
        snr_estimate = rms / (zcr + 0.01)  # Rough SNR proxy
        
        return {
            'rms': rms,
            'zcr': zcr,
            'dynamic_range': dynamic_range,
            'snr_estimate': snr_estimate
        }
    
    def get_adaptive_threshold(self, quality_metrics: Dict[str, float]) -> float:
        """Determine adaptive threshold based on signal quality."""
        rms = quality_metrics['rms']
        zcr = quality_metrics['zcr']
        snr = quality_metrics['snr_estimate']
        
        # Very high quality signal (strong amplitude, low ZCR, high SNR)
        if rms > 0.1 and zcr < 0.05 and snr > 2.0:
            return 0.85  # Stricter threshold
        
        # High quality (clean audio)
        elif rms > 0.01 and zcr < 0.15 and snr > 0.5:
            return 0.80  # Default threshold
        
        # Medium quality (some noise or low volume)
        elif zcr < 0.35:
            return 0.75  # More lenient
        
        # Low quality or very noisy
        else:
            return 0.70  # Most lenient for extreme cases
    
    def verify_temporal_consistency(self, similarities: List[float], 
                                   thresholds: List[float]) -> Dict[str, any]:
        """Check temporal consistency of matches."""
        # Count consecutive matches
        consecutive_matches = 0
        max_consecutive = 0
        valid_count = 0
        
        for sim, threshold in zip(similarities, thresholds):
            if sim >= threshold:
                consecutive_matches += 1
                valid_count += 1
                max_consecutive = max(max_consecutive, consecutive_matches)
            else:
                consecutive_matches = 0
        
        total_windows = len(similarities)
        valid_ratio = valid_count / total_windows if total_windows > 0 else 0
        consistency_ratio = max_consecutive / total_windows if total_windows > 0 else 0
        
        return {
            'valid_count': valid_count,
            'total_windows': total_windows,
            'valid_ratio': valid_ratio,
            'max_consecutive': max_consecutive,
            'consistency_ratio': consistency_ratio
        }
    
    def compute_weighted_similarity(self, hash1: bytes, hash2: bytes) -> float:
        """Compute similarity with feature weighting.
        
        Note: The base hasher already has two-stage polarity detection,
        so we just use the standard similarity here. This method is kept
        for future enhancements with more sophisticated weighting.
        """
        # Use the base hasher's similarity which already has two-stage detection
        return self.hasher.hash_similarity(hash1, hash2)
    
    def extract_multiscale_features(self, audio_window: np.ndarray, sr: int) -> bytes:
        """Extract features at multiple time scales."""
        features_all = []
        
        # Define scales (in seconds)
        scales = [0.5, 2.0]  # Short and medium term
        
        for scale in scales:
            scale_samples = int(scale * sr)
            
            # If window is long enough for this scale
            if len(audio_window) >= scale_samples:
                window_slice = audio_window[:scale_samples]
                
                # Extract features at this scale
                scale_features = self.hasher.extract_features(window_slice, sr)
                
                # Take most informative features (first 50%)
                n_features = len(scale_features) // 2
                features_all.extend(scale_features[:n_features])
        
        # Combine all scales
        combined = np.array(features_all)
        
        # Quantize
        median = np.median(combined)
        binary = (combined > median).astype(np.uint8)
        
        # Add polarity features
        polarity = self.hasher._extract_polarity_features(audio_window, sr)
        binary_polarity = (polarity > 0).astype(np.uint8)
        
        # Combine and pack
        final_features = np.concatenate([binary, binary_polarity])
        return np.packbits(final_features).tobytes()
    
    def verify_enhanced(self, current_windows: List[np.ndarray], 
                       stored_hashes: List[bytes],
                       sr: int) -> Dict[str, any]:
        """Enhanced verification with all improvements."""
        
        similarities = []
        thresholds = []
        weighted_similarities = []
        
        for current_window, stored_hash in zip(current_windows, stored_hashes):
            # Estimate quality
            quality = self.estimate_signal_quality(current_window)
            threshold = self.get_adaptive_threshold(quality)
            thresholds.append(threshold)
            
            # Compute hash (could use multiscale or standard)
            current_hash = self.hasher.compute_perceptual_hash(current_window, sr)
            
            # Standard similarity
            sim = self.hasher.hash_similarity(current_hash, stored_hash)
            similarities.append(sim)
            
            # Weighted similarity
            weighted_sim = self.compute_weighted_similarity(current_hash, stored_hash)
            weighted_similarities.append(weighted_sim)
        
        # Temporal consistency check
        consistency = self.verify_temporal_consistency(similarities, thresholds)
        
        # Compute metrics
        avg_similarity = np.mean(similarities)
        avg_weighted_similarity = np.mean(weighted_similarities)
        valid_ratio = consistency['valid_ratio']
        consistency_ratio = consistency['consistency_ratio']
        
        # Determine confidence level
        if (valid_ratio >= 0.90 and consistency_ratio >= 0.50 and 
            avg_weighted_similarity >= 0.90):
            confidence = "VERY_HIGH"
            status = "Authentic - Very High Confidence"
            result = "GREEN"
        elif (valid_ratio >= 0.85 and consistency_ratio >= 0.40 and 
              avg_weighted_similarity >= 0.85):
            confidence = "HIGH"
            status = "Authentic - High Confidence"
            result = "GREEN"
        elif (valid_ratio >= 0.80 and consistency_ratio >= 0.30 and 
              avg_weighted_similarity >= 0.80):
            confidence = "MEDIUM"
            status = "Authentic - Medium Confidence"
            result = "GREEN"
        elif valid_ratio >= 0.70 and avg_weighted_similarity >= 0.70:
            confidence = "LOW"
            status = "Possibly Authentic - Low Confidence"
            result = "AMBER"
        else:
            confidence = "REJECTED"
            status = "Not Authentic - Failed Verification"
            result = "RED"
        
        return {
            'result': result,
            'status': status,
            'confidence': confidence,
            'avg_similarity': avg_similarity,
            'avg_weighted_similarity': avg_weighted_similarity,
            'valid_ratio': valid_ratio,
            'consistency_ratio': consistency_ratio,
            'max_consecutive': consistency['max_consecutive'],
            'total_windows': consistency['total_windows'],
            'similarities': similarities,
            'thresholds': thresholds
        }
