"""Perceptual hashing module for robust audio fingerprinting with speech optimization."""

import numpy as np
import librosa
from typing import List, Tuple, Dict, Any
import warnings


class PerceptualHasher:
    """Computes robust perceptual hashes from audio windows, optimized for speech authentication."""
    
    def __init__(self, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512, 
                 speech_optimized: bool = True, polarity_threshold: float = 0.60,
                 polarity_penalty: float = 0.15):
        """Initialize hasher with spectral analysis parameters.
        
        Args:
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for spectral analysis
            speech_optimized: Enable speech-specific robust features
            polarity_threshold: Threshold for polarity match (0.0-1.0). Below this,
                               audio is considered phase-inverted. Default 0.60.
            polarity_penalty: Penalty factor applied to inverted audio (0.0-1.0).
                             Lower = more aggressive rejection. Default 0.15.
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.speech_optimized = speech_optimized
        self.polarity_threshold = polarity_threshold
        self.polarity_penalty = polarity_penalty
        self.polarity_bits = 32  # Number of bits dedicated to polarity features
    
    def extract_features(self, audio_window: np.ndarray, sr: int) -> np.ndarray:
        """Extract robust perceptual features from audio window."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.speech_optimized:
                return self._extract_speech_features(audio_window, sr)
            else:
                return self._extract_music_features(audio_window, sr)
    
    def _extract_speech_features(self, audio_window: np.ndarray, sr: int) -> np.ndarray:
        """Extract speech-optimized robust features."""
        features_list = []
        
        # 1. Fundamental frequency (F0) patterns - robust pitch estimation
        try:
            f0 = librosa.yin(audio_window, fmin=50, fmax=400, sr=sr)
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames
            if len(f0_clean) > 0:
                f0_stats = [
                    np.mean(f0_clean), np.std(f0_clean), np.median(f0_clean),
                    np.percentile(f0_clean, 25), np.percentile(f0_clean, 75)
                ]
            else:
                f0_stats = [0.0] * 5
            features_list.extend(f0_stats)
        except Exception:
            # F0 extraction failed (e.g., very short window, silence)
            features_list.extend([0.0] * 5)
        
        # 2. Formant-like features using MFCC (vocal tract characteristics)
        mfcc = librosa.feature.mfcc(
            y=audio_window, sr=sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        # Focus on first few MFCCs which capture formant-like information
        mfcc_robust = mfcc[:8]  # First 8 coefficients are most robust
        mfcc_stats = np.hstack([
            np.mean(mfcc_robust, axis=1),
            np.std(mfcc_robust, axis=1),
            np.median(mfcc_robust, axis=1)
        ])
        features_list.extend(mfcc_stats)
        
        # 3. Spectral envelope characteristics (voice quality)
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_window, sr=sr, hop_length=self.hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_window, sr=sr, hop_length=self.hop_length, roll_percent=0.85
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_window, sr=sr, hop_length=self.hop_length
        )
        
        spectral_features = np.vstack([spectral_centroids, spectral_rolloff, spectral_bandwidth])
        spectral_stats = np.hstack([
            np.mean(spectral_features, axis=1),
            np.std(spectral_features, axis=1)
        ])
        features_list.extend(spectral_stats)
        
        # 4. Voice activity and energy patterns
        rms_energy = librosa.feature.rms(y=audio_window, hop_length=self.hop_length)
        zcr = librosa.feature.zero_crossing_rate(audio_window, hop_length=self.hop_length)
        
        energy_features = np.vstack([rms_energy, zcr])
        energy_stats = np.hstack([
            np.mean(energy_features, axis=1),
            np.std(energy_features, axis=1)
        ])
        features_list.extend(energy_stats)
        
        # 5. Temporal characteristics (speaking rhythm)
        # Onset detection for speech rhythm
        onset_frames = librosa.onset.onset_detect(
            y=audio_window, sr=sr, hop_length=self.hop_length, units='frames'
        )
        if len(onset_frames) > 1:
            onset_intervals = np.diff(onset_frames) * self.hop_length / sr
            rhythm_stats = [
                np.mean(onset_intervals), np.std(onset_intervals),
                len(onset_frames) / (len(audio_window) / sr)  # onset rate
            ]
        else:
            rhythm_stats = [0.0, 0.0, 0.0]
        features_list.extend(rhythm_stats)
        
        return np.array(features_list)
    
    def _extract_polarity_features(self, audio_window: np.ndarray, sr: int) -> np.ndarray:
        """Extract phase-sensitive polarity features to detect phase inversion.
        
        These features are compression-resistant but flip with phase inversion.
        Returns 32 features (4 bytes when quantized) that capture waveform polarity.
        Higher feature count gives polarity adequate weight in final hash.
        """
        features = []
        
        # 1. Waveform mean (sign flips with inversion)
        waveform_mean = np.mean(audio_window)
        features.extend([waveform_mean] * 4)  # Replicate for weight
        
        # 2. Temporal skewness (asymmetry flips with inversion)
        if len(audio_window) > 2:
            skewness = np.mean(((audio_window - np.mean(audio_window)) / (np.std(audio_window) + 1e-8)) ** 3)
        else:
            skewness = 0.0
        features.extend([skewness] * 4)  # Replicate for weight
        
        # 3. First-moment weighted features (sign-sensitive)
        time_weights = np.linspace(-1, 1, len(audio_window))
        weighted_mean = np.mean(audio_window * time_weights)
        features.extend([weighted_mean] * 4)
        
        # 4. Sign of peak amplitude
        max_amp = np.max(np.abs(audio_window))
        if max_amp > 0:
            peak_sign = 1.0 if audio_window[np.argmax(np.abs(audio_window))] > 0 else -1.0
        else:
            peak_sign = 0.0
        features.extend([peak_sign] * 2)
        
        # 5. Zero-crossing pattern features (directional)
        signs = np.sign(audio_window)
        sign_changes = np.diff(signs)
        pos_to_neg = np.sum(sign_changes < 0)
        neg_to_pos = np.sum(sign_changes > 0)
        crossing_asymmetry = (pos_to_neg - neg_to_pos) / (len(audio_window) + 1)
        features.extend([crossing_asymmetry] * 4)
        
        # 6. Front-weighted average (captures initial phase)
        window_fade = np.exp(-np.arange(len(audio_window)) / (len(audio_window) * 0.1))
        front_weighted = np.sum(audio_window * window_fade) / np.sum(window_fade)
        features.extend([front_weighted] * 4)
        
        # 7. Energy-weighted polarity
        energy = audio_window ** 2
        energy_weighted_sign = np.sum(audio_window * energy) / (np.sum(energy) + 1e-8)
        features.extend([energy_weighted_sign] * 4)
        
        # 8. Back-weighted average (captures final phase)
        back_fade = np.exp(-np.arange(len(audio_window))[::-1] / (len(audio_window) * 0.1))
        back_weighted = np.sum(audio_window * back_fade) / np.sum(back_fade)
        features.extend([back_weighted] * 4)
        
        # 9. Quartile-based signed features
        q1_mask = (np.arange(len(audio_window)) < len(audio_window) // 4)
        q1_mean = np.mean(audio_window[q1_mask]) if np.any(q1_mask) else 0.0
        features.extend([q1_mean] * 2)
        
        return np.array(features)
    
    def _extract_music_features(self, audio_window: np.ndarray, sr: int) -> np.ndarray:
        """Extract music-optimized robust features (original approach)."""
        # Extract MFCC features (robust to compression)
        mfcc = librosa.feature.mfcc(
            y=audio_window, sr=sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_window, sr=sr, hop_length=self.hop_length
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_window, sr=sr, hop_length=self.hop_length
        )
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio_window, hop_length=self.hop_length
        )
        
        # Extract chromagram (pitch class profile)
        chroma = librosa.feature.chroma_stft(
            y=audio_window, sr=sr, hop_length=self.hop_length
        )
        
        # Combine all features
        features = np.vstack([
            mfcc,
            spectral_centroids,
            spectral_rolloff, 
            zero_crossing_rate,
            chroma
        ])
        
        # Compute statistics to create fixed-size representation
        feature_stats = np.hstack([
            np.mean(features, axis=1),
            np.std(features, axis=1),
            np.median(features, axis=1),
            np.percentile(features, 25, axis=1),
            np.percentile(features, 75, axis=1)
        ])
        
        return feature_stats
    
    def compute_perceptual_hash(self, audio_window: np.ndarray, sr: int) -> bytes:
        """Compute perceptual hash from audio window.
        
        Returns the binary features directly (not SHA-256 hashed).
        This allows proper Hamming distance comparison for similarity.
        Includes both compression-resistant features and phase-sensitive polarity features.
        """
        # Extract main perceptual features (phase-invariant)
        features = self.extract_features(audio_window, sr)
        
        # Extract polarity features (phase-sensitive) to detect inversions
        polarity_features = self._extract_polarity_features(audio_window, sr)
        
        # Quantize main features for robustness
        # Use median as threshold for binary quantization
        median_val = np.median(features)
        binary_features = (features > median_val).astype(np.uint8)
        
        # Quantize polarity features separately (sign-based)
        # These capture whether features are positive or negative (flips with inversion)
        binary_polarity = (polarity_features > 0).astype(np.uint8)
        
        # Combine: main features + polarity signature
        combined_features = np.concatenate([binary_features, binary_polarity])
        
        # Pack bits into bytes for storage efficiency
        packed_bits = np.packbits(combined_features)
        
        return packed_bits.tobytes()
    
    def compute_window_hashes(self, windows: List[np.ndarray], sr: int) -> List[bytes]:
        """Compute perceptual hashes for all windows."""
        return [self.compute_perceptual_hash(window, sr) for window in windows]
    
    def hash_similarity(self, hash1: bytes, hash2: bytes) -> float:
        """Compute similarity between two hashes with two-stage polarity check.
        
        Stage 1: Fast polarity check on dedicated polarity bits
                 If polarity mismatch detected → aggressive penalty
        Stage 2: Full perceptual hash comparison using Hamming distance
        
        This approach dramatically improves phase inversion detection while
        maintaining compression resistance.
        
        Args:
            hash1: First perceptual hash (bytes)
            hash2: Second perceptual hash (bytes)
            
        Returns:
            Similarity score 0.0-1.0, where:
            - 1.0 = identical
            - 0.85-0.95 = same content, compressed
            - 0.60-0.80 = possibly authentic, heavily modified
            - 0.0-0.60 = tampered or inverted
        """
        # Unpack bytes back to bits
        arr1 = np.unpackbits(np.frombuffer(hash1, dtype=np.uint8))
        arr2 = np.unpackbits(np.frombuffer(hash2, dtype=np.uint8))
        
        # Ensure arrays have same length
        if len(arr1) != len(arr2):
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]
        
        # Stage 1: Polarity check (last N bits are polarity features)
        # The polarity bits are appended last in compute_perceptual_hash
        if len(arr1) >= self.polarity_bits:
            polarity1 = arr1[-self.polarity_bits:]
            polarity2 = arr2[-self.polarity_bits:]
            
            # Check how many polarity bits match
            polarity_matches = np.sum(polarity1 == polarity2)
            polarity_similarity = polarity_matches / self.polarity_bits
            
            # If polarity is very different, likely phase-inverted
            if polarity_similarity < self.polarity_threshold:
                # Apply aggressive penalty to inverted audio
                # This drops similarity from ~65% to ~5-10%
                full_hamming_dist = np.sum(arr1 != arr2)
                full_similarity = 1.0 - (full_hamming_dist / len(arr1))
                penalized_similarity = full_similarity * self.polarity_penalty
                
                return penalized_similarity
        
        # Stage 2: Full perceptual hash comparison
        # Used when polarity check passes (likely authentic or compressed)
        hamming_dist = np.sum(arr1 != arr2)
        similarity = 1.0 - (hamming_dist / len(arr1))
        
        return similarity
    
    def analyze_polarity_mismatch(self, hash1: bytes, hash2: bytes) -> Dict[str, Any]:
        """Detailed polarity analysis for debugging and reporting.
        
        Returns dict with:
            - polarity_similarity: float (0.0-1.0)
            - is_likely_inverted: bool
            - polarity_bits_flipped: int
            - full_similarity: float (without penalty)
            - penalized_similarity: float (with penalty if inverted)
        """
        arr1 = np.unpackbits(np.frombuffer(hash1, dtype=np.uint8))
        arr2 = np.unpackbits(np.frombuffer(hash2, dtype=np.uint8))
        
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
        
        result = {
            'polarity_similarity': 1.0,
            'is_likely_inverted': False,
            'polarity_bits_flipped': 0,
            'full_similarity': 1.0 - (np.sum(arr1 != arr2) / len(arr1)),
            'penalized_similarity': None
        }
        
        if len(arr1) >= self.polarity_bits:
            polarity1 = arr1[-self.polarity_bits:]
            polarity2 = arr2[-self.polarity_bits:]
            
            polarity_matches = np.sum(polarity1 == polarity2)
            result['polarity_similarity'] = polarity_matches / self.polarity_bits
            result['polarity_bits_flipped'] = self.polarity_bits - polarity_matches
            result['is_likely_inverted'] = result['polarity_similarity'] < self.polarity_threshold
            
            if result['is_likely_inverted']:
                result['penalized_similarity'] = result['full_similarity'] * self.polarity_penalty
        
        return result
    
    def classify_similarity(self, similarity: float, content_type: str = "speech") -> Dict[str, Any]:
        """Classify similarity level based on content type and thresholds."""
        if content_type == "speech":
            # Speech-optimized thresholds
            if similarity >= 0.95:
                return {
                    "level": "EXACT",
                    "verdict": "AUTHENTIC",
                    "confidence": "very_high",
                    "description": "Identical or unmodified content"
                }
            elif similarity >= 0.80:
                return {
                    "level": "HIGH",
                    "verdict": "LIKELY_AUTHENTIC", 
                    "confidence": "high",
                    "description": "Minor compression or processing"
                }
            elif similarity >= 0.60:
                return {
                    "level": "MEDIUM",
                    "verdict": "POSSIBLY_AUTHENTIC",
                    "confidence": "medium", 
                    "description": "Significant compression or quality loss"
                }
            else:
                return {
                    "level": "LOW",
                    "verdict": "NOT_AUTHENTIC",
                    "confidence": "low",
                    "description": "Content likely tampered or different"
                }
        else:
            # Music/general audio thresholds (more tolerant)
            if similarity >= 0.90:
                return {
                    "level": "EXACT",
                    "verdict": "AUTHENTIC",
                    "confidence": "very_high",
                    "description": "Identical or unmodified content"
                }
            elif similarity >= 0.70:
                return {
                    "level": "HIGH", 
                    "verdict": "LIKELY_AUTHENTIC",
                    "confidence": "high",
                    "description": "Minor compression or processing"
                }
            elif similarity >= 0.50:
                return {
                    "level": "MEDIUM",
                    "verdict": "POSSIBLY_AUTHENTIC",
                    "confidence": "medium",
                    "description": "Significant compression or quality loss"
                }
            else:
                return {
                    "level": "LOW",
                    "verdict": "NOT_AUTHENTIC", 
                    "confidence": "low",
                    "description": "Content likely tampered or different"
                }
    
    def get_similarity_thresholds(self, content_type: str = "speech") -> Dict[str, float]:
        """Get similarity thresholds for different content types."""
        if content_type == "speech":
            return {
                "exact": 0.95,
                "high": 0.80,
                "medium": 0.60,
                "low": 0.0
            }
        else:
            return {
                "exact": 0.90,
                "high": 0.70, 
                "medium": 0.50,
                "low": 0.0
            }
