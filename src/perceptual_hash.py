"""Perceptual hashing module for robust audio fingerprinting with speech optimization."""

import numpy as np
import librosa
from typing import List, Tuple, Dict, Any
import hashlib
import warnings


class PerceptualHasher:
    """Computes robust perceptual hashes from audio windows, optimized for speech authentication."""
    
    def __init__(self, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512, 
                 speech_optimized: bool = True):
        """Initialize hasher with spectral analysis parameters.
        
        Args:
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for spectral analysis
            speech_optimized: Enable speech-specific robust features
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.speech_optimized = speech_optimized
    
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
        except:
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
        """Compute perceptual hash from audio window."""
        # Extract features
        features = self.extract_features(audio_window, sr)
        
        # Quantize features for robustness
        # Use median as threshold for binary quantization
        median_val = np.median(features)
        binary_features = (features > median_val).astype(np.uint8)
        
        # Create hash from binary features
        feature_bytes = binary_features.tobytes()
        
        # Use SHA-256 for cryptographic strength
        hash_obj = hashlib.sha256(feature_bytes)
        return hash_obj.digest()
    
    def compute_window_hashes(self, windows: List[np.ndarray], sr: int) -> List[bytes]:
        """Compute perceptual hashes for all windows."""
        return [self.compute_perceptual_hash(window, sr) for window in windows]
    
    def hash_similarity(self, hash1: bytes, hash2: bytes) -> float:
        """Compute similarity between two hashes (Hamming distance based)."""
        # Convert to binary arrays for comparison
        arr1 = np.frombuffer(hash1, dtype=np.uint8)
        arr2 = np.frombuffer(hash2, dtype=np.uint8)
        
        # Compute Hamming distance
        hamming_dist = np.sum(arr1 != arr2)
        max_dist = len(arr1) * 8  # 8 bits per byte
        
        # Return similarity (1 - normalized hamming distance)
        return 1.0 - (hamming_dist / max_dist)
    
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
