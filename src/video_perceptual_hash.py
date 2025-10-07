"""Video perceptual hashing for compression-resistant video fingerprinting."""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
from pathlib import Path


class VideoPerceptualHasher:
    """Computes robust perceptual hashes from video frames."""
    
    def __init__(self, keyframe_interval: float = 2.0):
        """Initialize video hasher.
        
        Args:
            keyframe_interval: Time interval between keyframes in seconds
        """
        self.keyframe_interval = keyframe_interval
    
    def extract_keyframes(self, video_path: Path) -> List[Tuple[int, np.ndarray]]:
        """Extract keyframes from video at regular intervals."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 30.0  # Default fallback
        
        frame_interval = int(fps * self.keyframe_interval)
        keyframes = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keyframes at intervals
            if frame_idx % frame_interval == 0:
                keyframes.append((frame_idx, frame))
            
            frame_idx += 1
        
        cap.release()
        return keyframes
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract compression-resistant features from a frame."""
        features_list = []
        
        # Convert to grayscale and HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Edge features (structural information)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_stats = [
            np.mean(edges),
            np.std(edges),
            edge_density
        ]
        features_list.extend(edge_stats)
        
        # 2. Color histogram features (HSV space, more robust)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        
        h_hist_norm = cv2.normalize(h_hist, h_hist).flatten()
        s_hist_norm = cv2.normalize(s_hist, s_hist).flatten()
        
        features_list.extend(h_hist_norm)
        features_list.extend(s_hist_norm)
        
        # 3. Difference hash (dHash) - perceptual hash
        # Resize to 9x8 for horizontal differences
        resized = cv2.resize(gray, (9, 8))
        diff = resized[:, 1:] > resized[:, :-1]
        dhash = diff.flatten().astype(np.float32)
        features_list.extend(dhash)
        
        # 4. Texture features (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_stats = [
            np.mean(np.abs(laplacian)),
            np.std(laplacian),
            np.var(laplacian)
        ]
        features_list.extend(texture_stats)
        
        # 5. Brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        features_list.extend([brightness, contrast])
        
        return np.array(features_list)
    
    def compute_perceptual_hash(self, frame: np.ndarray) -> bytes:
        """Compute perceptual hash from frame.
        
        Returns the binary features directly (not SHA-256 hashed).
        This allows proper Hamming distance comparison.
        """
        # Extract features
        features = self.extract_features(frame)
        
        # Quantize features using median threshold
        median_val = np.median(features)
        binary_features = (features > median_val).astype(np.uint8)
        
        # Pack bits into bytes for storage efficiency
        # 96 bits -> 12 bytes
        packed_bits = np.packbits(binary_features)
        
        return packed_bits.tobytes()
    
    def compute_keyframe_hashes(self, video_path: Path) -> List[Tuple[int, bytes, float]]:
        """Compute perceptual hashes for all keyframes.
        
        Returns:
            List of (frame_index, hash, timestamp) tuples
        """
        keyframes = self.extract_keyframes(video_path)
        
        # Get video properties for timestamp calculation
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        cap.release()
        
        results = []
        for frame_idx, frame in keyframes:
            phash = self.compute_perceptual_hash(frame)
            timestamp = frame_idx / fps
            results.append((frame_idx, phash, timestamp))
        
        return results
    
    def hash_similarity(self, hash1: bytes, hash2: bytes) -> float:
        """Compute similarity between two hashes using Hamming distance."""
        # Unpack bytes back to bits
        arr1 = np.unpackbits(np.frombuffer(hash1, dtype=np.uint8))
        arr2 = np.unpackbits(np.frombuffer(hash2, dtype=np.uint8))
        
        # Hamming distance (count differing bits)
        hamming_dist = np.sum(arr1 != arr2)
        max_dist = len(arr1)  # Total number of bits
        
        return 1.0 - (hamming_dist / max_dist)
    
    def classify_similarity(self, similarity: float) -> Dict[str, Any]:
        """Classify similarity level for video frames."""
        if similarity >= 0.90:
            return {
                "level": "EXACT",
                "verdict": "AUTHENTIC",
                "confidence": "very_high",
                "description": "Identical or minimal processing"
            }
        elif similarity >= 0.75:
            return {
                "level": "HIGH",
                "verdict": "LIKELY_AUTHENTIC",
                "confidence": "high",
                "description": "Minor compression or quality loss"
            }
        elif similarity >= 0.55:
            return {
                "level": "MEDIUM",
                "verdict": "POSSIBLY_AUTHENTIC",
                "confidence": "medium",
                "description": "Significant compression or processing"
            }
        else:
            return {
                "level": "LOW",
                "verdict": "NOT_AUTHENTIC",
                "confidence": "low",
                "description": "Content likely tampered or different"
            }

