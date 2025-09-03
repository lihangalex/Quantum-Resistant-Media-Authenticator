"""Media canonicalization module for consistent preprocessing."""

import numpy as np
import librosa
import soundfile as sf
import warnings
from typing import Tuple, Optional
from pathlib import Path

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')


class MediaCanonicalizer:
    """Canonicalizes audio/video media for consistent processing."""
    
    def __init__(self, target_sr: int = 44100, target_channels: int = 1):
        """Initialize canonicalizer with target parameters."""
        self.target_sr = target_sr
        self.target_channels = target_channels
    
    def canonicalize_audio(self, input_path: Path, output_path: Optional[Path] = None, 
                          preserve_lsb: bool = False) -> Tuple[np.ndarray, int]:
        """Canonicalize audio file to standard format."""
        # Load audio with librosa for robust format support
        audio, sr = librosa.load(str(input_path), sr=self.target_sr, mono=(self.target_channels == 1))
        
        if preserve_lsb:
            # LSB-preserving canonicalization: minimal processing to preserve steganographic data
            # Only apply essential normalization without aggressive filtering
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95  # Gentle normalization to prevent clipping
        else:
            # Full canonicalization for perceptual hashing
            # Normalize audio to prevent clipping and ensure consistent levels
            audio = librosa.util.normalize(audio, norm=np.inf, axis=None)
            
            # Apply pre-emphasis filter to enhance perceptual features
            audio = librosa.effects.preemphasis(audio, coef=0.97)
            
            # Apply gentle low-pass filter to remove high-frequency noise
            audio = librosa.effects.trim(audio, top_db=20)[0]  # Remove silence
        
        # Ensure consistent length padding if needed
        if len(audio) < self.target_sr:  # Less than 1 second
            audio = np.pad(audio, (0, self.target_sr - len(audio)), mode='constant')
        
        # Save canonicalized version if output path provided
        if output_path:
            sf.write(str(output_path), audio, self.target_sr)
        
        return audio, self.target_sr
    
    def get_audio_windows(self, audio: np.ndarray, window_size: float = 1.0, 
                         overlap: float = 0.5) -> list[np.ndarray]:
        """Split audio into overlapping windows."""
        window_samples = int(window_size * self.target_sr)
        hop_samples = int(window_samples * (1 - overlap))
        
        windows = []
        start = 0
        
        while start + window_samples <= len(audio):
            window = audio[start:start + window_samples]
            windows.append(window)
            start += hop_samples
        
        # Handle remaining samples in final window
        if start < len(audio):
            remaining = audio[start:]
            # Pad to window size
            padded = np.pad(remaining, (0, window_samples - len(remaining)), mode='constant')
            windows.append(padded)
        
        return windows
