"""Audio steganography for embedding signed messages into media."""

import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
import base64
import zlib
from scipy.fft import dct, idct


class AudioEmbedder:
    """Embeds signed messages into audio using DCT-based steganography."""
    
    def __init__(self, frame_size: int = 2048, embed_strength: float = 0.4):
        """Initialize DCT-based embedder.
        
        Args:
            frame_size: Size of DCT frames (should be power of 2)
            embed_strength: Embedding strength (0.05-0.3, higher = more robust but less inaudible)
        """
        self.frame_size = frame_size
        self.embed_strength = embed_strength
        self.magic_header = b"QRMN_DCT"  # Quantum Resistant Media Notarizer DCT
        
        # DCT coefficient indices for embedding (balanced range for capacity vs robustness)
        # Use stable mid-frequency coefficients
        self.embed_indices = list(range(frame_size // 16, frame_size // 6))
    
    def _prepare_payload(self, messages: List[Dict[str, Any]], 
                        signatures: List[Dict[str, bytes]]) -> bytes:
        """Prepare payload for embedding."""
        # Combine messages and signatures
        payload_data = {
            'messages': messages,
            'signatures': [
                {scheme: sig.hex() for scheme, sig in sig_dict.items()}
                for sig_dict in signatures
            ]
        }
        
        # Serialize to JSON
        json_data = json.dumps(payload_data, separators=(',', ':')).encode('utf-8')
        
        # Compress data
        compressed_data = zlib.compress(json_data, level=9)
        
        # Add length prefix and magic header
        length = len(compressed_data)
        payload = self.magic_header + length.to_bytes(4, 'big') + compressed_data
        
        return payload
    
    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to list of bits."""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes."""
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)
        
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte |= bits[i + j] << (7 - j)
            bytes_data.append(byte)
        
        return bytes(bytes_data)
    
    def embed_messages(self, audio: np.ndarray, messages: List[Dict[str, Any]], 
                      signatures: List[Dict[str, bytes]]) -> np.ndarray:
        """Embed signed messages using DCT steganography."""
        # Prepare payload
        payload = self._prepare_payload(messages, signatures)
        payload_bits = self._bytes_to_bits(payload)
        
        # Pad audio to multiple of frame_size
        padded_length = ((len(audio) + self.frame_size - 1) // self.frame_size) * self.frame_size
        padded_audio = np.pad(audio, (0, padded_length - len(audio)), mode='constant')
        
        # Calculate capacity
        num_frames = len(padded_audio) // self.frame_size
        capacity_bits = num_frames * len(self.embed_indices)
        
        if len(payload_bits) > capacity_bits:
            raise ValueError(f"Payload too large: {len(payload_bits)} bits > {capacity_bits} capacity")
        
        # Embed bits into DCT coefficients
        embedded_audio = padded_audio.copy()
        bit_index = 0
        
        for frame_idx in range(num_frames):
            if bit_index >= len(payload_bits):
                break
                
            # Extract frame
            start = frame_idx * self.frame_size
            end = start + self.frame_size
            frame = embedded_audio[start:end]
            
            # Compute DCT
            dct_coeffs = dct(frame, type=2, norm='ortho')
            
            # Embed bits in mid-frequency coefficients
            for coeff_idx in self.embed_indices:
                if bit_index >= len(payload_bits):
                    break
                
                bit_value = payload_bits[bit_index]
                coeff = dct_coeffs[coeff_idx]
                
                if abs(coeff) > 1e-4:  # Higher threshold for stability
                    # Very robust quantization for complex audio
                    base_step = 0.1  # Large base quantization step
                    quant_step = max(abs(coeff) * self.embed_strength, base_step)
                    
                    # Simple, robust embedding: force coefficient to specific values
                    if bit_value == 1:
                        # For bit 1: make coefficient clearly positive and odd-quantized
                        target = quant_step * (2 * abs(round(coeff / (2 * quant_step))) + 1)
                        if coeff < 0:
                            target = -target
                    else:
                        # For bit 0: make coefficient clearly even-quantized
                        target = quant_step * (2 * abs(round(coeff / (2 * quant_step))))
                        if coeff < 0 and target != 0:
                            target = -target
                    
                    dct_coeffs[coeff_idx] = target
                
                bit_index += 1
            
            # Inverse DCT to get modified frame
            modified_frame = idct(dct_coeffs, type=2, norm='ortho')
            embedded_audio[start:end] = modified_frame
        
        # Remove padding and return
        return embedded_audio[:len(audio)]
    
    def extract_messages(self, audio: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, bytes]]]:
        """Extract signed messages using DCT steganography."""
        # Pad audio to multiple of frame_size
        padded_length = ((len(audio) + self.frame_size - 1) // self.frame_size) * self.frame_size
        padded_audio = np.pad(audio, (0, padded_length - len(audio)), mode='constant')
        
        num_frames = len(padded_audio) // self.frame_size
        
        # Extract bits from DCT coefficients
        extracted_bits = []
        
        for frame_idx in range(num_frames):
            start = frame_idx * self.frame_size
            end = start + self.frame_size
            frame = padded_audio[start:end]
            
            # Compute DCT
            dct_coeffs = dct(frame, type=2, norm='ortho')
            
            # Extract bits from mid-frequency coefficients
            for coeff_idx in self.embed_indices:
                coeff = dct_coeffs[coeff_idx]
                
                if abs(coeff) > 1e-4:  # Match embedding threshold
                    # Determine quantization step (match embedding)
                    base_step = 0.1
                    quant_step = max(abs(coeff) * self.embed_strength, base_step)
                    
                    # Extract bit using the same logic as embedding
                    quant_level = abs(round(coeff / (2 * quant_step)))
                    if abs(coeff) < quant_step * 0.5:
                        bit_value = 0  # Very small coefficients are 0
                    else:
                        # Check if it's odd-quantized (bit 1) or even-quantized (bit 0)
                        bit_value = 1 if (quant_level % 2 == 1) or (abs(coeff) % (2 * quant_step) > quant_step) else 0
                    
                    extracted_bits.append(bit_value)
                else:
                    extracted_bits.append(0)
        
        # Find magic header
        header_bits = self._bytes_to_bits(self.magic_header)
        header_found = False
        header_start = 0
        
        for start_idx in range(len(extracted_bits) - len(header_bits) + 1):
            if extracted_bits[start_idx:start_idx + len(header_bits)] == header_bits:
                header_found = True
                header_start = start_idx
                break
        
        if not header_found:
            raise ValueError("No embedded messages found")
        
        # Extract length (4 bytes = 32 bits)
        length_start = header_start + len(header_bits)
        if length_start + 32 > len(extracted_bits):
            raise ValueError("Incomplete length field")
        
        length_bits = extracted_bits[length_start:length_start + 32]
        length_bytes = self._bits_to_bytes(length_bits)
        payload_length = int.from_bytes(length_bytes, 'big')
        
        # Extract payload
        payload_start = length_start + 32
        if payload_start + payload_length * 8 > len(extracted_bits):
            raise ValueError("Incomplete payload")
        
        payload_bits = extracted_bits[payload_start:payload_start + payload_length * 8]
        payload_bytes = self._bits_to_bytes(payload_bits)
        
        try:
            decompressed_data = zlib.decompress(payload_bytes)
            payload_data = json.loads(decompressed_data.decode('utf-8'))
        except Exception as e:
            raise ValueError(f"Failed to decode payload: {e}")
        
        # Parse messages and signatures
        messages = payload_data['messages']
        signatures = [
            {scheme: bytes.fromhex(sig_hex) for scheme, sig_hex in sig_dict.items()}
            for sig_dict in payload_data['signatures']
        ]
        
        return messages, signatures
    
    def estimate_capacity(self, audio_length: int) -> int:
        """Estimate embedding capacity in bytes."""
        num_frames = audio_length // self.frame_size
        capacity_bits = num_frames * len(self.embed_indices)
        
        # Subtract header and length overhead
        overhead_bits = (len(self.magic_header) + 4) * 8  # Magic header + length field
        data_bits = capacity_bits - overhead_bits
        
        return max(0, data_bits // 8)
