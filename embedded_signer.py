#!/usr/bin/env python3
"""Embedded signature system that creates self-contained signed media files."""

import sys
import json
import hashlib
import warnings
import tempfile
import subprocess
import base64
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from crypto_signing import HybridSigner, KeyManager
from hashchain import HashChain
from perceptual_hash import PerceptualHasher
from canonicalization import MediaCanonicalizer

# Suppress warnings
warnings.filterwarnings('ignore')

def create_signed_media(input_file: str, private_key_file: str, output_file: str = None):
    """Create a self-contained signed media file with embedded signatures."""
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_file is None:
        output_file = str(input_path.stem) + "_signed" + input_path.suffix
    
    output_path = Path(output_file)
    
    print(f"Creating signed media file: {input_file} → {output_file}")
    
    # Initialize components
    canonicalizer = MediaCanonicalizer()
    hasher = PerceptualHasher(speech_optimized=True)  # Enable speech optimization
    signer = HybridSigner()
    
    # Load private keys
    try:
        private_keys = KeyManager.load_keys(private_key_file)
        signer.load_private_keys(private_keys)
        print(f"Loaded private keys from: {private_key_file}")
    except Exception as e:
        raise ValueError(f"Failed to load private keys: {e}")
    
    # Process audio for signature generation
    try:
        canon_audio, canon_sr = canonicalizer.canonicalize_audio(input_path)
        windows = canonicalizer.get_audio_windows(canon_audio, window_size=1.0, overlap=0.5)
        print(f"Processed {len(windows)} audio windows for signing")
    except Exception as e:
        raise ValueError(f"Failed to process audio: {e}")
    
    # Create signatures
    chain = HashChain()
    messages = []
    signatures = []
    
    print("Generating cryptographic signatures...")
    for i, window in enumerate(windows):
        try:
            perceptual_hash = hasher.compute_perceptual_hash(window, canon_sr)
            
            metadata = {
                'method': 'embedded',
                'window_index': i,
                'timestamp': i * 0.5,
                'version': '1.0'
            }
            
            chain_message = chain.add_window(i, perceptual_hash, metadata)
            message_bytes = chain_message.serialize_for_signing()
            signature = signer.sign_message(message_bytes)
            
            messages.append(chain_message.to_dict())
            signatures.append({k: v.hex() for k, v in signature.items()})
            
            if (i + 1) % 10 == 0:
                print(f"  Signed {i + 1}/{len(windows)} windows")
                
        except Exception as e:
            print(f"Warning: Failed to sign window {i}: {e}")
            continue
    
    if not messages:
        raise ValueError("No windows were successfully signed")
    
    # Create signature payload
    signature_data = {
        'version': '1.0',
        'method': 'embedded',
        'total_windows': len(messages),
        'messages': messages,
        'signatures': signatures,
        'audio_info': {
            'sample_rate': canon_sr,
            'duration': len(canon_audio) / canon_sr,
            'channels': 1
        }
    }
    
    # Encode signature data for embedding
    signature_json = json.dumps(signature_data, separators=(',', ':'))  # Compact JSON
    signature_b64 = base64.b64encode(signature_json.encode()).decode()
    
    print(f"Signature payload: {len(signature_b64)} characters")
    
    # Embed signatures in media file using ffmpeg metadata
    # For MP4, we need to use MKV container which supports arbitrary metadata
    try:
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # Convert to MKV for metadata support, then back to MP4
            temp_mkv = tempfile.NamedTemporaryFile(suffix='.mkv', delete=False)
            temp_mkv.close()
            
            # First: Convert to MKV with metadata
            cmd1 = [
                'ffmpeg', '-y',
                '-i', str(input_path),
                '-c', 'copy',
                '-metadata', f'qrmn_auth={signature_b64}',
                '-metadata', f'qrmn_version=1.0',
                '-metadata', f'qrmn_windows={len(messages)}',
                temp_mkv.name
            ]
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True)
            if result1.returncode != 0:
                raise ValueError(f"ffmpeg MKV conversion failed: {result1.stderr}")
            
            # Second: Convert back to MP4 (metadata will be preserved in a compatible way)
            cmd2 = [
                'ffmpeg', '-y',
                '-i', temp_mkv.name,
                '-c', 'copy',
                '-movflags', '+use_metadata_tags',  # Force metadata preservation
                str(output_path)
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            
            # Clean up temp file
            Path(temp_mkv.name).unlink(missing_ok=True)
            
            if result2.returncode != 0:
                raise ValueError(f"ffmpeg MP4 conversion failed: {result2.stderr}")
                
        elif input_path.suffix.lower() in ['.mkv']:
            # MKV file - directly embed metadata
            cmd = [
                'ffmpeg', '-y',
                '-i', str(input_path),
                '-c', 'copy',
                '-metadata', f'qrmn_auth={signature_b64}',
                '-metadata', f'qrmn_version=1.0',
                '-metadata', f'qrmn_windows={len(messages)}',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError(f"ffmpeg MKV embedding failed: {result.stderr}")
        else:
            # Audio file - embed in metadata
            cmd = [
                'ffmpeg', '-y',
                '-i', str(input_path),
                '-c', 'copy',
                '-metadata', f'qrmn_auth={signature_b64}',
                '-metadata', f'qrmn_version=1.0',
                '-metadata', f'qrmn_windows={len(messages)}',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError(f"ffmpeg audio embedding failed: {result.stderr}")
        
        print(f"✅ Signed media file created: {output_file}")
        print(f"   Original size: {input_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"   Signed size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"   Signature overhead: {(output_path.stat().st_size - input_path.stat().st_size) / 1024:.1f} KB")
        print(f"   Windows signed: {len(messages)}")
        
        return str(output_path)
        
    except Exception as e:
        raise ValueError(f"Failed to embed signatures: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 embedded_signer.py <input_file> <private_key_file> [output_file]")
        print("\nCreates a self-contained signed media file with embedded authentication.")
        print("The signed file can be shared and verified without separate signature files.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    private_key_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        signed_file = create_signed_media(input_file, private_key_file, output_file)
        print(f"\n🎉 Success! Signed media ready: {signed_file}")
        print("   This file contains embedded quantum-resistant signatures.")
        print("   Share this file - viewers can verify it directly!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
