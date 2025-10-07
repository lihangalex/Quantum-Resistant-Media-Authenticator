#!/usr/bin/env python3
"""External signature system that works with any content complexity."""

import sys
import json
import hashlib
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from crypto_signing import HybridSigner, KeyManager
from hashchain import HashChain
from perceptual_hash import PerceptualHasher
from canonicalization import MediaCanonicalizer
from video_perceptual_hash import VideoPerceptualHasher

# Suppress warnings
warnings.filterwarnings('ignore')

def create_external_signature(media_file: str, private_key_file: str, output_file: str = None):
    """Create external signature file for any media content."""
    
    media_path = Path(media_file)
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_file}")
    
    if output_file is None:
        output_file = str(media_path) + '.qrmn'
    
    print(f"Creating external signature for: {media_file}")
    
    # Calculate file hash for binding
    with open(media_file, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"File hash: {file_hash[:16]}...")
    
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
    
    # Process audio (canonicalized for hashing)
    try:
        canon_audio, canon_sr = canonicalizer.canonicalize_audio(media_path)
        print(f"Canonicalized audio: {len(canon_audio)} samples at {canon_sr}Hz")
        
        # Get windows for processing
        windows = canonicalizer.get_audio_windows(canon_audio, window_size=1.0, overlap=0.5)
        print(f"Created {len(windows)} windows for processing")
        
    except Exception as e:
        raise ValueError(f"Failed to process audio: {e}")
    
    # Create hash chain and signatures
    chain = HashChain()
    messages = []
    signatures = []
    
    print("Creating signatures...")
    for i, window in enumerate(windows):
        try:
            # Compute perceptual hash
            perceptual_hash = hasher.compute_perceptual_hash(window, canon_sr)
            
            # Add to chain
            metadata = {
                'method': 'external',
                'window_index': i,
                'timestamp': i * 0.5  # 50% overlap
            }
            chain_message = chain.add_window(i, perceptual_hash, metadata)
            
            # Sign the message
            message_bytes = chain_message.serialize_for_signing()
            signature = signer.sign_message(message_bytes)
            
            # Store results
            messages.append(chain_message.to_dict())
            signatures.append({k: v.hex() for k, v in signature.items()})
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(windows)} windows")
                
        except Exception as e:
            print(f"Warning: Failed to process window {i}: {e}")
            continue
    
    if not messages:
        raise ValueError("No windows were successfully processed")
    
    # Process video keyframes
    video_messages = []
    video_signatures = []
    video_hasher = VideoPerceptualHasher(keyframe_interval=2.0)
    
    print("Processing video keyframes...")
    try:
        keyframe_hashes = video_hasher.compute_keyframe_hashes(media_path)
        print(f"Extracted {len(keyframe_hashes)} video keyframes")
        
        for i, (frame_idx, phash, timestamp) in enumerate(keyframe_hashes):
            # Add to chain
            metadata = {
                'method': 'external',
                'frame_index': frame_idx,
                'timestamp': timestamp,
                'type': 'video'
            }
            chain_message = chain.add_window(i + len(messages), phash, metadata)
            
            # Sign the message
            message_bytes = chain_message.serialize_for_signing()
            signature = signer.sign_message(message_bytes)
            
            # Store results
            video_messages.append(chain_message.to_dict())
            video_signatures.append({k: v.hex() for k, v in signature.items()})
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(keyframe_hashes)} keyframes")
    
    except Exception as e:
        print(f"Warning: Video processing failed: {e}")
        print("Continuing with audio-only signatures...")
    
    # Create external signature data
    external_data = {
        'version': '2.0',
        'method': 'external',
        'file_hash': file_hash,
        'media_file': media_path.name,
        'total_windows': len(messages),
        'messages': messages,
        'signatures': signatures,
        'audio_info': {
            'sample_rate': canon_sr,
            'duration': len(canon_audio) / canon_sr,
            'channels': 1
        },
        'video_keyframes': len(video_messages),
        'video_messages': video_messages,
        'video_signatures': video_signatures
    }
    
    # Save signature file
    with open(output_file, 'w') as f:
        json.dump(external_data, f, indent=2)
    
    print(f"✅ External signatures saved to: {output_file}")
    print(f"   Audio: Signed {len(messages)} windows")
    print(f"   Video: Signed {len(video_messages)} keyframes")
    print(f"   File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 external_signer.py <media_file> <private_key_file> [output_file]")
        sys.exit(1)
    
    media_file = sys.argv[1]
    private_key_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        create_external_signature(media_file, private_key_file, output_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
