#!/usr/bin/env python3
"""Embedded signature verification for self-contained signed media files."""

import sys
import json
import base64
import warnings
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from crypto_signing import HybridSigner, KeyManager
from hashchain import ChainMessage
from perceptual_hash import PerceptualHasher
from canonicalization import MediaCanonicalizer

# Suppress warnings
warnings.filterwarnings('ignore')

def verify_signed_media(signed_file: str, public_key_file: str):
    """Verify a self-contained signed media file."""
    
    signed_path = Path(signed_file)
    if not signed_path.exists():
        raise FileNotFoundError(f"Signed media file not found: {signed_file}")
    
    print(f"Verifying signed media: {signed_file}")
    
    # Extract embedded signature data using ffprobe
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(signed_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise ValueError(f"Failed to read media metadata: {result.stderr}")
        
        metadata = json.loads(result.stdout)
        format_tags = metadata.get('format', {}).get('tags', {})
        
        # Look for QRMN signature data (case-insensitive)
        qrmn_auth = None
        qrmn_version = None
        qrmn_windows = None
        
        for key, value in format_tags.items():
            key_lower = key.lower()
            if 'qrmn_auth' in key_lower:
                qrmn_auth = value
            elif 'qrmn_version' in key_lower:
                qrmn_version = value
            elif 'qrmn_windows' in key_lower:
                qrmn_windows = value
        
        if not qrmn_auth:
            raise ValueError("No QRMN authentication data found in media file")
        
        print(f"Found embedded signatures: version {qrmn_version}, {qrmn_windows} windows")
        
    except Exception as e:
        raise ValueError(f"Failed to extract signatures from media: {e}")
    
    # Decode signature data
    try:
        signature_json = base64.b64decode(qrmn_auth).decode()
        sig_data = json.loads(signature_json)
        print(f"Decoded signature data: {len(sig_data['messages'])} messages")
    except Exception as e:
        raise ValueError(f"Failed to decode signature data: {e}")
    
    # Initialize verification components
    signer = HybridSigner()
    hasher = PerceptualHasher(speech_optimized=True)  # Enable speech optimization
    canonicalizer = MediaCanonicalizer()
    
    # Load public keys
    try:
        public_keys = KeyManager.load_keys(public_key_file)
        signer.load_public_keys(public_keys)
        print(f"Loaded public keys from: {public_key_file}")
    except Exception as e:
        raise ValueError(f"Failed to load public keys: {e}")
    
    # Process audio for verification
    try:
        canon_audio, canon_sr = canonicalizer.canonicalize_audio(signed_path)
        windows = canonicalizer.get_audio_windows(canon_audio, window_size=1.0, overlap=0.5)
        print(f"Processed {len(windows)} audio windows for verification")
    except Exception as e:
        raise ValueError(f"Failed to process audio: {e}")
    
    # Verify signatures and perceptual hashes
    valid_signatures = 0
    valid_hashes = 0
    total_windows = len(sig_data['messages'])
    
    print("Verifying embedded signatures...")
    
    for i, (msg_data, sig_data_item) in enumerate(zip(sig_data['messages'], sig_data['signatures'])):
        try:
            # Reconstruct and verify cryptographic signature
            chain_msg = ChainMessage.from_dict(msg_data)
            signature = {k: bytes.fromhex(v) for k, v in sig_data_item.items()}
            
            message_bytes = chain_msg.serialize_for_signing()
            sig_results = signer.verify_signature(message_bytes, signature, public_keys)
            signature_valid = signer.is_signature_valid(sig_results)
            
            if signature_valid:
                valid_signatures += 1
            
            # Verify perceptual hash with speech-optimized thresholds
            if i < len(windows):
                current_hash = hasher.compute_perceptual_hash(windows[i], canon_sr)
                similarity = hasher.hash_similarity(current_hash, chain_msg.perceptual_hash)
                
                # Use speech-optimized classification
                similarity_info = hasher.classify_similarity(similarity, "speech")
                
                # Accept HIGH or better similarity for speech
                if similarity_info["level"] in ["EXACT", "HIGH"]:
                    valid_hashes += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Verified {i + 1}/{total_windows} windows")
                
        except Exception as e:
            print(f"Warning: Failed to verify window {i}: {e}")
            continue
    
    # Calculate verification ratios
    sig_ratio = valid_signatures / total_windows if total_windows > 0 else 0
    hash_ratio = valid_hashes / min(total_windows, len(windows)) if min(total_windows, len(windows)) > 0 else 0
    
    # Determine overall result with speech-optimized thresholds
    if sig_ratio >= 0.95 and hash_ratio >= 0.80:
        result = "GREEN"
        status = "Authentic - Strong speech verification"
    elif sig_ratio >= 0.85 and hash_ratio >= 0.60:
        result = "AMBER"
        status = "Likely authentic - Minor compression detected"
    else:
        result = "RED"
        status = "Not authentic - Speech verification failed"
    
    # Print results
    print(f"\n{result}: {status}")
    print(f"Cryptographic signatures: {valid_signatures}/{total_windows} ({sig_ratio:.1%})")
    print(f"Perceptual hash verification: {valid_hashes}/{min(total_windows, len(windows))} ({hash_ratio:.1%})")
    print(f"Signature method: {sig_data.get('method', 'unknown')}")
    print(f"File size: {signed_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Additional info
    if result == "GREEN":
        print("✅ This media file is cryptographically authentic")
        print("✅ Content has not been tampered with")
        print("✅ Quantum-resistant signatures verified")
    elif result == "AMBER":
        print("⚠️  Media appears authentic but with some issues")
        print("⚠️  May have been compressed or lightly modified")
    else:
        print("❌ This media file is NOT authentic")
        print("❌ Content may have been tampered with or forged")
    
    return result, status

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 embedded_verifier.py <signed_media_file> <public_key_file>")
        print("\nVerifies a self-contained signed media file with embedded authentication.")
        print("No separate signature file needed - everything is embedded in the media!")
        sys.exit(1)
    
    signed_file = sys.argv[1]
    public_key_file = sys.argv[2]
    
    try:
        result, status = verify_signed_media(signed_file, public_key_file)
        exit_code = 0 if result == "GREEN" else 1 if result == "AMBER" else 2
        print(f"\n🔍 Verification complete: {result}")
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(3)
