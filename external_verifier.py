#!/usr/bin/env python3
"""External signature verification system."""

import sys
import json
import hashlib
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from crypto_signing import HybridSigner, KeyManager
from hashchain import ChainMessage
from perceptual_hash import PerceptualHasher
from canonicalization import MediaCanonicalizer

# Suppress warnings
warnings.filterwarnings('ignore')

def verify_external_signature(media_file: str, signature_file: str, public_key_file: str):
    """Verify external signature file."""
    
    media_path = Path(media_file)
    sig_path = Path(signature_file)
    
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_file}")
    if not sig_path.exists():
        raise FileNotFoundError(f"Signature file not found: {signature_file}")
    
    print(f"Verifying: {media_file}")
    print(f"Using signatures: {signature_file}")
    
    # Load signature data
    try:
        with open(signature_file, 'r') as f:
            sig_data = json.load(f)
        print(f"Loaded signature data: {len(sig_data['messages'])} windows")
    except Exception as e:
        raise ValueError(f"Failed to load signature file: {e}")
    
    # Verify file binding (integrity check)
    try:
        with open(media_file, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        
        stored_hash = sig_data.get('file_hash')
        if current_hash != stored_hash:
            print("❌ CRITICAL: File hash mismatch - file has been modified")
            return "RED", "File integrity compromised"
        else:
            print("✅ File integrity verified (hash match)")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify file hash: {e}")
    
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
        canon_audio, canon_sr = canonicalizer.canonicalize_audio(media_path)
        windows = canonicalizer.get_audio_windows(canon_audio, window_size=1.0, overlap=0.5)
        print(f"Processed {len(windows)} windows from audio")
    except Exception as e:
        raise ValueError(f"Failed to process audio: {e}")
    
    # Verify signatures and perceptual hashes
    valid_signatures = 0
    valid_hashes = 0
    total_windows = len(sig_data['messages'])
    
    print("Verifying signatures and perceptual hashes...")
    
    for i, (msg_data, sig_data_item) in enumerate(zip(sig_data['messages'], sig_data['signatures'])):
        try:
            # Reconstruct message
            chain_msg = ChainMessage.from_dict(msg_data)
            signature = {k: bytes.fromhex(v) for k, v in sig_data_item.items()}
            
            # Verify cryptographic signature
            message_bytes = chain_msg.serialize_for_signing()
            sig_results = signer.verify_signature(message_bytes, signature, public_keys)
            signature_valid = signer.is_signature_valid(sig_results)
            
            if signature_valid:
                valid_signatures += 1
            
            # Verify perceptual hash (if we have the corresponding window)
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
    
    # Determine overall result
    if sig_ratio >= 0.9 and hash_ratio >= 0.7:
        result = "GREEN"
        status = "Authentic - Strong verification"
    elif sig_ratio >= 0.7 and hash_ratio >= 0.5:
        result = "AMBER"
        status = "Likely authentic - Some verification issues"
    else:
        result = "RED"
        status = "Not authentic - Verification failed"
    
    # Print results
    print(f"\n{result}: {status}")
    print(f"Signature verification: {valid_signatures}/{total_windows} ({sig_ratio:.1%})")
    print(f"Perceptual hash verification: {valid_hashes}/{min(total_windows, len(windows))} ({hash_ratio:.1%})")
    print(f"Method: {sig_data.get('method', 'unknown')}")
    print(f"Original file: {sig_data.get('media_file', 'unknown')}")
    
    return result, status

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 external_verifier.py <media_file> <signature_file> <public_key_file>")
        sys.exit(1)
    
    media_file = sys.argv[1]
    signature_file = sys.argv[2]
    public_key_file = sys.argv[3]
    
    try:
        result, status = verify_external_signature(media_file, signature_file, public_key_file)
        exit_code = 0 if result == "GREEN" else 1 if result == "AMBER" else 2
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(3)
