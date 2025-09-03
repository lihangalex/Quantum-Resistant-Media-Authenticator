#!/usr/bin/env python3
"""Example usage of the quantum-resistant media notarizer system."""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from canonicalization import MediaCanonicalizer
from perceptual_hash import PerceptualHasher
from hashchain import HashChain
from crypto_signing import HybridSigner, KeyManager
from embedding import AudioEmbedder


def create_test_audio(filename: str, duration: float = 5.0, sample_rate: int = 44100):
    """Create a test audio file suitable for steganography."""
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Create a simple, stable audio signal that works well with LSB embedding
    # Use a single sine wave with some gentle modulation
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
    
    # Add very gentle amplitude modulation to make it more natural
    mod_freq = 2.0  # 2 Hz modulation
    audio = audio * (0.8 + 0.2 * np.sin(2 * np.pi * mod_freq * t))
    
    # Normalize to a safe level for embedding
    audio = audio * 0.5
    
    sf.write(filename, audio, sample_rate)
    print(f"Created test audio file: {filename} ({duration}s)")


def demonstrate_signing_and_verification():
    """Demonstrate the complete signing and verification process."""
    print("=== Quantum-Resistant Media Notarizer Demo ===\n")
    
    # Create test audio
    test_audio_file = "test_audio.wav"
    create_test_audio(test_audio_file, duration=3.0)
    
    try:
        # 1. Generate keys
        print("1. Generating cryptographic keys...")
        signer = HybridSigner(use_pqc=False)  # Classical only for demo
        private_keys, public_keys = signer.generate_keys()
        
        # Save keys
        KeyManager.save_keys(private_keys, public_keys, "demo_private.json", "demo_public.json")
        print("   Keys saved to demo_private.json and demo_public.json")
        
        # 2. Initialize components
        print("\n2. Initializing system components...")
        canonicalizer = MediaCanonicalizer(target_sr=44100)
        hasher = PerceptualHasher()
        embedder = AudioEmbedder()
        
        # 3. Load audio for embedding and canonicalization
        print("\n3. Loading audio...")
        embedding_audio, sr = sf.read(test_audio_file)
        canon_audio, canon_sr = canonicalizer.canonicalize_audio(Path(test_audio_file))
        print(f"   Audio length: {len(embedding_audio)/sr:.2f} seconds")
        print(f"   Sample rate: {sr} Hz")
        
        # 4. Split into windows
        print("\n4. Splitting audio into windows...")
        canon_windows = canonicalizer.get_audio_windows(canon_audio, window_size=1.0, overlap=0.5)
        temp_canonicalizer = MediaCanonicalizer(target_sr=sr)
        embedding_windows = temp_canonicalizer.get_audio_windows(embedding_audio, window_size=1.0, overlap=0.5)
        print(f"   Created {len(canon_windows)} overlapping windows")
        
        # 5. Create hash chain
        print("\n5. Building cryptographic hash chain...")
        chain = HashChain()
        messages = []
        signatures = []
        
        metadata = {
            'demo': True,
            'original_file': test_audio_file,
            'processing_params': {'window_size': 1.0, 'overlap': 0.5}
        }
        
        for i, (canon_window, embed_window) in enumerate(zip(canon_windows, embedding_windows)):
            # Compute perceptual hash from canonicalized window
            perceptual_hash = hasher.compute_perceptual_hash(canon_window, canon_sr)
            
            # Add to chain
            chain_message = chain.add_window(i, perceptual_hash, metadata.copy())
            
            # Sign the message
            message_bytes = chain_message.serialize_for_signing()
            signature = signer.sign_message(message_bytes)
            
            messages.append(chain_message.to_dict())
            signatures.append(signature)
        
        print(f"   Generated {len(signatures)} signatures")
        
        # 6. Verify chain integrity
        print("\n6. Verifying chain integrity...")
        chain_valid = chain.verify_chain_integrity()
        print(f"   Chain integrity: {'PASS' if chain_valid else 'FAIL'}")
        
        # 7. Embed signatures
        print("\n7. Embedding signatures into audio...")
        capacity = embedder.estimate_capacity(len(embedding_audio))
        print(f"   Embedding capacity: {capacity} bytes")
        
        signed_audio = embedder.embed_messages(embedding_audio, messages, signatures)
        
        # Save signed audio using 32-bit float to preserve embedded data
        signed_filename = "test_audio_signed.wav"
        sf.write(signed_filename, signed_audio, sr, subtype='FLOAT')
        print(f"   Signed audio saved to: {signed_filename}")
        
        # 8. Verification process
        print("\n8. Verifying signed audio...")
        
        # Load signed audio WITHOUT canonicalization to preserve embedded data
        verification_audio, verification_sr = sf.read(signed_filename)
        
        # Extract embedded messages
        extracted_messages, extracted_signatures = embedder.extract_messages(verification_audio)
        print(f"   Extracted {len(extracted_messages)} messages and {len(extracted_signatures)} signatures")
        
        # Verify signatures
        verification_signer = HybridSigner(use_pqc=False)
        verification_signer.load_public_keys(public_keys)
        
        valid_count = 0
        total_similarity = 0.0
        
        verification_windows = canonicalizer.get_audio_windows(verification_audio, 1.0, 0.5)
        
        for i, (msg_data, signature) in enumerate(zip(extracted_messages, extracted_signatures)):
            # Reconstruct chain message
            from hashchain import ChainMessage
            chain_msg = ChainMessage.from_dict(msg_data)
            
            # Verify signature
            message_bytes = chain_msg.serialize_for_signing()
            sig_results = verification_signer.verify_signature(message_bytes, signature, public_keys)
            signature_valid = verification_signer.is_signature_valid(sig_results)
            
            # Verify perceptual hash
            if i < len(verification_windows):
                current_hash = hasher.compute_perceptual_hash(verification_windows[i], verification_sr)
                similarity = hasher.hash_similarity(current_hash, chain_msg.perceptual_hash)
                total_similarity += similarity
                
                window_valid = signature_valid and similarity > 0.8
                if window_valid:
                    valid_count += 1
                
                print(f"   Window {i:2d}: {'PASS' if window_valid else 'FAIL'} "
                     f"(sig: {signature_valid}, hash: {similarity:.3f})")
        
        # Final verification result
        valid_ratio = valid_count / len(extracted_messages) if extracted_messages else 0
        avg_similarity = total_similarity / len(extracted_messages) if extracted_messages else 0
        
        print(f"\n9. Verification Results:")
        print(f"   Valid windows: {valid_count}/{len(extracted_messages)} ({valid_ratio:.1%})")
        print(f"   Average similarity: {avg_similarity:.3f}")
        
        if valid_ratio >= 0.9:
            result = "GREEN - Full verification passed"
        elif valid_ratio >= 0.5:
            result = "AMBER - Partial verification"
        else:
            result = "RED - Verification failed"
        
        print(f"   Overall result: {result}")
        
        # Cleanup
        print(f"\n10. Cleaning up temporary files...")
        Path(test_audio_file).unlink(missing_ok=True)
        Path(signed_filename).unlink(missing_ok=True)
        Path("demo_private.json").unlink(missing_ok=True)
        Path("demo_public.json").unlink(missing_ok=True)
        
        print("\n=== Demo completed successfully ===")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_signing_and_verification()
