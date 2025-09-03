"""Hybrid cryptographic signing supporting classical and post-quantum schemes."""

from typing import Optional, Tuple, Dict, Any
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
import os
import json

try:
    # Optional post-quantum crypto support
    import pqcrypto.sign.dilithium3 as dilithium
    PQC_AVAILABLE = True
except ImportError:
    PQC_AVAILABLE = False


class HybridSigner:
    """Hybrid signer supporting classical Ed25519 and optional post-quantum Dilithium."""
    
    def __init__(self, use_pqc: bool = False):
        """Initialize signer with optional post-quantum support."""
        self.use_pqc = use_pqc and PQC_AVAILABLE
        self.ed25519_private_key: Optional[ed25519.Ed25519PrivateKey] = None
        self.ed25519_public_key: Optional[ed25519.Ed25519PublicKey] = None
        self.pqc_private_key: Optional[bytes] = None
        self.pqc_public_key: Optional[bytes] = None
    
    def generate_keys(self) -> Tuple[Dict[str, bytes], Dict[str, bytes]]:
        """Generate key pairs for signing."""
        # Generate Ed25519 keys
        self.ed25519_private_key = ed25519.Ed25519PrivateKey.generate()
        self.ed25519_public_key = self.ed25519_private_key.public_key()
        
        # Serialize Ed25519 keys
        ed25519_private_pem = self.ed25519_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        ed25519_public_pem = self.ed25519_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        private_keys = {'ed25519': ed25519_private_pem}
        public_keys = {'ed25519': ed25519_public_pem}
        
        # Generate post-quantum keys if enabled
        if self.use_pqc:
            self.pqc_public_key, self.pqc_private_key = dilithium.keypair()
            private_keys['dilithium3'] = self.pqc_private_key
            public_keys['dilithium3'] = self.pqc_public_key
        
        return private_keys, public_keys
    
    def load_private_keys(self, private_keys: Dict[str, bytes]) -> None:
        """Load private keys from dictionary."""
        # Load Ed25519 private key
        self.ed25519_private_key = serialization.load_pem_private_key(
            private_keys['ed25519'], password=None
        )
        self.ed25519_public_key = self.ed25519_private_key.public_key()
        
        # Load post-quantum private key if available
        if 'dilithium3' in private_keys and self.use_pqc:
            self.pqc_private_key = private_keys['dilithium3']
    
    def load_public_keys(self, public_keys: Dict[str, bytes]) -> None:
        """Load public keys from dictionary."""
        # Load Ed25519 public key
        self.ed25519_public_key = serialization.load_pem_public_key(
            public_keys['ed25519']
        )
        
        # Load post-quantum public key if available
        if 'dilithium3' in public_keys and self.use_pqc:
            self.pqc_public_key = public_keys['dilithium3']
    
    def sign_message(self, message: bytes) -> Dict[str, bytes]:
        """Sign message with hybrid approach."""
        if not self.ed25519_private_key:
            raise ValueError("No private keys loaded")
        
        signatures = {}
        
        # Sign with Ed25519
        ed25519_signature = self.ed25519_private_key.sign(message)
        signatures['ed25519'] = ed25519_signature
        
        # Sign with post-quantum if enabled
        if self.use_pqc and self.pqc_private_key:
            pqc_signature = dilithium.sign(self.pqc_private_key, message)
            signatures['dilithium3'] = pqc_signature
        
        return signatures
    
    def verify_signature(self, message: bytes, signatures: Dict[str, bytes], 
                        public_keys: Dict[str, bytes]) -> Dict[str, bool]:
        """Verify signatures against message."""
        results = {}
        
        # Verify Ed25519 signature
        try:
            ed25519_public = serialization.load_pem_public_key(public_keys['ed25519'])
            ed25519_public.verify(signatures['ed25519'], message)
            results['ed25519'] = True
        except Exception:
            results['ed25519'] = False
        
        # Verify post-quantum signature if present
        if 'dilithium3' in signatures and 'dilithium3' in public_keys:
            try:
                # Dilithium verification returns the message if valid, raises exception if not
                verified_message = dilithium.open(
                    signatures['dilithium3'], 
                    public_keys['dilithium3']
                )
                results['dilithium3'] = verified_message == message
            except Exception:
                results['dilithium3'] = False
        
        return results
    
    def is_signature_valid(self, verification_results: Dict[str, bool]) -> bool:
        """Check if signature verification passed required schemes."""
        # Require Ed25519 to pass (always available)
        if not verification_results.get('ed25519', False):
            return False
        
        # If post-quantum was used, require it to pass too
        if 'dilithium3' in verification_results:
            return verification_results['dilithium3']
        
        return True


class KeyManager:
    """Utility for managing cryptographic keys."""
    
    @staticmethod
    def save_keys(private_keys: Dict[str, bytes], public_keys: Dict[str, bytes], 
                  private_path: str, public_path: str) -> None:
        """Save keys to files."""
        # Convert bytes to hex for JSON serialization
        private_data = {k: v.hex() for k, v in private_keys.items()}
        public_data = {k: v.hex() for k, v in public_keys.items()}
        
        with open(private_path, 'w') as f:
            json.dump(private_data, f, indent=2)
        
        with open(public_path, 'w') as f:
            json.dump(public_data, f, indent=2)
        
        # Set restrictive permissions on private key file
        os.chmod(private_path, 0o600)
    
    @staticmethod
    def load_keys(key_path: str) -> Dict[str, bytes]:
        """Load keys from file."""
        with open(key_path, 'r') as f:
            data = json.load(f)
        
        # Convert hex back to bytes
        return {k: bytes.fromhex(v) for k, v in data.items()}
